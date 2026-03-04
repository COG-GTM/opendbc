#!/usr/bin/env python3
"""
DBC signal-to-code cross-reference index builder for Ford.

Parses carstate.py and carcontroller.py to extract every cp.vl[<msg>][<signal>]
access and links each to the corresponding signal definition in the Ford DBC file.

Outputs a structured JSON mapping that lets an agent correlate raw CAN frame bits
with the code decisions they feed into.

Usage:
  python -m opendbc.car.ford.dbc_signal_xref [--output path/to/output.json]
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from opendbc.can.dbc import DBC, Signal

FORD_DIR = Path(__file__).parent
CARSTATE_FILE = FORD_DIR / "carstate.py"
CARCONTROLLER_FILE = FORD_DIR / "carcontroller.py"
FORDCAN_FILE = FORD_DIR / "fordcan.py"
DBC_NAME = "ford_lincoln_base_pt"


@dataclass
class CodeRef:
  file: str
  line: int
  usage: str


@dataclass
class SignalXref:
  dbc: str
  code_refs: list[CodeRef] = field(default_factory=list)
  start_bit: int = 0
  size: int = 0
  is_signed: bool = False
  factor: float = 0.0
  offset: float = 0.0
  is_little_endian: bool = False


def _relative_path(filepath: Path) -> str:
  """Return a short relative path for display (e.g. 'carstate.py')."""
  return filepath.name


def _infer_usage_from_context(line_text: str, signal_name: str) -> str:
  """Infer how a signal is used from the surrounding assignment or expression."""
  stripped = line_text.strip()

  # Handle assignments like: ret.fieldName = ...
  assign_match = re.match(r'(ret\.\w+)\s*[=|]', stripped)
  if assign_match:
    return assign_match.group(1).replace("ret.", "")

  # Handle augmented assignments like: ret.fieldName |= ...
  aug_assign_match = re.match(r'(ret\.\w+)\s*\|=', stripped)
  if aug_assign_match:
    return aug_assign_match.group(1).replace("ret.", "")

  # Handle self.field = ...
  self_match = re.match(r'(self\.\w+)\s*=', stripped)
  if self_match:
    return self_match.group(1).replace("self.", "")

  # Handle if conditions
  if stripped.startswith(("if ", "elif ")):
    return "condition"

  # Handle return or other contexts
  return signal_name


def extract_vl_accesses(filepath: Path) -> list[tuple[str, str, int, str]]:
  """
  Extract all cp.vl["MessageName"]["SignalName"] accesses from a Python file.

  Uses AST parsing to reliably find subscript access patterns on .vl attributes,
  then falls back to regex for patterns not easily captured by AST (e.g. multi-line).

  Returns list of (message_name, signal_name, line_number, usage_context).
  """
  source = filepath.read_text()
  lines = source.splitlines()
  results: list[tuple[str, str, int, str]] = []
  seen: set[tuple[str, str, int]] = set()

  # AST-based extraction for reliable parsing
  try:
    tree = ast.parse(source, filename=str(filepath))
    for node in ast.walk(tree):
      # Look for pattern: <expr>.vl["MsgName"]["SignalName"]
      if not isinstance(node, ast.Subscript):
        continue

      # The outer subscript should have a string key (signal name)
      signal_key = _extract_string_key(node.slice)
      if signal_key is None:
        continue

      # The value should be another subscript: <expr>.vl["MsgName"]
      inner = node.value
      if not isinstance(inner, ast.Subscript):
        continue

      msg_key = _extract_string_key(inner.slice)
      if msg_key is None:
        continue

      # The inner value should be an attribute access ending in .vl
      if not isinstance(inner.value, ast.Attribute) or inner.value.attr != "vl":
        continue

      line_num = node.lineno
      line_text = lines[line_num - 1] if line_num <= len(lines) else ""
      usage = _infer_usage_from_context(line_text, signal_key)

      key = (msg_key, signal_key, line_num)
      if key not in seen:
        seen.add(key)
        results.append((msg_key, signal_key, line_num, usage))
  except SyntaxError:
    pass

  # Regex fallback for any patterns missed by AST (e.g. in comments, multi-line)
  # Matches: cp.vl["MsgName"]["SignalName"] or cp_cam.vl["MsgName"]["SignalName"]
  vl_pattern = re.compile(
    r'\b\w+\.vl\[(["\'])(\w+)\1\]\[(["\'])(\w+)\3\]'
  )
  for line_num, line_text in enumerate(lines, start=1):
    for m in vl_pattern.finditer(line_text):
      msg_name = m.group(2)
      signal_name = m.group(4)
      key = (msg_name, signal_name, line_num)
      if key not in seen:
        seen.add(key)
        usage = _infer_usage_from_context(line_text, signal_name)
        results.append((msg_name, signal_name, line_num, usage))

  return results


def _extract_string_key(node: ast.expr) -> str | None:
  """Extract a string constant from an AST subscript slice."""
  if isinstance(node, ast.Constant) and isinstance(node.value, str):
    return node.value
  return None


def extract_packer_signals(filepath: Path) -> list[tuple[str, str, int, str]]:
  """
  Extract signal names from CANPacker make_can_msg calls in fordcan.py.

  For each function, finds make_can_msg calls and associates signal name keys
  from dict literals (values dicts and stock_values passthrough) with the correct
  CAN message name.

  Returns list of (message_name, signal_name, line_number, usage_context).
  """
  source = filepath.read_text()
  results: list[tuple[str, str, int, str]] = []
  seen: set[tuple[str, str, int]] = set()

  try:
    tree = ast.parse(source, filename=str(filepath))
  except SyntaxError:
    return results

  # Process each function definition independently to scope signals to their message
  for func_node in ast.walk(tree):
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
      continue

    # Find all make_can_msg calls in this function and their message names
    msg_names_in_func: list[str] = []
    for node in ast.walk(func_node):
      if not isinstance(node, ast.Call):
        continue
      if not (isinstance(node.func, ast.Attribute) and node.func.attr == "make_can_msg"):
        continue
      if not node.args:
        continue
      msg_name_node = node.args[0]
      if isinstance(msg_name_node, ast.Constant) and isinstance(msg_name_node.value, str):
        msg_names_in_func.append(msg_name_node.value)

    if not msg_names_in_func:
      continue

    # Use the last make_can_msg call's message name as the primary message for this function.
    # Most fordcan.py functions create exactly one message type (possibly multiple calls for
    # checksum computation). The last call is typically the final return value.
    primary_msg = msg_names_in_func[-1]

    # Extract signal names from all Dict nodes in this function
    for node in ast.walk(func_node):
      if isinstance(node, ast.Dict):
        for key_node in node.keys:
          if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            signal_name = key_node.value
            line_num = key_node.lineno
            key_tuple = (primary_msg, signal_name, line_num)
            if key_tuple not in seen:
              seen.add(key_tuple)
              results.append((primary_msg, signal_name, line_num, "packer"))

      # Also extract from List nodes inside stock_values passthrough comprehensions
      # Pattern: {s: stock_values[s] for s in ["Signal1", "Signal2", ...]}
      if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
        _extract_from_comprehension(node, primary_msg, results, seen)

    # Also extract from plain List nodes used in stock_values passthrough
    # Pattern: values = {s: stock_values[s] for s in ["Sig1", ...]}
    _extract_stock_values_from_func(func_node, primary_msg, results, seen)

  return results


def _extract_from_comprehension(
  node: ast.expr, msg_name: str,
  results: list[tuple[str, str, int, str]],
  seen: set[tuple[str, str, int]],
) -> None:
  """Extract signal names from comprehension iterables (e.g. for s in [...])."""
  iters: list[ast.expr] = []
  if isinstance(node, ast.DictComp):
    iters = [comp.iter for comp in node.generators]
  elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
    iters = [comp.iter for comp in node.generators]

  for iter_node in iters:
    if isinstance(iter_node, ast.List):
      for elt in iter_node.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
          signal_name = elt.value
          line_num = elt.lineno
          key_tuple = (msg_name, signal_name, line_num)
          if key_tuple not in seen:
            seen.add(key_tuple)
            results.append((msg_name, signal_name, line_num, "stock_passthrough"))


def _extract_stock_values_from_func(
  func_node: ast.FunctionDef | ast.AsyncFunctionDef, msg_name: str,
  results: list[tuple[str, str, int, str]],
  seen: set[tuple[str, str, int]],
) -> None:
  """Extract signal names from stock_values dict comprehension patterns in a function."""
  for node in ast.walk(func_node):
    # Match: {s: stock_values[s] for s in ["Sig1", "Sig2", ...]}
    if isinstance(node, ast.DictComp):
      for comp in node.generators:
        if isinstance(comp.iter, ast.List):
          for elt in comp.iter.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
              signal_name = elt.value
              line_num = elt.lineno
              key_tuple = (msg_name, signal_name, line_num)
              if key_tuple not in seen:
                seen.add(key_tuple)
                results.append((msg_name, signal_name, line_num, "stock_passthrough"))


def build_xref_index(
  carstate_path: Path = CARSTATE_FILE,
  carcontroller_path: Path = CARCONTROLLER_FILE,
  fordcan_path: Path = FORDCAN_FILE,
  dbc_name: str = DBC_NAME,
) -> dict[str, dict]:
  """
  Build the complete cross-reference index.

  Parses carstate.py, carcontroller.py, and fordcan.py for signal accesses,
  then enriches each with DBC signal metadata.

  Returns a dict keyed by "MessageName.SignalName" with structure:
    {
      "dbc": str,
      "code_refs": [{"file": str, "line": int, "usage": str}, ...],
      "start_bit": int,
      "size": int,
      "is_signed": bool,
      "factor": float,
      "offset": float,
      "is_little_endian": bool
    }
  """
  dbc = DBC(dbc_name)

  # Collect all signal accesses
  accesses: list[tuple[str, str, int, str, Path]] = []

  # From carstate.py - direct cp.vl accesses
  for msg, sig, line, usage in extract_vl_accesses(carstate_path):
    accesses.append((msg, sig, line, usage, carstate_path))

  # From carcontroller.py - direct cp.vl accesses (e.g. CS.acc_tja_status_stock_values refs)
  for msg, sig, line, usage in extract_vl_accesses(carcontroller_path):
    accesses.append((msg, sig, line, usage, carcontroller_path))

  # From fordcan.py - packer signal references
  for msg, sig, line, usage in extract_packer_signals(fordcan_path):
    accesses.append((msg, sig, line, usage, fordcan_path))

  # Build the index
  index: dict[str, SignalXref] = {}

  for msg_name, signal_name, line_num, usage, filepath in accesses:
    key = f"{msg_name}.{signal_name}"

    if key not in index:
      # Look up signal metadata from DBC
      dbc_signal = _lookup_signal(dbc, msg_name, signal_name)
      if dbc_signal is not None:
        index[key] = SignalXref(
          dbc=dbc_name,
          start_bit=dbc_signal.start_bit,
          size=dbc_signal.size,
          is_signed=dbc_signal.is_signed,
          factor=dbc_signal.factor,
          offset=dbc_signal.offset,
          is_little_endian=dbc_signal.is_little_endian,
        )
      else:
        # Signal not found in DBC - still record the code reference
        index[key] = SignalXref(dbc=dbc_name)

    index[key].code_refs.append(CodeRef(
      file=_relative_path(filepath),
      line=line_num,
      usage=usage,
    ))

  # Convert to plain dicts for JSON serialization
  result: dict[str, dict] = {}
  for key, xref in sorted(index.items()):
    entry = asdict(xref)
    # Deduplicate code_refs (same file+line might appear from AST + regex)
    unique_refs: list[dict] = []
    seen_refs: set[tuple[str, int]] = set()
    for ref in entry["code_refs"]:
      ref_key = (ref["file"], ref["line"])
      if ref_key not in seen_refs:
        seen_refs.add(ref_key)
        unique_refs.append(ref)
    entry["code_refs"] = unique_refs
    result[key] = entry

  return result


def _lookup_signal(dbc: DBC, msg_name: str, signal_name: str) -> Signal | None:
  """Look up a signal in the DBC by message name and signal name."""
  msg = dbc.name_to_msg.get(msg_name)
  if msg is None:
    return None
  return msg.sigs.get(signal_name)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Build DBC signal-to-code cross-reference index for Ford"
  )
  parser.add_argument(
    "--output", "-o",
    type=str,
    default=None,
    help="Output JSON file path (default: stdout)",
  )
  args = parser.parse_args()

  index = build_xref_index()

  output = json.dumps(index, indent=2)
  if args.output:
    Path(args.output).write_text(output + "\n")
    print(f"Wrote cross-reference index to {args.output} ({len(index)} signals)", file=sys.stderr)
  else:
    print(output)


if __name__ == "__main__":
  main()
