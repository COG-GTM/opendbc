import json
import textwrap
from pathlib import Path

from opendbc.car.ford.dbc_signal_xref import (
  build_xref_index,
  extract_packer_signals,
  extract_vl_accesses,
)


class TestExtractVlAccesses:
  def test_basic_vl_access(self, tmp_path: Path):
    src = tmp_path / "test_carstate.py"
    src.write_text(textwrap.dedent("""\
      ret.speed = cp.vl["BrakeSysFeatures"]["Veh_V_ActlBrk"]
      ret.yaw = cp.vl["Yaw_Data_FD1"]["VehYaw_W_Actl"]
    """))
    results = extract_vl_accesses(src)
    assert len(results) == 2
    assert results[0] == ("BrakeSysFeatures", "Veh_V_ActlBrk", 1, "speed")
    assert results[1] == ("Yaw_Data_FD1", "VehYaw_W_Actl", 2, "yaw")

  def test_cam_vl_access(self, tmp_path: Path):
    src = tmp_path / "test_cam.py"
    src.write_text(textwrap.dedent("""\
      ret.stockFcw = bool(cp_cam.vl["ACCDATA_3"]["FcwVisblWarn_B_Rq"])
    """))
    results = extract_vl_accesses(src)
    assert len(results) == 1
    assert results[0][0] == "ACCDATA_3"
    assert results[0][1] == "FcwVisblWarn_B_Rq"
    assert results[0][3] == "stockFcw"

  def test_conditional_usage(self, tmp_path: Path):
    src = tmp_path / "test_cond.py"
    src.write_text(textwrap.dedent("""\
      if cp.vl["EngBrakeData"]["CcStat_D_Actl"] in (4, 5):
        pass
    """))
    results = extract_vl_accesses(src)
    assert len(results) == 1
    assert results[0][3] == "condition"

  def test_self_assignment_usage(self, tmp_path: Path):
    src = tmp_path / "test_self.py"
    src.write_text(textwrap.dedent("""\
      self.distance_button = cp.vl["Steering_Data_FD1"]["AccButtnGapTogglePress"]
    """))
    results = extract_vl_accesses(src)
    assert len(results) == 1
    assert results[0][3] == "distance_button"

  def test_no_duplicates(self, tmp_path: Path):
    src = tmp_path / "test_nodup.py"
    src.write_text(textwrap.dedent("""\
      ret.speed = cp.vl["Msg"]["Sig"]
    """))
    results = extract_vl_accesses(src)
    # AST and regex should not produce duplicates
    assert len(results) == 1

  def test_empty_file(self, tmp_path: Path):
    src = tmp_path / "test_empty.py"
    src.write_text("")
    results = extract_vl_accesses(src)
    assert results == []

  def test_no_vl_accesses(self, tmp_path: Path):
    src = tmp_path / "test_novl.py"
    src.write_text(textwrap.dedent("""\
      x = some_dict["key"]["value"]
      y = foo.bar["baz"]
    """))
    results = extract_vl_accesses(src)
    assert results == []


class TestExtractPackerSignals:
  def test_basic_packer_values(self, tmp_path: Path):
    src = tmp_path / "test_packer.py"
    src.write_text(textwrap.dedent("""\
      def create_msg(packer, bus):
        values = {
          "Signal_A": 1,
          "Signal_B": 0.5,
        }
        return packer.make_can_msg("TestMessage", bus, values)
    """))
    results = extract_packer_signals(src)
    msg_sig_pairs = [(r[0], r[1]) for r in results]
    assert ("TestMessage", "Signal_A") in msg_sig_pairs
    assert ("TestMessage", "Signal_B") in msg_sig_pairs

  def test_stock_values_passthrough(self, tmp_path: Path):
    src = tmp_path / "test_stock.py"
    src.write_text(textwrap.dedent("""\
      def create_ui_msg(packer, bus, stock_values):
        values = {s: stock_values[s] for s in [
          "StockSig1",
          "StockSig2",
        ]}
        return packer.make_can_msg("UIMessage", bus, values)
    """))
    results = extract_packer_signals(src)
    msg_sig_pairs = [(r[0], r[1]) for r in results]
    assert ("UIMessage", "StockSig1") in msg_sig_pairs
    assert ("UIMessage", "StockSig2") in msg_sig_pairs

  def test_multiple_functions_scoped(self, tmp_path: Path):
    src = tmp_path / "test_scoped.py"
    src.write_text(textwrap.dedent("""\
      def create_msg_a(packer, bus):
        values = {"SigA": 1}
        return packer.make_can_msg("MsgA", bus, values)

      def create_msg_b(packer, bus):
        values = {"SigB": 2}
        return packer.make_can_msg("MsgB", bus, values)
    """))
    results = extract_packer_signals(src)
    msg_sig_pairs = [(r[0], r[1]) for r in results]
    assert ("MsgA", "SigA") in msg_sig_pairs
    assert ("MsgB", "SigB") in msg_sig_pairs
    # Signals should NOT be cross-associated
    assert ("MsgA", "SigB") not in msg_sig_pairs
    assert ("MsgB", "SigA") not in msg_sig_pairs

  def test_no_make_can_msg(self, tmp_path: Path):
    src = tmp_path / "test_nomsg.py"
    src.write_text(textwrap.dedent("""\
      def helper():
        values = {"SomeKey": 123}
        return values
    """))
    results = extract_packer_signals(src)
    assert results == []


class TestBuildXrefIndex:
  def test_real_ford_files(self):
    """Integration test using the actual Ford source files."""
    index = build_xref_index()
    assert len(index) > 0

    # All entries should have valid DBC metadata (size > 0)
    for key, entry in index.items():
      assert entry["size"] > 0, f"{key} has size 0 (not found in DBC)"
      assert entry["dbc"] == "ford_lincoln_base_pt"
      assert len(entry["code_refs"]) > 0

  def test_ticket_example_signal(self):
    """Verify the exact example from the ticket description."""
    index = build_xref_index()
    key = "SteeringPinion_Data.StePinCompAnEst_D_Qf"
    assert key in index, f"Expected {key} in index"

    entry = index[key]
    assert entry["dbc"] == "ford_lincoln_base_pt"
    assert any(
      ref["file"] == "carstate.py" and ref["line"] == 31 and ref["usage"] == "vehicleSensorsInvalid"
      for ref in entry["code_refs"]
    )
    # Verify DBC metadata is populated
    assert entry["size"] > 0

  def test_carstate_signals_present(self):
    """Verify that all expected carstate.py signals are indexed."""
    index = build_xref_index()

    expected_carstate_signals = [
      "SteeringPinion_Data.StePinCompAnEst_D_Qf",
      "BrakeSysFeatures.Veh_V_ActlBrk",
      "Yaw_Data_FD1.VehYaw_W_Actl",
      "EngVehicleSpThrottle.ApedPos_Pc_ActlArb",
      "EPAS_INFO.SteeringColumnTorque",
      "SteeringPinion_Data.StePinComp_An_Est",
      "EngBrakeData.CcStat_D_Actl",
      "Steering_Data_FD1.TurnLghtSwtch_D_Stat",
    ]
    for sig_key in expected_carstate_signals:
      assert sig_key in index, f"Missing expected signal: {sig_key}"

  def test_fordcan_packer_signals_present(self):
    """Verify that fordcan.py packer signals are indexed."""
    index = build_xref_index()

    expected_packer_signals = [
      "LateralMotionControl.LatCtlCurv_No_Actl",
      "ACCDATA.AccBrkTot_A_Rq",
      "ACCDATA.AccPrpl_A_Rq",
      "ACCDATA_3.Tja_D_Stat",
    ]
    for sig_key in expected_packer_signals:
      assert sig_key in index, f"Missing expected packer signal: {sig_key}"

  def test_output_is_valid_json(self):
    """Verify the index can be serialized to valid JSON."""
    index = build_xref_index()
    json_str = json.dumps(index, indent=2)
    parsed = json.loads(json_str)
    assert parsed == index

  def test_no_duplicate_code_refs(self):
    """Verify no entry has duplicate code_refs (same file+line)."""
    index = build_xref_index()
    for key, entry in index.items():
      seen = set()
      for ref in entry["code_refs"]:
        ref_key = (ref["file"], ref["line"])
        assert ref_key not in seen, f"Duplicate code_ref in {key}: {ref_key}"
        seen.add(ref_key)
