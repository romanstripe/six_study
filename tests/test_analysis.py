from __future__ import annotations

import codecs
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import (
    AFFECTED_END,
    AFFECTED_START,
    EXPECTED_AFFECTED_LOTS,
    Inputs,
    affected_process_mask,
    boundary_lots,
    build_lot_summary,
    check_data_quality,
    correct_metrology,
    load_inputs,
    make_core_summary,
    require_columns,
    validate_results,
    write_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPLY_PATH = ROOT / "REPLY_고객A.md"


@pytest.fixture(scope="session")
def analysis_result() -> dict[str, object]:
    inputs = load_inputs(DATA_DIR)
    quality = check_data_quality(inputs)
    metrology = correct_metrology(inputs.metrology)
    summary = build_lot_summary(inputs, metrology)
    validate_results(summary)
    boundaries = boundary_lots(summary)
    core = make_core_summary(summary, boundaries)
    return {
        "inputs": inputs,
        "quality": quality,
        "metrology": metrology,
        "summary": summary,
        "boundaries": boundaries,
        "core": core,
    }


@pytest.fixture(scope="session")
def generated_outputs(
    tmp_path_factory: pytest.TempPathFactory,
    analysis_result: dict[str, object],
) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("generated")
    output_dir = base / "output"
    customer_output = base / "AFFECTED_LOTS_고객A.csv"
    write_outputs(
        output_dir,
        customer_output,
        analysis_result["quality"],
        analysis_result["metrology"],
        analysis_result["summary"],
        analysis_result["boundaries"],
        analysis_result["core"],
    )
    return {"base": base, "output": output_dir, "customer": customer_output}


def test_input_shape_and_data_quality(analysis_result: dict[str, object]) -> None:
    inputs: Inputs = analysis_result["inputs"]
    quality: pd.DataFrame = analysis_result["quality"]

    assert len(inputs.history) == 256
    assert inputs.history["lot_id"].nunique() == 64
    assert len(inputs.metrology) == 320
    assert inputs.metrology["lot_id"].nunique() == 64
    assert len(inputs.eds) == 64
    assert len(inputs.shipments) == 64
    assert quality["passed"].all()
    assert len(quality) == 14


def test_required_column_failure_is_explicit() -> None:
    frame = pd.DataFrame({"lot_id": ["L0001"]})
    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(frame, ["lot_id", "step"], "fixture.csv")


def test_duplicate_key_failure_is_explicit(analysis_result: dict[str, object]) -> None:
    inputs: Inputs = analysis_result["inputs"]
    duplicate_shipments = pd.concat(
        [inputs.shipments, inputs.shipments.iloc[[0]]], ignore_index=True
    )
    invalid = Inputs(
        inputs.history,
        inputs.metrology,
        inputs.eds,
        duplicate_shipments,
    )
    with pytest.raises(ValueError, match="duplicate keys"):
        check_data_quality(invalid)


def test_process_sequence_failure_is_explicit(analysis_result: dict[str, object]) -> None:
    inputs: Inputs = analysis_result["inputs"]
    invalid_history = inputs.history.copy()
    clean_index = invalid_history.index[
        (invalid_history["lot_id"] == "L6001")
        & (invalid_history["step"] == "CLEAN")
    ][0]
    invalid_history.loc[clean_index, "track_in_time"] = pd.Timestamp(
        "2026-10-05 07:00"
    )
    invalid = Inputs(
        invalid_history,
        inputs.metrology,
        inputs.eds,
        inputs.shipments,
    )
    with pytest.raises(ValueError, match="invalid process sequence"):
        check_data_quality(invalid)


def test_module_quantities_and_unshipped_date_are_parsed(
    analysis_result: dict[str, object],
) -> None:
    inputs: Inputs = analysis_result["inputs"]
    shipments = inputs.shipments.set_index("lot_id")
    assert shipments.loc["L6012", "module_qty"] == 2000
    assert shipments.loc["L6051", "module_qty"] == 2100
    assert pd.isna(shipments.loc["L6040", "ship_date"])


def test_mt02_correction_time_boundaries() -> None:
    measurements = pd.DataFrame(
        {
            "tool": ["MT-02", "MT-02", "MT-02", "MT-01"],
            "measure_time": pd.to_datetime(
                [
                    "2026-10-09 08:59",
                    "2026-10-09 09:00",
                    "2026-10-15 09:00",
                    "2026-10-10 12:00",
                ]
            ),
            "value_nm": [5.0, 5.0, 5.0, 5.0],
        }
    )
    corrected = correct_metrology(measurements)
    assert corrected["corrected_value_nm"].tolist() == [5.0, 4.85, 5.0, 5.0]
    assert corrected["value_nm"].tolist() == [5.0, 5.0, 5.0, 5.0]
    assert corrected.loc[1, "correction_status"] == (
        "calibration-record correction, provisional"
    )


def test_affected_process_time_boundaries() -> None:
    equipment = pd.Series(["DF-03", "DF-03", "DF-03", "DF-03", "DF-02"])
    process_time = pd.Series(
        pd.to_datetime(
            [
                AFFECTED_START - pd.Timedelta(minutes=1),
                AFFECTED_START,
                AFFECTED_END,
                AFFECTED_END + pd.Timedelta(minutes=1),
                AFFECTED_START,
            ]
        )
    )
    assert affected_process_mask(equipment, process_time).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_approved_affected_lots_and_totals(
    analysis_result: dict[str, object],
) -> None:
    summary: pd.DataFrame = analysis_result["summary"]
    affected = summary.loc[summary["is_affected"]]

    assert tuple(affected["lot_id"]) == EXPECTED_AFFECTED_LOTS
    assert len(affected) == 8
    assert affected["module_qty"].sum() == 16100

    grouped = affected.groupby("customer").agg(
        lots=("lot_id", "count"), qty=("module_qty", "sum")
    )
    assert grouped.loc["고객A"].to_dict() == {"lots": 5, "qty": 10200}
    assert grouped.loc["고객B"].to_dict() == {"lots": 2, "qty": 4100}
    assert grouped.loc["미출하"].to_dict() == {"lots": 1, "qty": 1800}


def test_boundary_lots(analysis_result: dict[str, object]) -> None:
    boundaries: pd.DataFrame = analysis_result["boundaries"]
    assert boundaries["lot_id"].tolist() == ["L6028", "L6059"]
    assert boundaries["boundary_role"].tolist() == [
        "last normal lot",
        "first normal lot after repair",
    ]


@pytest.mark.parametrize(
    ("lot_id", "raw_mean", "corrected_mean", "hot_fail"),
    [
        ("L6035", 4.884, 4.734, 1.72),
        ("L6039", 4.856, 4.706, 2.20),
        ("L6048", 4.858, 4.708, 2.34),
        ("L6040", 4.732, 4.732, 1.94),
    ],
)
def test_known_lot_calculations(
    analysis_result: dict[str, object],
    lot_id: str,
    raw_mean: float,
    corrected_mean: float,
    hot_fail: float,
) -> None:
    summary: pd.DataFrame = analysis_result["summary"]
    row = summary.set_index("lot_id").loc[lot_id]
    assert row["gox_raw_mean_nm"] == pytest.approx(raw_mean)
    assert row["gox_corrected_mean_nm"] == pytest.approx(corrected_mean)
    assert row["hot_refresh_fail_pct"] == pytest.approx(hot_fail)
    if lot_id == "L6040":
        assert row["metro_tool"] == "MT-01"
        assert row["corrected_point_count"] == 0


def test_l6035_full_check_values(analysis_result: dict[str, object]) -> None:
    summary: pd.DataFrame = analysis_result["summary"]
    row = summary.set_index("lot_id").loc["L6035"]
    assert row["gox_corrected_min_nm"] == pytest.approx(4.72)
    assert row["gox_corrected_max_nm"] == pytest.approx(4.75)
    assert row["module_qty"] == 2000
    assert row["customer"] == "고객A"


def test_customer_csv_contract(generated_outputs: dict[str, Path]) -> None:
    path = generated_outputs["customer"]
    raw = path.read_bytes()
    rows = pd.read_csv(path, dtype=str)

    assert path.name == "AFFECTED_LOTS_고객A.csv"
    assert not raw.startswith(codecs.BOM_UTF8)
    assert list(rows.columns) == ["LOT_ID", "SHIP_DATE", "MODULE_QTY"]
    assert len(rows) == 5
    assert rows["LOT_ID"].tolist() == [
        "L6035",
        "L6039",
        "L6048",
        "L6051",
        "L6056",
    ]
    assert rows["SHIP_DATE"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert rows["MODULE_QTY"].str.fullmatch(r"\d+").all()
    assert rows["MODULE_QTY"].astype(int).sum() == 10200
    assert "합계" not in path.read_text(encoding="utf-8")


def test_generated_figures_are_nonempty(generated_outputs: dict[str, Path]) -> None:
    figures = generated_outputs["output"] / "figures"
    for name in ("gox_eds_by_lot.png", "gox_vs_hot_refresh.png"):
        path = figures / name
        assert path.exists()
        assert path.stat().st_size > 10_000


def test_customer_reply_publication_scope() -> None:
    text = REPLY_PATH.read_text(encoding="utf-8")
    forbidden_patterns = [
        r"DF-\d+",
        r"MT-\d+",
        r"고객B",
        r"박도윤",
        r"오세린",
        r"010-\d{3,4}-\d{4}",
        r"내선",
        r"EDS",
        r"수율",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, text) is None, pattern

    assert "잠정 원인" in text
    for required_date in (
        "2026-11-04",
        "2026-10-14",
        "2026-10-15",
        "2026-11-01",
        "2026-11-15",
    ):
        assert required_date in text


def test_reproducible_tabular_outputs(
    tmp_path: Path,
    analysis_result: dict[str, object],
    generated_outputs: dict[str, Path],
) -> None:
    output_dir = tmp_path / "output"
    customer_output = tmp_path / "AFFECTED_LOTS_고객A.csv"
    write_outputs(
        output_dir,
        customer_output,
        analysis_result["quality"],
        analysis_result["metrology"],
        analysis_result["summary"],
        analysis_result["boundaries"],
        analysis_result["core"],
    )

    first_output = generated_outputs["output"]
    for relative_path in (
        "data_quality.csv",
        "metrology_corrected.csv",
        "lot_summary.csv",
        "affected_lots_internal.csv",
        "df03_boundary_lots.csv",
        "core_summary.json",
    ):
        assert (first_output / relative_path).read_bytes() == (
            output_dir / relative_path
        ).read_bytes()
    assert generated_outputs["customer"].read_bytes() == customer_output.read_bytes()


def test_core_summary_json_is_valid(generated_outputs: dict[str, Path]) -> None:
    path = generated_outputs["output"] / "core_summary.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["affected_lot_count"] == 8
    assert result["affected_module_qty"] == 16100
    assert result["boundary_lots"] == ["L6028", "L6059"]
