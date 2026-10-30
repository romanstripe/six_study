from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


AFFECTED_START = pd.Timestamp("2026-10-10 08:00")
AFFECTED_END = pd.Timestamp("2026-10-14 20:00")
MT02_CORRECTION_START = pd.Timestamp("2026-10-09 09:00")
MT02_CORRECTION_END = pd.Timestamp("2026-10-15 09:00")
MT02_OFFSET_NM = 0.15
EXPECTED_AFFECTED_LOTS = (
    "L6035",
    "L6039",
    "L6040",
    "L6043",
    "L6048",
    "L6051",
    "L6055",
    "L6056",
)


@dataclass(frozen=True)
class Inputs:
    history: pd.DataFrame
    metrology: pd.DataFrame
    eds: pd.DataFrame
    shipments: pd.DataFrame


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, keys].head(5).to_dict("records")
        raise ValueError(f"{name}: duplicate keys {keys}: {sample}")


def load_inputs(data_dir: Path) -> Inputs:
    history = pd.read_csv(data_dir / "lot_history.csv")
    metrology = pd.read_csv(data_dir / "metrology_gox.csv")
    eds = pd.read_csv(data_dir / "eds_summary.csv")
    shipments = pd.read_csv(
        data_dir / "shipments.csv",
        dtype={"module_qty": "string"},
    )

    require_columns(
        history,
        ["lot_id", "step", "eq_id", "track_in_time"],
        "lot_history.csv",
    )
    require_columns(
        metrology,
        [
            "lot_id",
            "wafer_id",
            "param",
            "value_nm",
            "spec_lo",
            "spec_hi",
            "tool",
            "measure_time",
        ],
        "metrology_gox.csv",
    )
    require_columns(
        eds,
        ["lot_id", "yield_pct", "hot_refresh_fail_pct"],
        "eds_summary.csv",
    )
    require_columns(
        shipments,
        ["lot_id", "customer", "ship_date", "module_qty"],
        "shipments.csv",
    )

    history["track_in_time"] = pd.to_datetime(
        history["track_in_time"], errors="raise"
    )
    metrology["measure_time"] = pd.to_datetime(
        metrology["measure_time"], errors="raise"
    )
    shipments["ship_date"] = pd.to_datetime(
        shipments["ship_date"], errors="coerce"
    )
    shipments["module_qty"] = pd.to_numeric(
        shipments["module_qty"].str.replace(",", "", regex=False),
        errors="raise",
    ).astype("int64")

    return Inputs(history, metrology, eds, shipments)


def check_data_quality(inputs: Inputs) -> pd.DataFrame:
    history = inputs.history
    metrology = inputs.metrology
    eds = inputs.eds
    shipments = inputs.shipments

    require_unique(history, ["lot_id", "step"], "lot_history.csv")
    require_unique(
        metrology,
        ["lot_id", "wafer_id", "param"],
        "metrology_gox.csv",
    )
    require_unique(eds, ["lot_id"], "eds_summary.csv")
    require_unique(shipments, ["lot_id"], "shipments.csv")

    expected_steps = ["CLEAN", "DIFF_GOX", "METRO_GOX", "EDS"]
    step_order = {step: index for index, step in enumerate(expected_steps)}
    history_steps = history.assign(
        step_order=history["step"].map(step_order)
    ).sort_values(["lot_id", "track_in_time"])
    if history_steps["step_order"].isna().any():
        bad = sorted(history_steps.loc[history_steps["step_order"].isna(), "step"].unique())
        raise ValueError(f"lot_history.csv: unexpected process steps: {bad}")

    bad_step_lots: list[str] = []
    bad_time_lots: list[str] = []
    for lot_id, group in history_steps.groupby("lot_id", sort=True):
        if group["step"].tolist() != expected_steps:
            bad_step_lots.append(lot_id)
        if not group["track_in_time"].is_monotonic_increasing:
            bad_time_lots.append(lot_id)
    if bad_step_lots:
        raise ValueError(f"lot_history.csv: invalid process sequence: {bad_step_lots}")
    if bad_time_lots:
        raise ValueError(f"lot_history.csv: non-increasing process times: {bad_time_lots}")

    wafer_counts = metrology.groupby("lot_id")["wafer_id"].nunique()
    bad_wafer_counts = wafer_counts[wafer_counts != 5]
    if not bad_wafer_counts.empty:
        raise ValueError(
            "metrology_gox.csv: expected five wafers per lot: "
            f"{bad_wafer_counts.to_dict()}"
        )

    lot_sets = {
        "history": set(history["lot_id"]),
        "metrology": set(metrology["lot_id"]),
        "eds": set(eds["lot_id"]),
        "shipments": set(shipments["lot_id"]),
    }
    if len({frozenset(lots) for lots in lot_sets.values()}) != 1:
        detail = {name: len(lots) for name, lots in lot_sets.items()}
        raise ValueError(f"input lot sets do not match: {detail}")

    checks = [
        ("history_rows", len(history), 256),
        ("history_lots", history["lot_id"].nunique(), 64),
        ("metrology_rows", len(metrology), 320),
        ("metrology_lots", metrology["lot_id"].nunique(), 64),
        ("eds_rows", len(eds), 64),
        ("eds_lots", eds["lot_id"].nunique(), 64),
        ("shipment_rows", len(shipments), 64),
        ("shipment_lots", shipments["lot_id"].nunique(), 64),
        ("history_duplicate_keys", 0, 0),
        ("metrology_duplicate_keys", 0, 0),
        ("eds_duplicate_keys", 0, 0),
        ("shipment_duplicate_keys", 0, 0),
        ("lots_with_four_ordered_steps", len(history_steps.groupby("lot_id")), 64),
        ("lots_with_five_metrology_wafers", int((wafer_counts == 5).sum()), 64),
    ]
    result = pd.DataFrame(checks, columns=["check_name", "actual", "expected"])
    result["passed"] = result["actual"] == result["expected"]
    if not result["passed"].all():
        failed = result.loc[~result["passed"]].to_dict("records")
        raise ValueError(f"data quality checks failed: {failed}")
    return result


def correct_metrology(metrology: pd.DataFrame) -> pd.DataFrame:
    result = metrology.copy()
    correction_mask = (
        result["tool"].eq("MT-02")
        & result["measure_time"].ge(MT02_CORRECTION_START)
        & result["measure_time"].lt(MT02_CORRECTION_END)
    )
    result["correction_nm"] = 0.0
    result.loc[correction_mask, "correction_nm"] = -MT02_OFFSET_NM
    result["corrected_value_nm"] = result["value_nm"] + result["correction_nm"]
    result["correction_status"] = "not corrected"
    result.loc[correction_mask, "correction_status"] = (
        "calibration-record correction, provisional"
    )
    return result


def affected_process_mask(
    equipment: pd.Series, process_time: pd.Series
) -> pd.Series:
    """Return the approved process-based affected-lot decision."""
    return equipment.eq("DF-03") & process_time.between(
        AFFECTED_START, AFFECTED_END, inclusive="both"
    )


def build_lot_summary(inputs: Inputs, metrology: pd.DataFrame) -> pd.DataFrame:
    process = inputs.history.pivot(
        index="lot_id",
        columns="step",
        values=["eq_id", "track_in_time"],
    )
    process.columns = [
        f"{field.lower()}_{step.lower()}" for field, step in process.columns
    ]
    process = process.reset_index()

    metro_summary = metrology.groupby("lot_id", as_index=False).agg(
        metro_tool=("tool", "first"),
        metro_measure_time=("measure_time", "first"),
        gox_raw_mean_nm=("value_nm", "mean"),
        gox_corrected_mean_nm=("corrected_value_nm", "mean"),
        gox_corrected_min_nm=("corrected_value_nm", "min"),
        gox_corrected_max_nm=("corrected_value_nm", "max"),
        gox_spec_lo_nm=("spec_lo", "first"),
        gox_spec_hi_nm=("spec_hi", "first"),
        corrected_point_count=(
            "correction_nm",
            lambda values: int((values != 0).sum()),
        ),
    )
    metro_summary["correction_status"] = metro_summary["corrected_point_count"].map(
        lambda count: "교정 기록 기준 보정, 잠정" if count else "보정 없음"
    )

    summary = process.merge(
        metro_summary, on="lot_id", how="inner", validate="one_to_one"
    )
    summary = summary.merge(
        inputs.eds, on="lot_id", how="inner", validate="one_to_one"
    )
    summary = summary.merge(
        inputs.shipments, on="lot_id", how="inner", validate="one_to_one"
    )
    if len(summary) != 64:
        raise ValueError(f"joined result has {len(summary)} rows; expected 64")

    summary["is_affected"] = affected_process_mask(
        summary["eq_id_diff_gox"], summary["track_in_time_diff_gox"]
    )
    summary["shipment_status"] = summary["ship_date"].map(
        lambda value: "미출하" if pd.isna(value) else "출하"
    )
    numeric_columns = [
        "gox_raw_mean_nm",
        "gox_corrected_mean_nm",
        "gox_corrected_min_nm",
        "gox_corrected_max_nm",
        "gox_spec_lo_nm",
        "gox_spec_hi_nm",
        "yield_pct",
        "hot_refresh_fail_pct",
    ]
    summary[numeric_columns] = summary[numeric_columns].round(3)
    return summary.sort_values("lot_id").reset_index(drop=True)


def validate_results(summary: pd.DataFrame) -> None:
    actual = tuple(summary.loc[summary["is_affected"], "lot_id"])
    if actual != EXPECTED_AFFECTED_LOTS:
        raise ValueError(
            f"affected lots differ from approved result: expected={EXPECTED_AFFECTED_LOTS}, "
            f"actual={actual}"
        )
    focus = summary.loc[summary["lot_id"].eq("L6035")].iloc[0]
    expected = {
        "gox_raw_mean_nm": 4.884,
        "gox_corrected_mean_nm": 4.734,
        "hot_refresh_fail_pct": 1.72,
        "module_qty": 2000,
    }
    for column, expected_value in expected.items():
        actual_value = float(focus[column])
        if abs(actual_value - expected_value) > 1e-9:
            raise ValueError(
                f"L6035 validation failed for {column}: "
                f"expected={expected_value}, actual={actual_value}"
            )


def boundary_lots(summary: pd.DataFrame) -> pd.DataFrame:
    df03 = summary.loc[
        summary["eq_id_diff_gox"].eq("DF-03")
    ].sort_values("track_in_time_diff_gox")
    before = df03.loc[
        df03["track_in_time_diff_gox"] < AFFECTED_START
    ].tail(1).copy()
    after = df03.loc[
        df03["track_in_time_diff_gox"] > AFFECTED_END
    ].head(1).copy()
    before["boundary_role"] = "last normal lot"
    after["boundary_role"] = "first normal lot after repair"
    return pd.concat([before, after], ignore_index=True)


def make_core_summary(summary: pd.DataFrame, boundaries: pd.DataFrame) -> dict:
    affected = summary.loc[summary["is_affected"]]
    customer_groups = affected.groupby("customer", dropna=False).agg(
        lot_count=("lot_id", "count"),
        module_qty=("module_qty", "sum"),
    )
    normal = summary.loc[~summary["is_affected"]]
    focus = summary.loc[summary["lot_id"].eq("L6035")].iloc[0]
    return {
        "affected_lots": affected["lot_id"].tolist(),
        "affected_lot_count": int(len(affected)),
        "affected_module_qty": int(affected["module_qty"].sum()),
        "by_customer": {
            str(customer): {
                "lot_count": int(row["lot_count"]),
                "module_qty": int(row["module_qty"]),
            }
            for customer, row in customer_groups.iterrows()
        },
        "affected_hot_refresh_fail_mean_pct": round(
            float(affected["hot_refresh_fail_pct"].mean()), 3
        ),
        "unaffected_hot_refresh_fail_mean_pct": round(
            float(normal["hot_refresh_fail_pct"].mean()), 3
        ),
        "boundary_lots": boundaries["lot_id"].tolist(),
        "l6035": {
            "gox_raw_mean_nm": round(float(focus["gox_raw_mean_nm"]), 3),
            "gox_corrected_mean_nm": round(
                float(focus["gox_corrected_mean_nm"]), 3
            ),
            "hot_refresh_fail_pct": float(focus["hot_refresh_fail_pct"]),
            "module_qty": int(focus["module_qty"]),
        },
    }


def plot_lot_trend(summary: pd.DataFrame, path: Path) -> None:
    figure, gox_axis = plt.subplots(figsize=(15, 7))
    positions = list(range(len(summary)))
    affected = summary["is_affected"]
    colors = ["#d62728" if value else "#7f7f7f" for value in affected]

    gox_axis.errorbar(
        positions,
        summary["gox_corrected_mean_nm"],
        yerr=[
            summary["gox_corrected_mean_nm"] - summary["gox_corrected_min_nm"],
            summary["gox_corrected_max_nm"] - summary["gox_corrected_mean_nm"],
        ],
        fmt="none",
        ecolor="#9a9a9a",
        alpha=0.45,
        capsize=2,
    )
    gox_axis.scatter(
        positions,
        summary["gox_corrected_mean_nm"],
        c=colors,
        s=34,
        label="Corrected GOX mean (red: affected)",
    )
    gox_axis.axhline(
        4.8, color="#ff7f0e", linestyle="--", label="GOX specification low"
    )
    gox_axis.set_xlabel("Lot")
    gox_axis.set_ylabel("GOX thickness (nm)")
    gox_axis.set_xticks(positions[::2])
    gox_axis.set_xticklabels(summary["lot_id"].iloc[::2], rotation=60)

    eds_axis = gox_axis.twinx()
    eds_axis.plot(
        positions,
        summary["hot_refresh_fail_pct"],
        color="#1f77b4",
        linewidth=1.3,
        label="EDS hot-refresh failure",
    )
    eds_axis.axhspan(0.3, 0.6, color="#1f77b4", alpha=0.08)
    eds_axis.set_ylabel("EDS hot-refresh failure (%)")

    focus_index = int(summary.index[summary["lot_id"].eq("L6035")][0])
    gox_axis.annotate(
        "L6035",
        (focus_index, summary.loc[focus_index, "gox_corrected_mean_nm"]),
        xytext=(8, -30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
        fontweight="bold",
    )
    for lot_id in ("L6028", "L6040", "L6059"):
        index = int(summary.index[summary["lot_id"].eq(lot_id)][0])
        gox_axis.annotate(
            lot_id,
            (index, summary.loc[index, "gox_corrected_mean_nm"]),
            xytext=(5, 12),
            textcoords="offset points",
            fontsize=8,
        )

    handles_1, labels_1 = gox_axis.get_legend_handles_labels()
    handles_2, labels_2 = eds_axis.get_legend_handles_labels()
    gox_axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="lower right")
    figure.suptitle("GOX thickness and EDS hot-refresh failure by lot")
    figure.text(
        0.5,
        0.01,
        "Affected lots are defined by DF-03 process time; thickness and EDS are supporting evidence.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_gox_vs_eds(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    normal = summary.loc[~summary["is_affected"]]
    affected = summary.loc[summary["is_affected"]]

    axis.scatter(
        normal["gox_corrected_mean_nm"],
        normal["hot_refresh_fail_pct"],
        color="#7f7f7f",
        alpha=0.65,
        label="Not affected",
    )
    axis.scatter(
        affected["gox_corrected_mean_nm"],
        affected["hot_refresh_fail_pct"],
        color="#d62728",
        s=65,
        label="Affected by process-time rule",
    )
    focus = affected.loc[affected["lot_id"].eq("L6035")]
    axis.scatter(
        focus["gox_corrected_mean_nm"],
        focus["hot_refresh_fail_pct"],
        color="#ffbf00",
        edgecolor="black",
        marker="*",
        s=260,
        label="L6035",
        zorder=5,
    )
    label_offsets = {
        "L6035": (8, -2),
        "L6039": (4, 4),
        "L6040": (4, 4),
        "L6043": (8, 8),
        "L6048": (4, 4),
        "L6051": (4, 4),
        "L6055": (4, 4),
        "L6056": (4, -10),
    }
    for row in affected.itertuples():
        offset = label_offsets[row.lot_id]
        axis.annotate(
            row.lot_id,
            (row.gox_corrected_mean_nm, row.hot_refresh_fail_pct),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Corrected GOX mean (nm)")
    axis.set_ylabel("EDS hot-refresh failure (%)")
    axis.set_title("Corrected GOX thickness vs. EDS hot-refresh failure")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.text(
        0.5,
        0.01,
        "MT-02 correction is provisional and based on the calibration record.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    customer_output: Path,
    quality: pd.DataFrame,
    metrology: pd.DataFrame,
    summary: pd.DataFrame,
    boundaries: pd.DataFrame,
    core_summary: dict,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    quality.to_csv(output_dir / "data_quality.csv", index=False, encoding="utf-8")
    metrology.to_csv(
        output_dir / "metrology_corrected.csv", index=False, encoding="utf-8"
    )
    summary.to_csv(output_dir / "lot_summary.csv", index=False, encoding="utf-8")
    summary.loc[summary["is_affected"]].to_csv(
        output_dir / "affected_lots_internal.csv", index=False, encoding="utf-8"
    )
    boundaries.to_csv(
        output_dir / "df03_boundary_lots.csv", index=False, encoding="utf-8"
    )
    (output_dir / "core_summary.json").write_text(
        json.dumps(core_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    affected_customer_a = summary.loc[
        summary["is_affected"] & summary["customer"].eq("고객A"),
        ["lot_id", "ship_date", "module_qty"],
    ].copy()
    affected_customer_a = affected_customer_a.rename(
        columns={
            "lot_id": "LOT_ID",
            "ship_date": "SHIP_DATE",
            "module_qty": "MODULE_QTY",
        }
    ).sort_values("LOT_ID")
    affected_customer_a["SHIP_DATE"] = affected_customer_a[
        "SHIP_DATE"
    ].dt.strftime("%Y-%m-%d")
    customer_output.parent.mkdir(parents=True, exist_ok=True)
    affected_customer_a.to_csv(
        customer_output,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    plot_lot_trend(summary, figures_dir / "gox_eds_by_lot.png")
    plot_gox_vs_eds(summary, figures_dir / "gox_vs_hot_refresh.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the customer A high-temperature refresh complaint."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--customer-output",
        type=Path,
        default=Path("AFFECTED_LOTS_고객A.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_inputs(args.data_dir)
    quality = check_data_quality(inputs)
    metrology = correct_metrology(inputs.metrology)
    summary = build_lot_summary(inputs, metrology)
    validate_results(summary)
    boundaries = boundary_lots(summary)
    core_summary = make_core_summary(summary, boundaries)
    write_outputs(
        args.output_dir,
        args.customer_output,
        quality,
        metrology,
        summary,
        boundaries,
        core_summary,
    )
    print(json.dumps(core_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
