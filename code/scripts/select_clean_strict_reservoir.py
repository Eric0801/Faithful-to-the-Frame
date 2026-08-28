#!/usr/bin/env python3
"""Select a clean main reservoir with complete CAR and one event per issuer."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from find_matching_events import SCREENING_COLUMNS


DEFAULT_EVENTS = "data/screening_2026_main/screening_events_all.csv"
DEFAULT_HIDDEN_OUTCOMES = "data/hidden_outcomes_2026_main/hidden_outcomes_screening_all.csv"
DEFAULT_OUTPUT_DIR = "data/screening_2026_main/clean_strict_reservoir"
MARKET_CAP_GROUPS = ("large_cap", "small_mid_cap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select an outcome-complete, one-issuer-per-event reservoir from "
            "the full screening pool. Defaults favor cleanliness over forcing "
            "the original 100-event target."
        )
    )
    parser.add_argument("--events-csv", default=DEFAULT_EVENTS)
    parser.add_argument("--hidden-outcomes-csv", default=DEFAULT_HIDDEN_OUTCOMES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--target-large-cap", type=int, default=50)
    parser.add_argument("--target-small-mid-cap", type=int, default=50)
    parser.add_argument(
        "--balance-to-common-count",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the same selected count for large_cap and small_mid_cap.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def load_and_merge(events_csv: Path, hidden_outcomes_csv: Path) -> pd.DataFrame:
    events = pd.read_csv(events_csv)
    hidden = pd.read_csv(hidden_outcomes_csv)
    require_columns(events, {"accession_number", "ticker", "market_cap_group"}, "events")
    require_columns(
        hidden,
        {
            "accession_number",
            "price_ok",
            "CAR_1_5",
            "hidden_valence",
            "price_error",
        },
        "hidden outcomes",
    )

    hidden_subset = hidden[
        [
            "accession_number",
            "ticker",
            "price_ok",
            "CAR_1_5",
            "hidden_valence",
            "price_error",
            "reaction_day_t0",
            "after_close",
            "ret_5d",
            "ret_20d",
            "market_ret_20d",
        ]
    ].rename(
        columns={
            "price_ok": "hidden_price_ok",
            "CAR_1_5": "hidden_CAR_1_5",
            "hidden_valence": "hidden_valence_joined",
            "price_error": "hidden_price_error",
        }
    )
    events["ticker"] = events["ticker"].astype(str).str.upper().str.strip()
    hidden_subset["ticker"] = hidden_subset["ticker"].astype(str).str.upper().str.strip()
    merged = events.merge(hidden_subset, on=["accession_number", "ticker"], how="left")
    merged["ticker"] = merged["ticker"].astype(str).str.upper().str.strip()
    return merged


def eligibility_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if clean_text(row.get("exclusion_reason")):
        reasons.append("exclusion_reason")
    for column, reason in (
        ("parse_ok", "parse_not_ok"),
        ("xbrl_ok", "xbrl_not_ok"),
        ("has_item_2_02", "missing_item_2_02"),
        ("masking_ok", "masking_not_ok"),
    ):
        if not bool_value(row.get(column)):
            reasons.append(reason)
    if not bool_value(row.get("hidden_price_ok")):
        price_error = clean_text(row.get("hidden_price_error")) or "price_not_ok"
        reasons.append(price_error)
    if pd.isna(pd.to_numeric(row.get("hidden_CAR_1_5"), errors="coerce")):
        reasons.append("missing_CAR_1_5")
    if not clean_text(row.get("ticker")):
        reasons.append("missing_ticker")
    if clean_text(row.get("market_cap_group")) not in MARKET_CAP_GROUPS:
        reasons.append("unsupported_market_cap_group")
    return reasons


def attach_audit_columns(events: pd.DataFrame) -> pd.DataFrame:
    audit = events.copy()
    all_reasons = [eligibility_reasons(row) for _, row in audit.iterrows()]
    audit["eligible_for_clean_reservoir"] = [not reasons for reasons in all_reasons]
    audit["clean_reservoir_fail_reasons"] = [
        ";".join(reasons) for reasons in all_reasons
    ]
    return audit


def issuer_cap_key(row: pd.Series) -> str:
    cik = clean_text(row.get("cik"))
    if cik:
        return f"cik:{cik}"
    return f"ticker:{clean_text(row.get('ticker')).upper()}"


def choose_one_per_issuer(eligible: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = eligible.copy()
    work["accepted_at_sort"] = pd.to_datetime(work["accepted_at_et"], errors="coerce", utc=True)
    work["source_token_count_num"] = pd.to_numeric(
        work["source_token_count"], errors="coerce"
    ).fillna(0)
    work["abs_car"] = pd.to_numeric(work["hidden_CAR_1_5"], errors="coerce").abs()
    work["issuer_cap_key"] = work.apply(issuer_cap_key, axis=1)
    work = work.sort_values(
        [
            "market_cap_group",
            "issuer_cap_key",
            "ticker",
            "accepted_at_sort",
            "source_token_count_num",
            "abs_car",
            "accession_number",
        ],
        ascending=[True, True, True, False, False, False, True],
    )
    chosen = work.drop_duplicates(
        subset=["market_cap_group", "issuer_cap_key"],
        keep="first",
    )
    chosen_keys = set(chosen["accession_number"].astype(str))
    dropped = work[~work["accession_number"].astype(str).isin(chosen_keys)].copy()
    return (
        chosen.drop(
            columns=[
                "accepted_at_sort",
                "source_token_count_num",
                "abs_car",
                "issuer_cap_key",
            ]
        ),
        dropped.drop(
            columns=[
                "accepted_at_sort",
                "source_token_count_num",
                "abs_car",
                "issuer_cap_key",
            ]
        ),
    )


def selected_count_targets(
    unique_events: pd.DataFrame,
    *,
    target_large_cap: int,
    target_small_mid_cap: int,
    balance_to_common_count: bool,
) -> dict[str, int]:
    requested = {
        "large_cap": target_large_cap,
        "small_mid_cap": target_small_mid_cap,
    }
    available = {
        group: int((unique_events["market_cap_group"] == group).sum())
        for group in MARKET_CAP_GROUPS
    }
    targets = {
        group: min(requested[group], available[group])
        for group in MARKET_CAP_GROUPS
    }
    if balance_to_common_count:
        common = min(targets.values())
        targets = {group: common for group in MARKET_CAP_GROUPS}
    return targets


def choose_group_rows(group: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or group.empty:
        return group.iloc[0:0]
    work = group.copy()
    work["accepted_at_sort"] = pd.to_datetime(work["accepted_at_et"], errors="coerce", utc=True)
    work["source_token_count_num"] = pd.to_numeric(
        work["source_token_count"], errors="coerce"
    ).fillna(0)
    work["abs_car"] = pd.to_numeric(work["hidden_CAR_1_5"], errors="coerce").abs()
    work = work.sort_values(
        [
            "hidden_valence_joined",
            "sector",
            "accepted_at_sort",
            "source_token_count_num",
            "abs_car",
            "ticker",
        ],
        ascending=[True, True, False, False, False, True],
    )
    return work.head(count).drop(
        columns=["accepted_at_sort", "source_token_count_num", "abs_car"]
    )


def build_selected_csv(selected: pd.DataFrame) -> pd.DataFrame:
    output = selected.copy()
    output["price_ok"] = True
    output["CAR_1_5"] = pd.to_numeric(output["hidden_CAR_1_5"], errors="coerce")
    output["hidden_valence"] = output["hidden_valence_joined"].fillna("unknown")
    for column in SCREENING_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[SCREENING_COLUMNS]


def build_selected_hidden_outcomes(selected: pd.DataFrame) -> pd.DataFrame:
    output = selected.copy()
    output["price_ok"] = True
    output["CAR_1_5"] = pd.to_numeric(output["hidden_CAR_1_5"], errors="coerce")
    output["hidden_valence"] = output["hidden_valence_joined"].fillna("unknown")
    output["price_error"] = ""
    columns = [
        "accession_number",
        "ticker",
        "event_date",
        "accepted_at_et",
        "reaction_day_t0",
        "after_close",
        "ret_5d",
        "ret_20d",
        "market_ret_20d",
        "CAR_1_5",
        "hidden_valence",
        "price_ok",
        "price_error",
    ]
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    return output[columns]


def counter_dict(values: pd.Series) -> dict[str, int]:
    return dict(Counter(clean_text(value) for value in values if clean_text(value)))


def build_summary(
    *,
    events_csv: Path,
    hidden_outcomes_csv: Path,
    output_csv: Path,
    hidden_output_csv: Path,
    audit_csv: Path,
    duplicate_drop_csv: Path,
    audit: pd.DataFrame,
    eligible: pd.DataFrame,
    unique_events: pd.DataFrame,
    selected: pd.DataFrame,
    dropped_duplicates: pd.DataFrame,
    targets: dict[str, int],
) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now_iso(),
        "inputs": {
            "events_csv": str(events_csv),
            "hidden_outcomes_csv": str(hidden_outcomes_csv),
        },
        "outputs": {
            "selected_csv": str(output_csv),
            "selected_hidden_outcomes_csv": str(hidden_output_csv),
            "audit_csv": str(audit_csv),
            "duplicate_drop_csv": str(duplicate_drop_csv),
        },
        "selection_policy": {
            "require_complete_CAR_1_5": True,
            "require_price_ok": True,
            "one_event_per_issuer_within_market_cap_group": True,
            "duplicate_issuer_choice": (
                "latest accepted_at_et, then larger source_token_count, then "
                "larger absolute CAR_1_5, then accession_number"
            ),
            "balanced_to_common_market_cap_count": True,
        },
        "counts": {
            "input_rows": int(len(audit)),
            "eligible_rows": int(len(eligible)),
            "unique_issuer_rows_after_cap": int(len(unique_events)),
            "selected_rows": int(len(selected)),
            "dropped_duplicate_issuer_rows": int(len(dropped_duplicates)),
        },
        "targets": targets,
        "selected_by_market_cap_group": counter_dict(selected["market_cap_group"]),
        "selected_unique_tickers_by_market_cap_group": {
            group: int(selected[selected["market_cap_group"] == group]["ticker"].nunique())
            for group in MARKET_CAP_GROUPS
        },
        "selected_by_hidden_valence": counter_dict(selected["hidden_valence_joined"]),
        "selected_by_sector": counter_dict(selected["sector"]),
        "eligible_fail_reasons": counter_dict(
            audit.loc[
                ~audit["eligible_for_clean_reservoir"],
                "clean_reservoir_fail_reasons",
            ]
        ),
        "duplicate_issuer_tickers_dropped": sorted(
            dropped_duplicates["ticker"].dropna().astype(str).unique().tolist()
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    events_csv = Path(args.events_csv)
    hidden_outcomes_csv = Path(args.hidden_outcomes_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output_csv) if args.output_csv else output_dir / "selected_events.csv"
    hidden_output_csv = output_dir / "selected_hidden_outcomes.csv"
    audit_csv = output_dir / "selection_audit.csv"
    duplicate_drop_csv = output_dir / "dropped_duplicate_ticker_events.csv"
    summary_json = output_dir / "selection_summary.json"

    events = load_and_merge(events_csv, hidden_outcomes_csv)
    audit = attach_audit_columns(events)
    eligible = audit[audit["eligible_for_clean_reservoir"]].copy()
    unique_events, dropped_duplicates = choose_one_per_issuer(eligible)
    targets = selected_count_targets(
        unique_events,
        target_large_cap=args.target_large_cap,
        target_small_mid_cap=args.target_small_mid_cap,
        balance_to_common_count=args.balance_to_common_count,
    )
    selected_parts = [
        choose_group_rows(
            unique_events[unique_events["market_cap_group"] == group],
            targets[group],
        )
        for group in MARKET_CAP_GROUPS
    ]
    selected = pd.concat(selected_parts, ignore_index=True)
    selected_csv = build_selected_csv(selected)
    selected_hidden_outcomes = build_selected_hidden_outcomes(selected)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_csv.to_csv(output_csv, index=False)
    selected_hidden_outcomes.to_csv(hidden_output_csv, index=False)
    audit.to_csv(audit_csv, index=False)
    dropped_duplicates.to_csv(duplicate_drop_csv, index=False)
    write_json(
        summary_json,
        build_summary(
            events_csv=events_csv,
            hidden_outcomes_csv=hidden_outcomes_csv,
            output_csv=output_csv,
            hidden_output_csv=hidden_output_csv,
            audit_csv=audit_csv,
            duplicate_drop_csv=duplicate_drop_csv,
            audit=audit,
            eligible=eligible,
            unique_events=unique_events,
            selected=selected,
            dropped_duplicates=dropped_duplicates,
            targets=targets,
        ),
    )

    print(f"wrote {len(selected_csv)} clean strict events to {output_csv}")
    print(selected_csv.groupby("market_cap_group").size().to_string())
    print(f"summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
