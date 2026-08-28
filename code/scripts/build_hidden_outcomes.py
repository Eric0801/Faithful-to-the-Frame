#!/usr/bin/env python3
"""Build hidden post-event outcomes plus pre-event price context.

This script keeps price-derived labels separate from any agent-facing packets.
It reads an events CSV plus a local long-format adjusted-price panel, aligns
each event to the first tradable day `t0`, computes pre-event price context,
and writes a separate hidden-outcomes CSV.

Default hidden-valence rule:
- positive if CAR_1_5 >= +3%
- negative if CAR_1_5 <= -3%
- mixed otherwise
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = PROJECT_ROOT / "data" / "candidate_events_sec_matching.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "hidden_outcomes" / "hidden_outcomes.csv"

EASTERN = ZoneInfo("America/New_York")
MARKET_PROXY = "SPY"
VALENCE_POSITIVE_CUTOFF = 0.03
VALENCE_NEGATIVE_CUTOFF = -0.03

REQUIRED_EVENT_COLUMNS = ["ticker", "event_date", "accepted_at_et"]
OUTPUT_METRIC_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a separate hidden-outcomes table from an events CSV and a "
            "local adjusted-price panel that includes SPY."
        )
    )
    parser.add_argument("--events-csv", default=str(DEFAULT_EVENTS))
    parser.add_argument(
        "--prices-csv",
        required=True,
        help="Long-format CSV with ticker,date,adj_close columns and SPY rows.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"--events-csv not found: {path}")
    events = pd.read_csv(path)
    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in events.columns]
    if missing:
        raise ValueError(
            "--events-csv is missing required columns: "
            + ", ".join(sorted(missing))
        )
    return events


def load_price_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"--prices-csv not found: {path}")

    raw = pd.read_csv(path)
    cols = {col.lower(): col for col in raw.columns}
    if not {"ticker", "date", "adj_close"}.issubset(cols):
        raise ValueError(
            "--prices-csv must be long format with ticker,date,adj_close columns."
        )

    panel = raw.rename(
        columns={
            cols["ticker"]: "ticker",
            cols["date"]: "date",
            cols["adj_close"]: "adj_close",
        }
    )[["ticker", "date", "adj_close"]]
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.date
    panel["adj_close"] = pd.to_numeric(panel["adj_close"], errors="coerce")
    panel = panel.dropna(subset=["ticker", "date", "adj_close"])
    panel = panel[panel["ticker"] != ""]

    duplicates = panel[panel.duplicated(subset=["ticker", "date"], keep=False)]
    if not duplicates.empty:
        sample = duplicates[["ticker", "date"]].head(5).astype(str)
        example = ", ".join(
            f"{row.ticker}@{row.date}" for row in sample.itertuples(index=False)
        )
        raise ValueError(
            "Price CSV has duplicate ticker/date rows. Remove duplicates before "
            f"running this script. Example rows: {example}"
        )

    if MARKET_PROXY not in set(panel["ticker"]):
        raise ValueError(
            f"--prices-csv must include {MARKET_PROXY} rows in the ticker column."
        )

    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_price_frames(price_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    market = (
        price_panel[price_panel["ticker"] == MARKET_PROXY][["date", "adj_close"]]
        .rename(columns={"adj_close": "market_adj_close"})
        .sort_values("date")
    )
    frames: dict[str, pd.DataFrame] = {}
    for ticker in sorted(price_panel["ticker"].unique()):
        if ticker == MARKET_PROXY:
            continue
        stock = (
            price_panel[price_panel["ticker"] == ticker][["date", "adj_close"]]
            .rename(columns={"adj_close": "stock_adj_close"})
            .sort_values("date")
        )
        frame = stock.merge(market, on="date", how="inner")
        if frame.empty:
            continue
        frame["stock_ret"] = frame["stock_adj_close"].pct_change()
        frame["market_ret"] = frame["market_adj_close"].pct_change()
        frames[ticker] = frame.reset_index(drop=True)
    return frames


def parse_acceptance_datetime(value: Any) -> dt.datetime | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        stamp = pd.Timestamp(text)
    except Exception:
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(EASTERN)
    else:
        stamp = stamp.tz_convert(EASTERN)
    return stamp.to_pydatetime()


def after_close_flag(accepted_at_et: dt.datetime) -> bool:
    accepted_time = accepted_at_et.timetz().replace(tzinfo=None)
    return accepted_time > dt.time(16, 0)


def align_t0(
    accepted_at_et: dt.datetime, trading_dates: list[dt.date]
) -> tuple[dt.date | None, bool]:
    filing_date = accepted_at_et.date()
    after_close = after_close_flag(accepted_at_et)
    for trading_date in trading_dates:
        if trading_date < filing_date:
            continue
        if trading_date == filing_date and after_close:
            continue
        return trading_date, after_close
    return None, after_close


def cumulative_return(end_price: float, start_price: float) -> float | None:
    if pd.isna(end_price) or pd.isna(start_price) or start_price <= 0:
        return None
    return float(end_price / start_price - 1.0)


def clean_cell(value: Any, uppercase: bool = False) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text.upper() if uppercase else text


def default_result(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "event_id" in row.index:
        result["event_id"] = clean_cell(row.get("event_id", ""))
    if "accession_number" in row.index:
        result["accession_number"] = clean_cell(row.get("accession_number", ""))
    result["ticker"] = clean_cell(row.get("ticker", ""), uppercase=True)
    result["event_date"] = clean_cell(row.get("event_date", ""))
    result["accepted_at_et"] = clean_cell(row.get("accepted_at_et", ""))
    result.update(
        {
            "reaction_day_t0": "",
            "after_close": "",
            "ret_5d": "",
            "ret_20d": "",
            "market_ret_20d": "",
            "CAR_1_5": "",
            "hidden_valence": "unknown",
            "price_ok": False,
            "price_error": "",
        }
    )
    return result


def compute_event_outcome(
    row: pd.Series, price_frames: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    result = default_result(row)
    accepted_at_et = parse_acceptance_datetime(row.get("accepted_at_et"))
    if accepted_at_et is None:
        result["price_error"] = "invalid_accepted_at_et"
        return result

    ticker = result["ticker"]
    frame = price_frames.get(ticker)
    if frame is None or frame.empty:
        result["price_error"] = "missing_ticker_or_market_prices"
        return result

    trading_dates = frame["date"].tolist()
    t0, after_close = align_t0(accepted_at_et, trading_dates)
    result["after_close"] = after_close
    if t0 is None:
        result["price_error"] = "no_trading_day_on_or_after_event"
        return result

    result["reaction_day_t0"] = t0.isoformat()
    loc_list = frame.index[frame["date"] == t0].tolist()
    if not loc_list:
        result["price_error"] = "aligned_t0_missing_from_price_frame"
        return result
    loc = loc_list[0]

    if loc < 21:
        result["price_error"] = "not_enough_pre_event_history"
        return result
    if loc + 5 >= len(frame):
        result["price_error"] = "not_enough_post_event_history"
        return result

    # Pre-event context stops at t0-1 so agent-facing packets never include
    # same-day reaction information when an event arrives during trading hours.
    ret_5d = cumulative_return(
        frame.at[loc - 1, "stock_adj_close"],
        frame.at[loc - 6, "stock_adj_close"],
    )
    ret_20d = cumulative_return(
        frame.at[loc - 1, "stock_adj_close"],
        frame.at[loc - 21, "stock_adj_close"],
    )
    market_ret_20d = cumulative_return(
        frame.at[loc - 1, "market_adj_close"],
        frame.at[loc - 21, "market_adj_close"],
    )
    if ret_5d is None or ret_20d is None or market_ret_20d is None:
        result["price_error"] = "invalid_pre_event_prices"
        return result

    window = frame.iloc[loc + 1 : loc + 6]
    if len(window) != 5:
        result["price_error"] = "not_enough_post_event_history"
        return result
    if window[["stock_ret", "market_ret"]].isna().any().any():
        result["price_error"] = "missing_returns_in_car_window"
        return result

    car_1_5 = float((window["stock_ret"] - window["market_ret"]).sum())
    if car_1_5 >= VALENCE_POSITIVE_CUTOFF:
        hidden_valence = "positive"
    elif car_1_5 <= VALENCE_NEGATIVE_CUTOFF:
        hidden_valence = "negative"
    else:
        hidden_valence = "mixed"

    result.update(
        {
            "ret_5d": round(ret_5d, 6),
            "ret_20d": round(ret_20d, 6),
            "market_ret_20d": round(market_ret_20d, 6),
            "CAR_1_5": round(car_1_5, 6),
            "hidden_valence": hidden_valence,
            "price_ok": True,
            "price_error": "",
        }
    )
    return result


def output_columns(events: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    if "event_id" in events.columns:
        columns.append("event_id")
    if "accession_number" in events.columns:
        columns.append("accession_number")
    columns.extend(["ticker", "event_date", "accepted_at_et"])
    columns.extend(OUTPUT_METRIC_COLUMNS)
    return columns


def main() -> int:
    args = parse_args()
    events = load_events(Path(args.events_csv))
    if args.limit:
        events = events.head(args.limit)

    price_panel = load_price_panel(Path(args.prices_csv))
    price_frames = build_price_frames(price_panel)

    rows = [
        compute_event_outcome(row, price_frames)
        for _, row in events.reset_index(drop=True).iterrows()
    ]
    columns = output_columns(events)
    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=columns)
    else:
        output = output.reindex(columns=columns)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    ok_count = int(output["price_ok"].fillna(False).sum())
    print(f"wrote {len(output)} hidden outcome rows to {output_path}")
    print(f"price-complete rows: {ok_count}/{len(output)}")
    if ok_count < len(output):
        print(output["price_error"].value_counts(dropna=True).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
