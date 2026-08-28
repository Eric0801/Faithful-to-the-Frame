#!/usr/bin/env python3
"""Build masked source packets from a validated events CSV or queue.

This script reads a validated event list, extracts the selected EX-99.x
earnings-release text from cached SEC full-submission containers, and writes
one agent-facing JSON packet per event plus aggregate outputs.

Notes:
- Hidden outcomes are never embedded in agent-facing packet JSON. If supplied
  through ``--hidden-outcomes-csv``, they are written to a separate sidecar.
- XBRL companyfacts often lag the earnings-release 8-K. If the relevant period
  cannot be recovered from cached companyfacts at or before the acceptance
  timestamp, this script emits an empty ``xbrl_facts`` list instead of failing.
- Pre-event price context is only computed when ``--prices-csv`` is supplied.
  Otherwise the packet keeps the expected keys with null values.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "data" / "cache"
DEFAULT_EVENTS = PROJECT_ROOT / "data" / "human_validation" / "validation_queue.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "source_packets"

MASKED_CIK = "masked_cik"
MASKED_ACCESSION = "masked_accession"
MASKED_TIMESTAMP = "masked_timestamp"
MASKED_REACTION_DAY = "masked_reaction_day"

BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "p", "li", "div", "td", "th")
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5"}

HIDDEN_INPUT_COLUMNS = {
    "CAR_1_5",
    "CAR_1_20",
    "hidden_valence",
    "hidden_outcomes",
}

SECTION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("guidance", (r"\bguidance\b", r"\boutlook\b", r"\bexpect(?:ed|s)?\b", r"\bforecast\b")),
    ("revenue", (r"\brevenue\b", r"\brevenues\b", r"\bsales\b")),
    (
        "earnings_or_eps",
        (
            r"\bearnings per share\b",
            r"\bdiluted eps\b",
            r"\bbasic eps\b",
            r"\beps\b",
            r"\bnet income\b",
            r"\bprofit\b",
        ),
    ),
    (
        "margins",
        (
            r"\bmargin\b",
            r"\bmargins\b",
            r"\bebitda\b",
            r"\bgross profit\b",
            r"\boperating income\b",
        ),
    ),
    (
        "costs_or_expenses",
        (
            r"\bexpense\b",
            r"\bexpenses\b",
            r"\bcost\b",
            r"\bcapex\b",
            r"\bcapital expenditure",
            r"\brestructur",
        ),
    ),
    (
        "balance_sheet_or_cash",
        (
            r"\bcash\b",
            r"\bliquidity\b",
            r"\bdebt\b",
            r"\bleverage\b",
            r"\bfree cash flow\b",
            r"\boperating cash flow\b",
            r"\bbalance sheet\b",
        ),
    ),
    (
        "risk_or_uncertainty",
        (
            r"\brisk\b",
            r"\buncertaint",
            r"\bheadwind",
            r"\binflation\b",
            r"\bvolatile\b",
            r"\bsoft(?:ness)?\b",
            r"\bchallenge",
        ),
    ),
    (
        "operating_metrics",
        (
            r"\bcustomer\b",
            r"\bcustomers\b",
            r"\bsubscriber\b",
            r"\bsubscribers\b",
            r"\border\b",
            r"\borders\b",
            r"\bbacklog\b",
            r"\btraffic\b",
            r"\bvolume\b",
            r"\bshipments?\b",
        ),
    ),
]

SKIP_SOURCE_PATTERNS = (
    r"\bconference call\b",
    r"\bwebcast\b",
    r"\breplay\b",
    r"\bforward-looking statements?\b",
    r"\bsafe harbor\b",
    r"\binvestor relations\b",
    r"\bnon-gaap reconciliation\b",
)

XBRL_TAG_PRIORITY = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "GrossProfit",
    "OperatingIncomeLoss",
    "OperatingExpenses",
    "NetIncomeLoss",
    "ProfitLoss",
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
    "NetCashProvidedByUsedInOperatingActivities",
]

COMMON_GENERIC_NAME_TOKENS = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "holdings",
    "holding",
    "group",
    "plc",
    "ltd",
    "llc",
    "technologies",
    "technology",
    "solutions",
    "services",
}

MONTH_TO_QUARTER = {
    3: "Q1",
    6: "Q2",
    9: "Q3",
    12: "Q4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-csv", default=str(DEFAULT_EVENTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--hidden-outcomes-csv", default="")
    parser.add_argument("--prices-csv", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return missing if isinstance(missing, bool) else False


def normalized_str(value: Any) -> str:
    if is_missing(value):
        return ""
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def extract_tag(block: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*([^\n\r<]+)", block, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def clean_html_text(raw_text: str) -> str:
    unescaped = html.unescape(raw_text)
    soup = BeautifulSoup(unescaped, "html.parser")
    for tag in soup(["script", "style", "ix:header", "xbrl"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_paragraphs(text: str, max_chars: int = 950) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    paragraphs: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            paragraphs.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        paragraphs.append(current.strip())
    return paragraphs


def rough_token_count(text: str) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / 4))


def parse_acceptance_datetime(value: Any) -> dt.datetime | None:
    raw = normalized_str(value)
    if not raw:
        return None
    candidate = raw.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(candidate)
    except ValueError:
        pass
    try:
        return pd.to_datetime(raw).to_pydatetime()
    except Exception:
        return None


def parse_date(value: Any) -> dt.date | None:
    raw = normalized_str(value)
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(raw[:10])
    except ValueError:
        try:
            return pd.to_datetime(raw).date()
        except Exception:
            return None


def after_close(accepted_at: dt.datetime | None) -> bool:
    if accepted_at is None:
        return False
    return (accepted_at.hour, accepted_at.minute, accepted_at.second) > (16, 0, 0)


def next_business_day(day: dt.date) -> dt.date:
    candidate = day + dt.timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += dt.timedelta(days=1)
    return candidate


def assign_event_ids(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy().reset_index(drop=True)
    existing = set()
    assigned: list[str] = []
    counter = 1
    for value in events["event_id"] if "event_id" in events.columns else [None] * len(events):
        current = normalized_str(value)
        if current:
            assigned.append(current)
            existing.add(current)
            continue
        while True:
            candidate = f"evt_{counter:04d}"
            counter += 1
            if candidate not in existing:
                assigned.append(candidate)
                existing.add(candidate)
                break
    events["event_id"] = assigned
    return events


def placeholder_suffix(index: int) -> str:
    letters = []
    current = index
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def issuer_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        normalized_str(row.get("cik")),
        normalized_str(row.get("company")),
        normalized_str(row.get("ticker")),
    )


def build_company_placeholders(events: pd.DataFrame) -> dict[str, str]:
    unique_issuers = sorted({issuer_key(row) for _, row in events.iterrows()})
    mapping = {
        "|".join(key): f"Company {placeholder_suffix(idx)}"
        for idx, key in enumerate(unique_issuers, start=1)
    }
    return mapping


def company_aliases(company: str, ticker: str) -> list[str]:
    aliases: set[str] = set()
    raw = company.strip()
    if raw:
        aliases.add(raw)
        aliases.add(raw.split("/")[0].strip())
        aliases.add(raw.split(",")[0].strip())
        cleaned_tokens = re.findall(r"[A-Za-z0-9]+", raw)
        cleaned = " ".join(cleaned_tokens).strip()
        if cleaned:
            aliases.add(cleaned)
            trimmed_tokens = [
                token
                for token in cleaned_tokens
                if token.lower() not in COMMON_GENERIC_NAME_TOKENS
            ]
            if trimmed_tokens:
                aliases.add(" ".join(trimmed_tokens))
                if len(trimmed_tokens[0]) >= 5:
                    aliases.add(trimmed_tokens[0])
    ticker_text = ticker.strip().upper()
    if ticker_text:
        aliases.add(ticker_text)
    return sorted(
        {alias for alias in aliases if len(alias) >= 3},
        key=lambda alias: (-len(alias), alias.lower()),
    )


def replace_alias(text: str, alias: str, replacement: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", alias)
    if not tokens:
        return text
    pattern = r"\b" + r"[\s&.,'()/:-]+".join(map(re.escape, tokens)) + r"\b"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def date_variants(day: dt.date) -> set[str]:
    month_full = day.strftime("%B")
    month_abbr = day.strftime("%b")
    return {
        day.isoformat(),
        f"{month_full} {day.day}, {day.year}",
        f"{month_full} {day.day:02d}, {day.year}",
        f"{month_abbr} {day.day}, {day.year}",
        f"{month_abbr}. {day.day}, {day.year}",
        f"{month_abbr} {day.day:02d}, {day.year}",
        f"{month_abbr}. {day.day:02d}, {day.year}",
    }


def mask_dates(text: str, days: list[dt.date], replacement: str) -> str:
    masked = text
    variants = sorted({variant for day in days for variant in date_variants(day)}, key=len, reverse=True)
    for variant in variants:
        masked = re.sub(re.escape(variant), replacement, masked, flags=re.IGNORECASE)
    return masked


def mask_executive_names(text: str) -> str:
    patterns = [
        (
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),\s+"
            r"(Chief Executive Officer|Chief Financial Officer|Chief Operating Officer|CEO|CFO|COO|President|Chair(?:man|person))\b",
            r"\2",
        ),
        (
            r"\b(Chief Executive Officer|Chief Financial Officer|Chief Operating Officer|CEO|CFO|COO|President|Chair(?:man|person))\s+"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
            r"\1",
        ),
    ]
    masked = text
    for pattern, replacement in patterns:
        masked = re.sub(pattern, replacement, masked)
    return masked


def mask_source_text(
    text: str,
    company: str,
    ticker: str,
    event_days: list[dt.date],
    accession_number: str,
    masked_company: str,
) -> str:
    masked = text
    for alias in company_aliases(company, ticker):
        masked = replace_alias(masked, alias, masked_company)
    accession = accession_number.strip()
    if accession:
        masked = re.sub(re.escape(accession), MASKED_ACCESSION, masked, flags=re.IGNORECASE)
        masked = re.sub(
            re.escape(accession.replace("-", "")),
            MASKED_ACCESSION,
            masked,
            flags=re.IGNORECASE,
        )
    masked = mask_dates(masked, event_days, MASKED_TIMESTAMP)
    masked = mask_executive_names(masked)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked


def looks_like_heading(tag_name: str, text: str) -> bool:
    if tag_name in HEADING_TAGS:
        return True
    lower = text.lower()
    if len(text) > 140:
        return False
    if text.endswith(":"):
        return True
    if text.isupper() and len(text.split()) <= 12:
        return True
    return any(
        phrase in lower
        for phrase in ("results", "financial highlights", "guidance", "outlook", "summary")
    ) and len(text.split()) <= 12


def classify_section(text: str, heading: str) -> str:
    haystack = f"{heading} {text}".lower()
    for section, patterns in SECTION_PATTERNS:
        if any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in patterns):
            return section
    return "general"


def should_skip_block(text: str) -> bool:
    if len(text) < 35:
        return True
    digit_count = sum(character.isdigit() for character in text)
    if (
        len(text) < 65
        and digit_count == 0
        and "." not in text
        and ":" not in text
        and "%" not in text
        and len(text.split()) <= 8
    ):
        return True
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SKIP_SOURCE_PATTERNS)


def extract_leaf_blocks(raw_text: str) -> list[tuple[str, str]]:
    unescaped = html.unescape(raw_text)
    soup = BeautifulSoup(unescaped, "html.parser")
    for tag in soup(["script", "style", "ix:header", "xbrl"]):
        tag.decompose()

    root = soup.body or soup
    blocks: list[tuple[str, str]] = []
    current_heading = ""
    seen: set[str] = set()
    for tag in root.find_all(BLOCK_TAGS):
        if tag.find_all(BLOCK_TAGS, recursive=False):
            continue
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        if looks_like_heading(tag.name, text):
            current_heading = text
            continue
        blocks.append((current_heading, text))
    return blocks


def fallback_source_units(cleaned_text: str) -> list[tuple[str, str]]:
    return [("", paragraph) for paragraph in split_paragraphs(cleaned_text) if len(paragraph) >= 35]


def find_exhibit_text(cik: int, accession: str, exhibit_url: str) -> tuple[str, str, str]:
    cache_path = CACHE_ROOT / "filings" / str(cik) / f"{accession}.txt"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    submission = cache_path.read_text(encoding="utf-8", errors="replace")
    expected_filename = exhibit_url.rstrip("/").split("/")[-1].lower()
    blocks = re.findall(
        r"<DOCUMENT>(.*?)</DOCUMENT>", submission, flags=re.IGNORECASE | re.DOTALL
    )
    fallback: tuple[str, str, str] | None = None
    for block in blocks:
        doc_type = extract_tag(block, "TYPE").upper()
        filename = extract_tag(block, "FILENAME")
        if not doc_type.startswith("EX-99"):
            continue
        text_match = re.search(
            r"<TEXT>(.*?)</TEXT>", block, flags=re.IGNORECASE | re.DOTALL
        )
        if not text_match:
            continue
        candidate = (doc_type, filename, text_match.group(1))
        if filename.lower() == expected_filename:
            return candidate
        if fallback is None:
            fallback = candidate
    if fallback is not None:
        return fallback
    raise ValueError(f"No EX-99.x text found for {accession}")


def infer_fiscal_period_from_text(text: str) -> str | None:
    cleaned = clean_html_text(text)[:3000]
    quarter_map = {
        "first": "Q1",
        "1st": "Q1",
        "q1": "Q1",
        "second": "Q2",
        "2nd": "Q2",
        "q2": "Q2",
        "third": "Q3",
        "3rd": "Q3",
        "q3": "Q3",
        "fourth": "Q4",
        "4th": "Q4",
        "q4": "Q4",
    }

    quarter_patterns = [
        re.compile(
            r"\b(?P<quarter>first|second|third|fourth|1st|2nd|3rd|4th|q[1-4])\s+quarter(?:\s+of|\s+for|\s+fiscal|\s+results\s+for|\s+ended)?\s*(?P<year>20\d{2})\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<year>20\d{2})\s+(?P<quarter>q[1-4]|first|second|third|fourth)\s+quarter\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:quarter|three months)\s+ended\s+(?P<month>[A-Za-z]+)\s+\d{1,2},\s+(?P<year>20\d{2})\b",
            flags=re.IGNORECASE,
        ),
    ]
    for pattern in quarter_patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        groups = match.groupdict()
        if "month" in groups and groups.get("month"):
            try:
                month = pd.to_datetime(groups["month"], format="%B").month
            except Exception:
                try:
                    month = pd.to_datetime(groups["month"], format="%b").month
                except Exception:
                    continue
            quarter = MONTH_TO_QUARTER.get(month)
            if quarter:
                return f"FY{groups['year']} {quarter}"
            continue
        quarter_raw = groups["quarter"].lower()
        quarter = quarter_map.get(quarter_raw)
        if quarter:
            return f"FY{groups['year']} {quarter}"

    full_year_match = re.search(
        r"\b(?:full year|fiscal year)\s+(?P<year>20\d{2})\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if full_year_match:
        return f"FY{full_year_match.group('year')}"
    return None


def format_xbrl_period(fact: dict[str, Any]) -> str:
    fiscal_year = normalized_str(fact.get("fy"))
    fiscal_period = normalized_str(fact.get("fp")).upper()
    if fiscal_year and fiscal_period:
        if fiscal_period == "FY":
            return f"FY{fiscal_year}"
        return f"{fiscal_year}{fiscal_period}"
    end_date = parse_date(fact.get("end"))
    if end_date and end_date.month in MONTH_TO_QUARTER:
        return f"{end_date.year}{MONTH_TO_QUARTER[end_date.month]}"
    return normalized_str(fact.get("end"))


def fiscal_period_target(fiscal_period: str | None) -> tuple[int, str] | None:
    if not fiscal_period:
        return None
    match = re.fullmatch(r"FY(?P<year>\d{4})(?:\s+(?P<period>Q[1-4]|FY))?", fiscal_period)
    if not match:
        return None
    return int(match.group("year")), normalized_str(match.group("period") or "FY").upper()


def load_companyfacts(cik: int) -> dict[str, Any] | None:
    path = CACHE_ROOT / "companyfacts" / f"CIK{cik:010d}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def choose_xbrl_fact(
    values: list[dict[str, Any]],
    accepted_at: dt.datetime | None,
    target_period: tuple[int, str] | None,
) -> dict[str, Any] | None:
    cutoff = accepted_at.date() if accepted_at is not None else None
    eligible = []
    for value in values:
        filed = parse_date(value.get("filed"))
        if cutoff is not None and (filed is None or filed > cutoff):
            continue
        eligible.append(value)
    if not eligible:
        return None

    if target_period is not None:
        year, period = target_period
        matched = [
            value
            for value in eligible
            if normalized_str(value.get("fy")) == str(year)
            and normalized_str(value.get("fp")).upper() == period
        ]
        if not matched:
            return None
        eligible = matched

    def sort_key(value: dict[str, Any]) -> tuple[dt.date, dt.date, int]:
        filed = parse_date(value.get("filed")) or dt.date.min
        end = parse_date(value.get("end")) or dt.date.min
        frame = 1 if normalized_str(value.get("fp")).upper().startswith("Q") else 0
        return filed, end, frame

    return sorted(eligible, key=sort_key)[-1]


def infer_fiscal_period_from_companyfacts(
    companyfacts: dict[str, Any] | None,
    accepted_at: dt.datetime | None,
) -> str | None:
    if not companyfacts:
        return None
    cutoff = accepted_at.date() if accepted_at is not None else None
    candidates: list[dict[str, Any]] = []
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    for tag in XBRL_TAG_PRIORITY[:6]:
        payload = us_gaap.get(tag, {})
        for values in payload.get("units", {}).values():
            for value in values:
                filed = parse_date(value.get("filed"))
                if cutoff is not None and (filed is None or filed > cutoff):
                    continue
                if normalized_str(value.get("fy")) and normalized_str(value.get("fp")):
                    candidates.append(value)
    if not candidates:
        return None
    candidates.sort(
        key=lambda value: (
            parse_date(value.get("filed")) or dt.date.min,
            parse_date(value.get("end")) or dt.date.min,
        ),
        reverse=True,
    )
    best = candidates[0]
    fiscal_year = normalized_str(best.get("fy"))
    fiscal_period = normalized_str(best.get("fp")).upper()
    if not fiscal_year or not fiscal_period:
        return None
    if fiscal_period.startswith("Q"):
        return f"FY{fiscal_year} {fiscal_period}"
    return f"FY{fiscal_year}"


def infer_fiscal_period(text: str, companyfacts: dict[str, Any] | None, accepted_at: dt.datetime | None) -> str | None:
    from_text = infer_fiscal_period_from_text(text)
    if from_text:
        return from_text
    return infer_fiscal_period_from_companyfacts(companyfacts, accepted_at)


def build_xbrl_facts(
    companyfacts: dict[str, Any] | None,
    accepted_at: dt.datetime | None,
    fiscal_period: str | None,
) -> list[dict[str, Any]]:
    if not companyfacts:
        return []

    target_period = fiscal_period_target(fiscal_period)
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    facts: list[dict[str, Any]] = []
    for tag in XBRL_TAG_PRIORITY:
        payload = us_gaap.get(tag)
        if not payload:
            continue
        selected: tuple[str, dict[str, Any]] | None = None
        for unit, values in sorted(payload.get("units", {}).items()):
            chosen = choose_xbrl_fact(values, accepted_at, target_period)
            if chosen is None:
                continue
            selected = (unit, chosen)
            break
        if selected is None:
            continue
        unit, chosen = selected
        facts.append(
            {
                "tag": tag,
                "period": format_xbrl_period(chosen),
                "value": json_scalar(chosen.get("val")),
                "unit": unit,
            }
        )

    ordered = []
    for index, fact in enumerate(facts, start=1):
        ordered.append(
            {
                "source_id": f"X{index:03d}",
                "tag": fact["tag"],
                "period": fact["period"],
                "value": fact["value"],
                "unit": fact["unit"],
            }
        )
    return ordered


def load_price_panel(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    raw = pd.read_csv(path)
    columns = {col.lower(): col for col in raw.columns}
    if {"ticker", "date", "adj_close"}.issubset(columns):
        df = raw.rename(
            columns={
                columns["ticker"]: "ticker",
                columns["date"]: "date",
                columns["adj_close"]: "adj_close",
            }
        )[["ticker", "date", "adj_close"]]
    elif {"ticker", "date", "close"}.issubset(columns):
        df = raw.rename(
            columns={
                columns["ticker"]: "ticker",
                columns["date"]: "date",
                columns["close"]: "adj_close",
            }
        )[["ticker", "date", "adj_close"]]
    else:
        raise ValueError(
            "Price CSV must be long format with ticker,date,adj_close columns "
            "(or ticker,date,close if already adjusted)."
        )
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    return df.dropna(subset=["ticker", "date", "adj_close"])


def reaction_day_from_prices(
    ticker: str,
    accepted_at: dt.datetime | None,
    price_panel: pd.DataFrame | None,
) -> dt.date | None:
    if price_panel is None or accepted_at is None:
        return None
    subset = price_panel[price_panel["ticker"] == ticker.upper()]
    if subset.empty:
        return None
    trading_dates = sorted(subset["date"].unique())
    accepted_date = accepted_at.date()
    accepted_after_close = after_close(accepted_at)
    for trading_date in trading_dates:
        if trading_date < accepted_date:
            continue
        if trading_date == accepted_date and accepted_after_close:
            continue
        return trading_date
    return None


def fallback_reaction_day(row: pd.Series, accepted_at: dt.datetime | None) -> dt.date | None:
    if accepted_at is not None:
        event_day = accepted_at.date()
    else:
        event_day = parse_date(row.get("event_date"))
    if event_day is None:
        return None
    if event_day.weekday() >= 5:
        return next_business_day(event_day)
    if accepted_at is not None and after_close(accepted_at):
        return next_business_day(event_day)
    return event_day


def compute_reaction_day_t0(
    row: pd.Series, accepted_at: dt.datetime | None, price_panel: pd.DataFrame | None
) -> dt.date | None:
    price_based = reaction_day_from_prices(normalized_str(row.get("ticker")), accepted_at, price_panel)
    if price_based is not None:
        return price_based
    return fallback_reaction_day(row, accepted_at)


def round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def compute_pre_event_price_context(
    row: pd.Series,
    accepted_at: dt.datetime | None,
    reaction_day: dt.date | None,
    price_panel: pd.DataFrame | None,
) -> dict[str, float | None]:
    direct_columns = {"ret_5d", "ret_20d", "market_ret_20d"}
    if direct_columns.issubset(row.index):
        values = {column: json_scalar(row.get(column)) for column in sorted(direct_columns)}
        if any(value is not None for value in values.values()):
            return {
                "ret_5d": round_or_none(values["ret_5d"]),
                "ret_20d": round_or_none(values["ret_20d"]),
                "market_ret_20d": round_or_none(values["market_ret_20d"]),
            }

    empty = {"ret_5d": None, "ret_20d": None, "market_ret_20d": None}
    ticker = normalized_str(row.get("ticker")).upper()
    if price_panel is None or reaction_day is None or not ticker:
        return empty

    subset = price_panel[price_panel["ticker"].isin([ticker, "SPY"])].copy()
    if subset.empty or set(subset["ticker"]) != {ticker, "SPY"}:
        return empty

    wide = subset.pivot_table(index="date", columns="ticker", values="adj_close")
    wide = wide.sort_index().dropna(subset=[ticker, "SPY"])
    if reaction_day not in wide.index:
        return empty

    loc = wide.index.get_loc(reaction_day)
    if isinstance(loc, slice):
        return empty
    pre_end = int(loc) - 1
    if pre_end < 20:
        return empty

    def cumulative_return(symbol: str, window: int) -> float | None:
        start = pre_end - window
        if start < 0:
            return None
        start_price = wide.iloc[start][symbol]
        end_price = wide.iloc[pre_end][symbol]
        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            return None
        return float(end_price / start_price - 1.0)

    return {
        "ret_5d": round_or_none(cumulative_return(ticker, 5)),
        "ret_20d": round_or_none(cumulative_return(ticker, 20)),
        "market_ret_20d": round_or_none(cumulative_return("SPY", 20)),
    }


def json_scalar(value: Any) -> Any:
    if is_missing(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else float(value)
    return value


def build_source_units(
    raw_text: str,
    company: str,
    ticker: str,
    event_days: list[dt.date],
    accession_number: str,
    masked_company: str,
) -> list[dict[str, Any]]:
    blocks = extract_leaf_blocks(raw_text)
    if not blocks:
        blocks = fallback_source_units(clean_html_text(raw_text))

    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for heading, text in blocks:
        masked_heading = mask_source_text(
            heading, company, ticker, event_days, accession_number, masked_company
        )
        masked_text = mask_source_text(
            text, company, ticker, event_days, accession_number, masked_company
        )
        if should_skip_block(masked_text):
            continue
        dedupe_key = masked_text.lower()
        if dedupe_key in seen:
            continue
        if len(masked_text) < 140 and any(
            masked_text in previous["text"] for previous in units[-12:]
        ):
            continue
        seen.add(dedupe_key)
        units.append(
            {
                "section": classify_section(masked_text, masked_heading),
                "text": masked_text,
                "heading": masked_heading,
            }
        )

    if not units:
        masked_full_text = mask_source_text(
            clean_html_text(raw_text), company, ticker, event_days, accession_number, masked_company
        )
        for paragraph in split_paragraphs(masked_full_text):
            if should_skip_block(paragraph):
                continue
            units.append({"section": classify_section(paragraph, ""), "text": paragraph, "heading": ""})

    ordered = []
    for index, unit in enumerate(units, start=1):
        ordered.append(
            {
                "source_id": f"S{index:03d}",
                "source_type": "earnings_release",
                "section": unit["section"],
                "text": unit["text"],
                "token_count": rough_token_count(unit["text"]),
            }
        )
    return ordered


def parse_items(value: Any) -> list[str]:
    raw = normalized_str(value)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def choose_hidden_join_fields(events: pd.DataFrame, hidden: pd.DataFrame) -> list[str]:
    candidates = [
        ["event_id"],
        ["accession_number"],
        ["ticker", "cik", "event_date"],
        ["ticker", "accepted_at_et"],
        ["ticker", "event_date"],
    ]
    best_fields: list[str] | None = None
    best_overlap = 0

    for fields in candidates:
        if not all(field in events.columns and field in hidden.columns for field in fields):
            continue
        event_keys = {tuple(normalized_str(row[field]) for field in fields) for _, row in events.iterrows()}
        hidden_keys = {tuple(normalized_str(row[field]) for field in fields) for _, row in hidden.iterrows()}
        overlap = len(event_keys.intersection(hidden_keys) - {tuple("" for _ in fields)})
        if overlap > best_overlap:
            best_overlap = overlap
            best_fields = fields
    if best_fields is None or best_overlap == 0:
        raise ValueError("Could not find a shared join key between events and hidden outcomes CSVs.")
    return best_fields


def build_hidden_sidecar(
    events: pd.DataFrame,
    hidden_csv: str,
) -> list[dict[str, Any]]:
    if not hidden_csv:
        return []
    hidden = pd.read_csv(hidden_csv)
    join_fields = choose_hidden_join_fields(events, hidden)

    def row_key(row: pd.Series) -> tuple[str, ...]:
        return tuple(normalized_str(row[field]) for field in join_fields)

    hidden_map: dict[tuple[str, ...], dict[str, Any]] = {}
    for _, row in hidden.iterrows():
        key = row_key(row)
        if not any(key):
            continue
        if key in hidden_map:
            raise ValueError(f"Duplicate hidden-outcome key for fields {join_fields}: {key}")
        hidden_map[key] = row.to_dict()

    sidecar_rows = []
    for _, event_row in events.iterrows():
        key = row_key(event_row)
        hidden_row = hidden_map.get(key)
        if hidden_row is None:
            continue
        payload = {"event_id": normalized_str(event_row.get("event_id"))}
        for column, value in hidden_row.items():
            if column in join_fields or column == "event_id":
                continue
            payload[column] = json_scalar(value)
        sidecar_rows.append(payload)
    return sidecar_rows


def manifest_row_base(row: pd.Series, event_id: str, packet_path: str) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "ticker": normalized_str(row.get("ticker")),
        "company": normalized_str(row.get("company")),
        "sector": normalized_str(row.get("sector")),
        "market_cap_group": normalized_str(row.get("market_cap_group")),
        "cik": normalized_str(row.get("cik")),
        "event_date": normalized_str(row.get("event_date")),
        "accepted_at_et": normalized_str(row.get("accepted_at_et")),
        "accession_number": normalized_str(row.get("accession_number")),
        "form_type": normalized_str(row.get("form_type")),
        "items": normalized_str(row.get("items")),
        "packet_path": packet_path,
    }


def main() -> int:
    args = parse_args()

    events = pd.read_csv(args.events_csv)
    if args.limit:
        events = events.head(args.limit)
    events = assign_event_ids(events)

    embedded_hidden = sorted(column for column in HIDDEN_INPUT_COLUMNS if column in events.columns)
    if embedded_hidden:
        events = events.drop(columns=embedded_hidden)

    output_dir = Path(args.output_dir)
    packet_dir = output_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)

    price_panel = load_price_panel(args.prices_csv) if args.prices_csv else None
    company_placeholders = build_company_placeholders(events)

    packets: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    errors = 0
    empty_xbrl_count = 0
    empty_price_context_count = 0

    for _, row in events.iterrows():
        event_id = normalized_str(row.get("event_id"))
        packet_path = packet_dir / f"{event_id}.json"
        base = manifest_row_base(row, event_id, str(packet_path))
        try:
            cik = int(float(row["cik"]))
            accession_number = normalized_str(row.get("accession_number"))
            exhibit_url = normalized_str(row.get("exhibit_url"))
            accepted_at = parse_acceptance_datetime(row.get("accepted_at_et"))
            reaction_day = compute_reaction_day_t0(row, accepted_at, price_panel)

            _, _, exhibit_text = find_exhibit_text(cik, accession_number, exhibit_url)
            company = normalized_str(row.get("company"))
            ticker = normalized_str(row.get("ticker"))
            event_days = [day for day in {parse_date(row.get("event_date")), accepted_at.date() if accepted_at else None} if day is not None]
            masked_company = company_placeholders["|".join(issuer_key(row))]

            companyfacts = load_companyfacts(cik)
            fiscal_period = infer_fiscal_period(exhibit_text, companyfacts, accepted_at)
            source_units = build_source_units(
                exhibit_text,
                company,
                ticker,
                event_days,
                accession_number,
                masked_company,
            )
            xbrl_facts = build_xbrl_facts(companyfacts, accepted_at, fiscal_period)
            pre_event_price_context = compute_pre_event_price_context(
                row, accepted_at, reaction_day, price_panel
            )

            if not xbrl_facts:
                empty_xbrl_count += 1
            if all(value is None for value in pre_event_price_context.values()):
                empty_price_context_count += 1

            packet = {
                "event_id": event_id,
                "masked_company": masked_company,
                "sector": normalized_str(row.get("sector")),
                "market_cap_group": normalized_str(row.get("market_cap_group")),
                "fiscal_period": fiscal_period,
                "filing_metadata": {
                    "cik": MASKED_CIK,
                    "accession_number": MASKED_ACCESSION,
                    "form_type": normalized_str(row.get("form_type")),
                    "accepted_at_et": MASKED_TIMESTAMP,
                    "reaction_day_t0": MASKED_REACTION_DAY,
                    "after_close": after_close(accepted_at),
                    "items": parse_items(row.get("items")),
                },
                "source_units": source_units,
                "xbrl_facts": xbrl_facts,
                "pre_event_price_context": pre_event_price_context,
            }
            packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
            packets.append(packet)

            manifest = {
                **base,
                "masked_company": masked_company,
                "fiscal_period": fiscal_period or "",
                "reaction_day_t0": reaction_day.isoformat() if reaction_day else "",
                "source_unit_count": len(source_units),
                "xbrl_fact_count": len(xbrl_facts),
                "build_ok": True,
                "error": "",
            }
            manifest_rows.append(manifest)
        except Exception as exc:
            errors += 1
            manifest_rows.append(
                {
                    **base,
                    "masked_company": "",
                    "fiscal_period": "",
                    "reaction_day_t0": "",
                    "source_unit_count": 0,
                    "xbrl_fact_count": 0,
                    "build_ok": False,
                    "error": str(exc),
                }
            )

    source_packets_path = output_dir / "source_packets.jsonl"
    with source_packets_path.open("w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet) + "\n")

    manifest_path = output_dir / "packet_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "event_id",
            "ticker",
            "company",
            "sector",
            "market_cap_group",
            "cik",
            "event_date",
            "accepted_at_et",
            "accession_number",
            "form_type",
            "items",
            "masked_company",
            "fiscal_period",
            "reaction_day_t0",
            "source_unit_count",
            "xbrl_fact_count",
            "packet_path",
            "build_ok",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    hidden_sidecar_rows = build_hidden_sidecar(events, args.hidden_outcomes_csv)
    if hidden_sidecar_rows:
        hidden_path = output_dir / "hidden_outcomes.jsonl"
        with hidden_path.open("w", encoding="utf-8") as handle:
            for row in hidden_sidecar_rows:
                handle.write(json.dumps(row) + "\n")
        print(f"wrote hidden outcomes sidecar to {hidden_path}")

    print(f"wrote {len(packets)} source packets to {packet_dir}")
    print(f"wrote aggregate packets to {source_packets_path}")
    print(f"wrote manifest to {manifest_path}")
    print(f"events with empty xbrl_facts: {empty_xbrl_count}")
    print(f"events with empty pre_event_price_context: {empty_price_context_count}")
    if embedded_hidden:
        print(
            "ignored hidden-like columns embedded in the events CSV: "
            + ", ".join(embedded_hidden)
        )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
