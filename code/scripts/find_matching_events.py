#!/usr/bin/env python3
"""Build an SEC earnings-event screening table for the LLM leakage study.

This script intentionally treats price data as an injected input rather than a
hard-coded web source. SEC provides filing metadata and XBRL facts, but it does
not provide adjusted daily stock prices. If a local price panel is supplied, the
script computes market-adjusted CARs using SPY as the benchmark; otherwise CAR
fields are left blank and the output is SEC-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup


START_DATE = dt.date(2022, 1, 1)
END_DATE = dt.date(2025, 12, 31)
EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_UNIVERSE_FILE = PROJECT_ROOT / "data" / "reference" / "screening_universe_2026.csv"
DEFAULT_SELECTION_TARGETS = {"large_cap": 50, "small_mid_cap": 50}

SEC_BASE = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

MANUAL_TICKER_METADATA = {
    # SEC company_tickers.json is a current ticker map. These historical tickers
    # can disappear after merger/delisting but still belong in the 2022-2025
    # candidate universe.
    "IPG": {"cik": 51644, "company": "INTERPUBLIC GROUP OF COMPANIES, INC."},
    "K": {"cik": 55067, "company": "KELLANOVA"},
}

COMPANY_SCREEN_COLUMNS = [
    "ticker",
    "company",
    "sector",
    "market_cap_group",
    "cik",
    "submissions_ok",
    "xbrl_ok",
    "domestic_reporting",
    "fpi_like",
    "has_8k",
    "company_screen_pass",
    "notes",
]

SCREENING_COLUMNS = [
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
    "has_item_2_02",
    "has_exhibit_99_1",
    "exhibit_url",
    "parse_ok",
    "xbrl_ok",
    "price_ok",
    "source_token_count",
    "CAR_1_5",
    "CAR_1_20",
    "hidden_valence",
    "masking_ok",
    "memory_probe_pass",
    "exclusion_reason",
    "notes",
]

LEAKAGE_TERMS = [
    "iPhone",
    "Azure",
    "H100",
    "Cybertruck",
    "Ozempic",
    "Disney+",
    "Prime Day",
    "Marvel",
    "Pixar",
    "ESPN",
    "HBO",
    "Max Original",
    "Call of Duty",
    "Grand Theft Auto",
    "FIFA Ultimate Team",
    "Madden NFL",
    "F-150",
    "Corvette",
    "Starbucks Rewards",
    "McCafe",
    "Keytruda",
    "Humira",
    "Skyrizi",
    "Stelara",
]

DIRTY_PATTERNS: dict[str, list[str]] = {
    "merger_or_acquisition_announcement_dominates": [
        r"\bdefinitive merger agreement\b",
        r"\bdefinitive acquisition agreement\b",
        r"\bto be acquired by\b",
        r"\bentered into a merger agreement\b",
        r"\btender offer\b",
        r"\bgoing private transaction\b",
    ],
    "bankruptcy_or_going_concern_shock": [
        r"\bchapter 11\b",
        r"\bbankruptcy\b",
        r"\bgoing concern\b",
        r"\bliquidity crisis\b",
    ],
    "fraud_or_major_litigation_event": [
        r"\bfraud investigation\b",
        r"\baccounting irregularit",
        r"\brestatement investigation\b",
        r"\bcriminal investigation\b",
        r"\bsecurities litigation\b",
    ],
    "major_regulatory_binary_event": [
        r"\bantitrust lawsuit\b",
        r"\bregulatory approval denied\b",
        r"\bconsent decree\b",
        r"\bexport ban\b",
    ],
    "FDA_approval_or_rejection_binary_event": [
        r"\bfda approval\b",
        r"\bfda approved\b",
        r"\bfda rejection\b",
        r"\bcomplete response letter\b",
        r"\bcrl\b",
    ],
    "activist_takeover_or_proxy_fight": [
        r"\bproxy fight\b",
        r"\bactivist investor\b",
        r"\bproxy contest\b",
        r"\bpoison pill\b",
    ],
    "CEO_scandal_or_sudden_resignation_dominates": [
        r"\bceo resigns\b",
        r"\bchief executive officer resigns\b",
        r"\bterminated for cause\b",
        r"\bmisconduct\b",
    ],
}


@dataclass
class FilingDocument:
    doc_type: str
    sequence: str
    filename: str
    description: str
    text: str


class SecClient:
    def __init__(self, user_agent: str, min_interval: float = 0.12) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/plain,text/html,*/*",
            }
        )
        self.min_interval = min_interval
        self.last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def get_text(self, url: str, cache_path: Path) -> str:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")

        self._wait()
        response = self.session.get(url, timeout=45)
        self.last_request = time.monotonic()
        response.raise_for_status()
        text = response.text
        cache_path.write_text(text, encoding="utf-8")
        return text

    def get_json(self, url: str, cache_path: Path) -> Any:
        text = self.get_text(url, cache_path)
        return json.loads(text)
def load_universe_by_ticker(path: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_csv(path)
    required = {"ticker", "sector", "market_cap_group"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Universe file missing required columns: {sorted(missing)}")

    members: dict[str, dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        ticker = str(row.get("ticker", "")).strip().upper()
        sector = str(row.get("sector", "")).strip()
        market_cap_group = str(row.get("market_cap_group", "")).strip()
        if not ticker or not sector or not market_cap_group:
            raise ValueError(f"Universe row has blank fields: {row}")
        previous = members.get(ticker)
        current = {
            "sector": sector,
            "market_cap_group": market_cap_group,
        }
        if previous is not None and previous != current:
            raise ValueError(f"Conflicting universe definition for {ticker}")
        members[ticker] = current
    if not members:
        raise ValueError(f"No universe members found in {path}")
    return members


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_acceptance_datetime(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    value = value.strip()
    if value.endswith("Z"):
        for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"]:
            try:
                parsed = dt.datetime.strptime(value, fmt)
                return parsed.replace(tzinfo=UTC).astimezone(EASTERN)
            except ValueError:
                pass
    try:
        parsed = dt.datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            return parsed.astimezone(EASTERN)
    except ValueError:
        pass
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y%m%d%H%M%S"]:
        try:
            parsed = dt.datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=EASTERN)
        except ValueError:
            pass
    return None


def format_acceptance_et(value: str | None) -> str:
    parsed = parse_acceptance_datetime(value)
    return parsed.isoformat() if parsed else (value or "")


def load_ticker_metadata(client: SecClient) -> dict[str, dict[str, Any]]:
    raw = client.get_json(
        "https://www.sec.gov/files/company_tickers.json",
        CACHE_DIR / "sec_company_tickers.json",
    )
    output = {}
    for row in raw.values():
        ticker = row["ticker"].upper()
        output[ticker] = {
            "cik": int(row["cik_str"]),
            "company": row["title"],
        }
    return output


def filing_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return []
    keys = list(recent.keys())
    n_rows = len(recent.get("accessionNumber", []))
    rows = []
    for idx in range(n_rows):
        row = {}
        for key in keys:
            values = recent.get(key, [])
            row[key] = values[idx] if idx < len(values) else None
        rows.append(row)
    return rows


def load_submissions(
    client: SecClient,
    cik: int,
    start_date: dt.date,
    end_date: dt.date,
) -> list[dict[str, Any]]:
    cik_padded = f"{cik:010d}"
    payload = client.get_json(
        f"{SEC_BASE}/submissions/CIK{cik_padded}.json",
        CACHE_DIR / "submissions" / f"CIK{cik_padded}.json",
    )
    rows = filing_rows_from_payload(payload)

    for file_info in payload.get("filings", {}).get("files", []):
        filing_from = parse_date(file_info.get("filingFrom"))
        filing_to = parse_date(file_info.get("filingTo"))
        if filing_from and filing_from > end_date:
            continue
        if filing_to and filing_to < start_date:
            continue
        name = file_info.get("name")
        if not name:
            continue
        more = client.get_json(
            f"{SEC_BASE}/submissions/{name}",
            CACHE_DIR / "submissions" / name,
        )
        rows.extend(filing_rows_from_payload({"filings": {"recent": more}}))

    deduped = {}
    for row in rows:
        accession = row.get("accessionNumber")
        if accession:
            deduped[accession] = row
    return list(deduped.values())


def load_companyfacts_ok(client: SecClient, cik: int) -> bool:
    cik_padded = f"{cik:010d}"
    try:
        facts = client.get_json(
            f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik_padded}.json",
            CACHE_DIR / "companyfacts" / f"CIK{cik_padded}.json",
        )
    except requests.HTTPError:
        return False
    return bool(facts.get("facts"))


def item_2_02_present(items: Any) -> bool:
    if not items:
        return False
    return "2.02" in str(items)


def archive_text_url(cik: int, accession_number: str) -> str:
    accession_no_dashes = accession_number.replace("-", "")
    return f"{SEC_ARCHIVES}/{cik}/{accession_no_dashes}/{accession_number}.txt"


def document_url(cik: int, accession_number: str, filename: str) -> str:
    accession_no_dashes = accession_number.replace("-", "")
    return f"{SEC_ARCHIVES}/{cik}/{accession_no_dashes}/{filename}"


def extract_tag(block: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*([^\n\r<]+)", block, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_filing_documents(submission_text: str) -> list[FilingDocument]:
    documents = []
    blocks = re.findall(
        r"<DOCUMENT>(.*?)</DOCUMENT>", submission_text, flags=re.IGNORECASE | re.DOTALL
    )
    for block in blocks:
        text_match = re.search(
            r"<TEXT>(.*?)</TEXT>", block, flags=re.IGNORECASE | re.DOTALL
        )
        documents.append(
            FilingDocument(
                doc_type=extract_tag(block, "TYPE").upper(),
                sequence=extract_tag(block, "SEQUENCE"),
                filename=extract_tag(block, "FILENAME"),
                description=extract_tag(block, "DESCRIPTION"),
                text=text_match.group(1) if text_match else "",
            )
        )
    return documents


def is_pdf_or_image(filename: str, text: str) -> bool:
    lower = filename.lower()
    if lower.endswith((".pdf", ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff")):
        return True
    return text.startswith("%PDF")


def choose_earnings_exhibit(documents: list[FilingDocument]) -> FilingDocument | None:
    candidates = [doc for doc in documents if doc.doc_type.startswith("EX-99")]
    if not candidates:
        return None

    def score(doc: FilingDocument) -> tuple[int, str]:
        haystack = f"{doc.doc_type} {doc.filename} {doc.description}".lower()
        value = 0
        if doc.doc_type == "EX-99.1":
            value += 100
        if "99.1" in doc.filename.lower():
            value += 60
        if any(word in haystack for word in ["earn", "result", "release", "quarter"]):
            value += 25
        if is_pdf_or_image(doc.filename, doc.text):
            value -= 200
        return value, doc.filename

    return sorted(candidates, key=score, reverse=True)[0]


def clean_source_text(raw_text: str) -> str:
    unescaped = html.unescape(raw_text)
    if "<html" in unescaped.lower() or re.search(r"<[a-zA-Z][^>]*>", unescaped):
        soup = BeautifulSoup(unescaped, "html.parser")
        for tag in soup(["script", "style", "ix:header", "xbrl"]):
            tag.decompose()
        text = soup.get_text(" ")
    else:
        text = unescaped
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def rough_token_count(text: str) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / 4))


def dirty_reasons(text: str) -> list[str]:
    first_chars = text[:5000].lower()
    reasons = []
    for reason, patterns in DIRTY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, first_chars, flags=re.IGNORECASE):
                reasons.append(reason)
                break
    return reasons


def leakage_clues(text: str) -> list[str]:
    lower = text.lower()
    clues = []
    for term in LEAKAGE_TERMS:
        if term.lower() in lower:
            clues.append(term)
    return clues


def company_screen_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forms = {str(row.get("form", "")).upper() for row in rows}
    domestic_reporting = bool(forms.intersection({"10-K", "10-Q", "8-K", "8-K/A"}))
    fpi_like = "6-K" in forms and not forms.intersection({"10-K", "10-Q", "8-K"})
    return {
        "domestic_reporting": domestic_reporting,
        "fpi_like": fpi_like,
        "has_8k": bool(forms.intersection({"8-K", "8-K/A"})),
        "forms_seen": ";".join(sorted(form for form in forms if form)),
    }


def load_price_panel(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    raw = pd.read_csv(path)
    cols = {col.lower(): col for col in raw.columns}
    if {"ticker", "date", "adj_close"}.issubset(cols):
        df = raw.rename(
            columns={
                cols["ticker"]: "ticker",
                cols["date"]: "date",
                cols["adj_close"]: "adj_close",
            }
        )[["ticker", "date", "adj_close"]]
    elif {"ticker", "date", "close"}.issubset(cols):
        df = raw.rename(
            columns={
                cols["ticker"]: "ticker",
                cols["date"]: "date",
                cols["close"]: "adj_close",
            }
        )[["ticker", "date", "adj_close"]]
    else:
        raise ValueError(
            "Price CSV must be long format with ticker,date,adj_close columns "
            "(or ticker,date,close if already adjusted)."
        )
    df["ticker"] = df["ticker"].str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    return df.dropna(subset=["ticker", "date", "adj_close"])


def compute_car_fields(
    event_row: dict[str, Any], price_panel: pd.DataFrame | None
) -> dict[str, Any]:
    empty = {
        "price_ok": False if price_panel is not None else "",
        "CAR_1_5": "",
        "CAR_1_20": "",
        "hidden_valence": "unknown",
    }
    if price_panel is None:
        return empty

    ticker = event_row["ticker"]
    accepted = parse_acceptance_datetime(event_row["accepted_at_et"])
    if accepted is None:
        return empty

    subset = price_panel[price_panel["ticker"].isin([ticker, "SPY"])].copy()
    if subset.empty or set(subset["ticker"]) != {ticker, "SPY"}:
        return empty

    wide = subset.pivot_table(index="date", columns="ticker", values="adj_close")
    wide = wide.sort_index().dropna(subset=[ticker, "SPY"])
    wide["stock_ret"] = wide[ticker].pct_change()
    wide["market_ret"] = wide["SPY"].pct_change()
    trading_dates = list(wide.index)
    if not trading_dates:
        return empty

    accepted_date = accepted.date()
    after_close = accepted.timetz() > dt.time(16, 0, tzinfo=EASTERN)
    t0 = None
    for trading_date in trading_dates:
        if trading_date < accepted_date:
            continue
        if trading_date == accepted_date and after_close:
            continue
        t0 = trading_date
        break
    if t0 is None or t0 not in wide.index:
        return empty

    loc = wide.index.get_loc(t0)
    if loc < 20 or loc + 20 >= len(wide):
        return empty

    def window_car(horizon: int) -> float | None:
        window = wide.iloc[loc + 1 : loc + horizon + 1]
        if len(window) != horizon:
            return None
        if window[["stock_ret", "market_ret"]].isna().any().any():
            return None
        return float(window["stock_ret"].sum() - window["market_ret"].sum())

    car_5 = window_car(5)
    car_20 = window_car(20)
    if car_5 is None or car_20 is None:
        return empty

    if car_5 >= 0.03:
        valence = "positive"
    elif car_5 <= -0.03:
        valence = "negative"
    else:
        valence = "mixed"
    return {
        "price_ok": True,
        "CAR_1_5": round(car_5, 6),
        "CAR_1_20": round(car_20, 6),
        "hidden_valence": valence,
    }


def build_screening_rows(
    args: argparse.Namespace, universe_by_ticker: dict[str, dict[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = SecClient(args.user_agent, min_interval=args.sec_delay)
    ticker_meta = load_ticker_metadata(client)
    price_panel = load_price_panel(Path(args.prices_csv)) if args.prices_csv else None

    event_rows = []
    company_rows = []
    tickers = sorted(universe_by_ticker)
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx:03d}/{len(tickers)}] {ticker}", file=sys.stderr, flush=True)
        member = universe_by_ticker[ticker]
        meta = ticker_meta.get(ticker) or MANUAL_TICKER_METADATA.get(ticker)
        if not meta:
            company_rows.append(
                {
                    "ticker": ticker,
                    "company": "",
                    "sector": member["sector"],
                    "market_cap_group": member["market_cap_group"],
                    "cik": "",
                    "submissions_ok": False,
                    "xbrl_ok": False,
                    "domestic_reporting": False,
                    "fpi_like": "",
                    "has_8k": False,
                    "company_screen_pass": False,
                    "notes": "ticker_not_found_in_sec_company_tickers",
                }
            )
            continue

        cik = meta["cik"]
        company = meta["company"]
        try:
            filings = load_submissions(
                client,
                cik,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            submissions_ok = True
        except Exception as exc:
            filings = []
            submissions_ok = False
            submissions_error = str(exc)
        else:
            submissions_error = ""

        xbrl_ok = load_companyfacts_ok(client, cik) if submissions_ok else False
        company_status = company_screen_status(filings)
        company_pass = (
            submissions_ok
            and xbrl_ok
            and company_status["domestic_reporting"]
            and company_status["has_8k"]
            and not company_status["fpi_like"]
        )
        company_rows.append(
            {
                "ticker": ticker,
                "company": company,
                "sector": member["sector"],
                "market_cap_group": member["market_cap_group"],
                "cik": cik,
                "submissions_ok": submissions_ok,
                "xbrl_ok": xbrl_ok,
                "domestic_reporting": company_status["domestic_reporting"],
                "fpi_like": company_status["fpi_like"],
                "has_8k": company_status["has_8k"],
                "company_screen_pass": company_pass,
                "notes": submissions_error or company_status["forms_seen"],
            }
        )
        if not company_pass:
            continue

        for filing in filings:
            form = str(filing.get("form", "")).upper()
            if form not in {"8-K", "8-K/A"}:
                continue
            filing_date = parse_date(filing.get("filingDate"))
            if filing_date is None or filing_date < args.start_date or filing_date > args.end_date:
                continue
            items = filing.get("items", "")
            has_item = item_2_02_present(items)
            if not has_item:
                continue
            accession = filing.get("accessionNumber")
            if not accession:
                continue

            row = {
                "ticker": ticker,
                "company": company,
                "sector": member["sector"],
                "market_cap_group": member["market_cap_group"],
                "cik": cik,
                "event_date": filing.get("filingDate", ""),
                "accepted_at_et": format_acceptance_et(filing.get("acceptanceDateTime")),
                "accession_number": accession,
                "form_type": form,
                "items": items,
                "has_item_2_02": has_item,
                "has_exhibit_99_1": False,
                "exhibit_url": "",
                "parse_ok": False,
                "xbrl_ok": xbrl_ok,
                "price_ok": "",
                "source_token_count": "",
                "CAR_1_5": "",
                "CAR_1_20": "",
                "hidden_valence": "unknown",
                "masking_ok": False,
                "memory_probe_pass": "not_run",
                "exclusion_reason": "",
                "notes": "",
            }

            try:
                submission_text = client.get_text(
                    archive_text_url(cik, accession),
                    CACHE_DIR / "filings" / str(cik) / f"{accession}.txt",
                )
            except Exception as exc:
                row["exclusion_reason"] = "filing_text_fetch_failed"
                row["notes"] = str(exc)
                row.update(compute_car_fields(row, price_panel))
                event_rows.append(row)
                continue

            docs = parse_filing_documents(submission_text)
            row["has_exhibit_99_1"] = any(doc.doc_type == "EX-99.1" for doc in docs)
            exhibit = choose_earnings_exhibit(docs)
            if exhibit is None:
                row["exclusion_reason"] = "no_exhibit_99_x"
                row.update(compute_car_fields(row, price_panel))
                event_rows.append(row)
                continue

            row["exhibit_url"] = document_url(cik, accession, exhibit.filename)
            if is_pdf_or_image(exhibit.filename, exhibit.text):
                row["exclusion_reason"] = "exhibit_only_pdf_or_image_without_clean_text"
                row["notes"] = f"{exhibit.doc_type}:{exhibit.filename}"
                row.update(compute_car_fields(row, price_panel))
                event_rows.append(row)
                continue

            source_text = clean_source_text(exhibit.text)
            token_count = rough_token_count(source_text)
            row["source_token_count"] = token_count
            row["parse_ok"] = token_count >= args.min_tokens

            reasons = []
            if not row["parse_ok"]:
                reasons.append("unparseable_exhibit")
            if token_count > args.max_tokens:
                reasons.append("source_text_over_token_budget")
            reasons.extend(dirty_reasons(source_text))
            clues = leakage_clues(source_text)
            row["masking_ok"] = not clues and token_count <= args.max_tokens and row["parse_ok"]
            if clues:
                reasons.append("packet_contains_unmaskable_unique_identifiers")

            price_fields = compute_car_fields(row, price_panel)
            row.update(price_fields)
            if price_panel is not None and not row["price_ok"]:
                reasons.append("no_valid_price_window")

            row["exclusion_reason"] = ";".join(sorted(set(reasons)))
            notes = [f"{exhibit.doc_type}:{exhibit.filename}"]
            if clues:
                notes.append("leakage_clues=" + "|".join(clues))
            row["notes"] = ";".join(notes)
            event_rows.append(row)

    events = pd.DataFrame(event_rows)
    companies = pd.DataFrame(company_rows)
    if companies.empty:
        companies = pd.DataFrame(columns=COMPANY_SCREEN_COLUMNS)
    else:
        companies = companies[COMPANY_SCREEN_COLUMNS]
    if events.empty:
        events = pd.DataFrame(columns=SCREENING_COLUMNS)
    else:
        events = events[SCREENING_COLUMNS]
    return companies, events


def eligible_for_candidate(row: pd.Series, require_price: bool) -> bool:
    if row.get("exclusion_reason"):
        return False
    if not bool(row.get("parse_ok")):
        return False
    if not bool(row.get("xbrl_ok")):
        return False
    if not bool(row.get("has_item_2_02")):
        return False
    if require_price and not bool(row.get("price_ok")):
        return False
    if not bool(row.get("masking_ok")):
        return False
    return True


def choose_diverse_rows(group: pd.DataFrame, n: int) -> pd.DataFrame:
    if group.empty or n <= 0:
        return group.iloc[0:0]
    work = group.copy()
    work["event_date_sort"] = pd.to_datetime(work["event_date"], errors="coerce")
    work["source_token_count_num"] = pd.to_numeric(
        work["source_token_count"], errors="coerce"
    ).fillna(0)
    work = work.sort_values(
        ["ticker", "event_date_sort", "source_token_count_num"],
        ascending=[True, False, False],
    )

    picked = []
    per_ticker = {}
    max_per_ticker = max(1, math.ceil(n / max(1, work["ticker"].nunique())))
    for _, row in work.iterrows():
        ticker = row["ticker"]
        if per_ticker.get(ticker, 0) >= max_per_ticker:
            continue
        picked.append(row)
        per_ticker[ticker] = per_ticker.get(ticker, 0) + 1
        if len(picked) >= n:
            break
    if len(picked) < n:
        already = {row["accession_number"] for row in picked}
        for _, row in work.iterrows():
            if row["accession_number"] in already:
                continue
            picked.append(row)
            if len(picked) >= n:
                break
    return pd.DataFrame(picked).drop(columns=["event_date_sort", "source_token_count_num"])


def choose_time_balanced_rows(group: pd.DataFrame, n: int) -> pd.DataFrame:
    if group.empty or n <= 0:
        return group.iloc[0:0]
    work = group.copy()
    work["year"] = pd.to_datetime(work["event_date"], errors="coerce").dt.year
    years = [
        int(year)
        for year in sorted(work["year"].dropna().astype(int).unique().tolist())
    ]
    if not years:
        return choose_diverse_rows(group, n)
    per_year = max(1, n // len(years))
    selected = []
    used_accessions = set()
    for year in years:
        year_df = work[work["year"] == year]
        year_df = year_df[~year_df["accession_number"].isin(used_accessions)]
        picked = choose_diverse_rows(year_df, per_year)
        selected.append(picked)
        used_accessions.update(picked["accession_number"].tolist())

    combined = pd.concat(selected, ignore_index=True) if selected else group.iloc[0:0]
    if len(combined) < n:
        remaining = work[~work["accession_number"].isin(used_accessions)]
        filler = choose_diverse_rows(remaining, n - len(combined))
        combined = pd.concat([combined, filler], ignore_index=True)
    return combined.drop(columns=["year"], errors="ignore").head(n)


def allocate_even_targets(total: int, labels: list[str]) -> dict[str, int]:
    if total <= 0 or not labels:
        return {label: 0 for label in labels}
    base = total // len(labels)
    remainder = total % len(labels)
    return {
        label: base + (1 if idx < remainder else 0)
        for idx, label in enumerate(labels)
    }


def choose_sector_time_balanced_rows(group: pd.DataFrame, n: int) -> pd.DataFrame:
    if group.empty or n <= 0:
        return group.iloc[0:0]
    sectors = sorted(str(sector) for sector in group["sector"].dropna().unique().tolist())
    if not sectors:
        return choose_time_balanced_rows(group, n)
    selected = []
    used_accessions: set[str] = set()
    sector_targets = allocate_even_targets(n, sectors)
    for sector in sectors:
        sector_df = group[group["sector"] == sector]
        sector_df = sector_df[~sector_df["accession_number"].isin(used_accessions)]
        picked = choose_time_balanced_rows(sector_df, sector_targets[sector])
        selected.append(picked)
        used_accessions.update(picked["accession_number"].tolist())
    combined = pd.concat(selected, ignore_index=True) if selected else group.iloc[0:0]
    if len(combined) < n:
        remaining = group[~group["accession_number"].isin(used_accessions)]
        filler = choose_time_balanced_rows(remaining, n - len(combined))
        combined = pd.concat([combined, filler], ignore_index=True)
    return combined.head(n)


def choose_market_cap_group_rows(
    group: pd.DataFrame, n: int, require_price: bool
) -> pd.DataFrame:
    if group.empty or n <= 0:
        return group.iloc[0:0]
    car_available = require_price and (group["hidden_valence"] != "unknown").any()
    if not car_available:
        return choose_sector_time_balanced_rows(group, n)

    buckets = ["positive", "negative", "mixed"]
    bucket_targets = allocate_even_targets(n, buckets)
    selected = []
    used_accessions: set[str] = set()
    for bucket in buckets:
        bucket_df = group[group["hidden_valence"] == bucket]
        bucket_df = bucket_df[~bucket_df["accession_number"].isin(used_accessions)]
        picked = choose_sector_time_balanced_rows(bucket_df, bucket_targets[bucket])
        selected.append(picked)
        used_accessions.update(picked["accession_number"].tolist())

    combined = pd.concat(selected, ignore_index=True) if selected else group.iloc[0:0]
    if len(combined) < n:
        remaining = group[~group["accession_number"].isin(used_accessions)]
        filler = choose_sector_time_balanced_rows(remaining, n - len(combined))
        combined = pd.concat([combined, filler], ignore_index=True)
    return combined.head(n)


def resolve_selection_targets(args: argparse.Namespace) -> dict[str, int]:
    targets = dict(DEFAULT_SELECTION_TARGETS)
    if args.target_large_cap is not None:
        targets["large_cap"] = args.target_large_cap
    if args.target_small_mid_cap is not None:
        targets["small_mid_cap"] = args.target_small_mid_cap
    return targets


def select_candidates(
    events: pd.DataFrame,
    target_by_market_cap_group: dict[str, int],
    require_price: bool,
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    eligible = events[events.apply(lambda row: eligible_for_candidate(row, require_price), axis=1)]
    eligible = eligible.drop_duplicates(subset=["accession_number"])
    selected = []
    group_order = ["large_cap", "small_mid_cap"]
    seen_groups = {
        str(group)
        for group in eligible["market_cap_group"].dropna().unique().tolist()
    }
    group_order.extend(
        group for group in sorted(seen_groups) if group not in set(group_order)
    )
    for market_cap_group in group_order:
        target = int(target_by_market_cap_group.get(market_cap_group, 0))
        if target <= 0:
            continue
        group_df = eligible[eligible["market_cap_group"] == market_cap_group]
        selected.append(
            choose_market_cap_group_rows(
                group_df, n=target, require_price=require_price
            )
        )
    if not selected:
        return events.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)[SCREENING_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=dt.date.fromisoformat, default=START_DATE)
    parser.add_argument("--end-date", type=dt.date.fromisoformat, default=END_DATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for company/event/candidate CSV outputs.",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=DEFAULT_UNIVERSE_FILE,
        help="CSV with ticker, sector, and market_cap_group columns.",
    )
    parser.add_argument(
        "--target-per-sector",
        type=int,
        default=24,
        help="Legacy option kept for compatibility; market-cap-group targets now drive candidate selection.",
    )
    parser.add_argument("--target-large-cap", type=int, default=None)
    parser.add_argument("--target-small-mid-cap", type=int, default=None)
    parser.add_argument("--min-tokens", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--sec-delay", type=float, default=0.12)
    parser.add_argument("--prices-csv", default="")
    parser.add_argument(
        "--user-agent",
        default=os.environ.get(
            "SEC_USER_AGENT",
            "chiuyiting academic earnings-event screening chiuyiting@example.com",
        ),
    )
    args = parser.parse_args()
    if args.start_date > args.end_date:
        raise ValueError("--start-date must be on or before --end-date")
    return args


def main() -> int:
    args = parse_args()
    universe_by_ticker = load_universe_by_ticker(args.universe_file)
    companies, events = build_screening_rows(args, universe_by_ticker)
    companies_path = args.output_dir / "company_screen.csv"
    events_path = args.output_dir / "screening_events_all.csv"
    candidates_path = args.output_dir / "candidate_events_sec_matching.csv"

    companies.to_csv(companies_path, index=False)
    events.to_csv(events_path, index=False)
    require_price = bool(args.prices_csv)
    candidates = select_candidates(events, resolve_selection_targets(args), require_price)
    candidates.to_csv(candidates_path, index=False)

    print(f"company_screen={companies_path}")
    print(f"screening_events_all={events_path} rows={len(events)}")
    print(f"candidate_events={candidates_path} rows={len(candidates)}")
    if not candidates.empty:
        print("candidate_events_by_market_cap_group=")
        print(candidates.groupby("market_cap_group").size().to_string())
    if not require_price:
        print("price_source=not_supplied; CAR fields left blank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
