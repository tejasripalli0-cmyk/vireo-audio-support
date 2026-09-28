"""Cleaning steps for tickets.csv.

Kept deliberately minimal: the EDA in docs/methodology.md found no duplicate
ticket_id values, no invalid channel/status/team values, and no rows where
resolved_at precedes created_at. The main real issues are (a) the reliability
of the `category` tag itself, which is handled in categorize.py, not here,
and (b) legacy-migrated timestamps/costs, which are flagged rather than
silently altered.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    # Drop exact full-row duplicates, if any (defensive; none found in EDA).
    df = df.drop_duplicates()
    # Duplicate ticket_id would indicate a real data problem -> fail loudly
    # rather than silently keep one copy, since ticket_id is the primary key.
    dup_ids = df["ticket_id"][df["ticket_id"].duplicated()].tolist()
    if dup_ids:
        raise ValueError(f"Duplicate ticket_id values found: {dup_ids[:10]} ...")

    # Rows where resolved_at is before created_at are a data-quality error.
    bad_dates = df["resolved_at"].notna() & (df["resolved_at"] < df["created_at"])
    if bad_dates.any():
        logger.warning(
            "%d tickets have resolved_at before created_at; flagging, not dropping.",
            bad_dates.sum(),
        )
    df["invalid_date_order"] = bad_dates

    # Normalise free-text fields: keep original text (required for
    # categorization) but strip surrounding whitespace only.
    for col in ["customer_message", "agent_notes"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Numeric/flag columns: make blanks explicit rather than NaN-ambiguous.
    df["transfers"] = pd.to_numeric(df["transfers"], errors="coerce")
    df["csat_score"] = pd.to_numeric(df["csat_score"], errors="coerce")
    df["refund_amount_inr"] = pd.to_numeric(df["refund_amount_inr"], errors="coerce")
    df["replacement_issued"] = df["replacement_issued"].fillna("N")

    # Known category label set: anything outside it is a data problem, not
    # a new category we should silently accept.
    unknown_categories = set(df["category"].dropna().unique()) - set(config.CATEGORIES)
    if unknown_categories:
        logger.warning("Unexpected category labels in source data: %s", unknown_categories)

    logger.info("clean_tickets: %d -> %d rows (dedup), %d flagged bad date order",
                before, len(df), bad_dates.sum())
    return df
