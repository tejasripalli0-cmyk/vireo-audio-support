"""Tests for load_data.py and clean_data.py."""
import pandas as pd
import pytest

from src import clean_data, load_data


def make_ticket_row(**overrides):
    base = dict(
        ticket_id="TK-000001", created_at="2025-01-01 10:00", first_response_at="2025-01-01 10:05",
        resolved_at="2025-01-01 12:00", status="resolved", channel="chat", customer_id="C1",
        order_id="VR1", product_sku="VA-EB-PL1", category="Billing & Payments", priority="Normal",
        assigned_team="Billing", agent_id="A1", transfers=0, csat_score=4, refund_amount_inr=None,
        refund_reason_code=None, replacement_issued="N", customer_message="paid twice",
        agent_notes="refunded duplicate", source_system="helpdesk",
    )
    base.update(overrides)
    return base


def to_df(rows):
    df = pd.DataFrame(rows)
    for col in load_data.DATE_COLS_TICKETS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def test_load_tickets_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data.load_tickets(tmp_path / "does_not_exist.csv")


def test_load_tickets_parses_dates(tmp_path):
    df = pd.DataFrame([make_ticket_row()])
    p = tmp_path / "tickets.csv"
    df.to_csv(p, index=False)
    loaded = load_data.load_tickets(p)
    assert pd.api.types.is_datetime64_any_dtype(loaded["created_at"])
    assert loaded.loc[0, "created_at"] == pd.Timestamp("2025-01-01 10:00")


def test_clean_tickets_raises_on_duplicate_ticket_id():
    # Same ticket_id, different content -> a genuine primary-key conflict,
    # not a full-row duplicate that drop_duplicates() would quietly collapse.
    df = to_df([make_ticket_row(), make_ticket_row(customer_message="different message")])
    with pytest.raises(ValueError):
        clean_data.clean_tickets(df)


def test_clean_tickets_flags_bad_date_order_without_dropping():
    rows = [
        make_ticket_row(ticket_id="TK-1"),
        make_ticket_row(ticket_id="TK-2", created_at="2025-01-05 10:00", resolved_at="2025-01-01 10:00"),
    ]
    df = to_df(rows)
    cleaned = clean_data.clean_tickets(df)
    assert len(cleaned) == 2  # nothing dropped
    assert cleaned.set_index("ticket_id").loc["TK-2", "invalid_date_order"] == True
    assert cleaned.set_index("ticket_id").loc["TK-1", "invalid_date_order"] == False


def test_clean_tickets_fills_blank_free_text():
    df = to_df([make_ticket_row(customer_message=None, agent_notes=None)])
    cleaned = clean_data.clean_tickets(df)
    assert cleaned.loc[0, "customer_message"] == ""
    assert cleaned.loc[0, "agent_notes"] == ""


def test_clean_tickets_defaults_replacement_issued_to_n():
    df = to_df([make_ticket_row(replacement_issued=None)])
    cleaned = clean_data.clean_tickets(df)
    assert cleaned.loc[0, "replacement_issued"] == "N"
