"""Tests for analyze.py and validate.py -- aggregation and business-calc logic."""
import pandas as pd
import pytest

from src import analyze, config, validate


def _tickets_df():
    rows = [
        dict(ticket_id="TK-1", created_at="2025-10-01 09:00", first_response_at="2025-10-01 09:10",
             resolved_at="2025-10-01 10:00", status="resolved", channel="chat", assigned_team="Billing",
             predicted_category="Billing & Payments", category="Billing & Payments", priority="Normal",
             transfers=0, source_system="helpdesk", invalid_date_order=False),
        dict(ticket_id="TK-2", created_at="2025-10-02 09:00", first_response_at="2025-10-02 09:40",
             resolved_at="2025-10-02 11:00", status="resolved", channel="chat", assigned_team="Billing",
             predicted_category="Delivery & Shipping", category="Billing & Payments", priority="High",
             transfers=1, source_system="helpdesk", invalid_date_order=False),
        dict(ticket_id="TK-3", created_at="2025-11-01 09:00", first_response_at="2025-11-01 20:00",
             resolved_at="2025-11-01 21:00", status="resolved", channel="email", assigned_team="Logistics",
             predicted_category="Delivery & Shipping", category="Delivery & Shipping", priority="Normal",
             transfers=0, source_system="helpdesk", invalid_date_order=False),
    ]
    df = pd.DataFrame(rows)
    for c in ["created_at", "first_response_at", "resolved_at"]:
        df[c] = pd.to_datetime(df[c])
    return df


def _agents_df():
    return pd.DataFrame([
        dict(agent_id="A1", team="Billing", to_date=None),
        dict(agent_id="A2", team="Billing", to_date=None),
        dict(agent_id="A3", team="Logistics", to_date=None),
        dict(agent_id="A4", team="Logistics", to_date="2024-01-01"),  # inactive, excluded
    ])


def test_monthly_by_category_groups_correctly():
    out = analyze.monthly_by_category(_tickets_df())
    row = out[(out.month == "2025-10") & (out.predicted_category == "Billing & Payments")]
    assert row["ticket_count"].iloc[0] == 1


def test_team_volume_summary_counts_by_assigned_team():
    out = analyze.team_volume_summary(_tickets_df())
    billing = out[out.assigned_team == "Billing"]
    assert billing["ticket_count"].iloc[0] == 2
    assert abs(billing["share_of_total"].iloc[0] - 2 / 3) < 1e-9


def test_agents_per_team_excludes_inactive():
    out = analyze.agents_per_team(_agents_df())
    logistics = out[out.team == "Logistics"]
    assert logistics["active_agents"].iloc[0] == 1  # A4 has a to_date, excluded


def test_team_workload_summary_includes_active_agent_headcount():
    # Regression test: team_workload_summary()'s own comment claimed it
    # included agent headcount, but it never merged agents_df in. It now
    # takes agents_df and must return active_agents per team.
    out = analyze.team_workload_summary(_tickets_df(), _agents_df())
    assert "active_agents" in out.columns
    billing = out[out.assigned_team == "Billing"]
    assert billing["active_agents"].iloc[0] == 2
    logistics = out[out.assigned_team == "Logistics"]
    assert logistics["active_agents"].iloc[0] == 1  # A4 inactive, excluded


def test_team_workload_summary_handles_team_with_no_active_agents():
    # A team with ticket volume but zero currently-active agents (e.g. a
    # fully-turned-over team) must not silently disappear or raise -- it
    # should show active_agents == 0, not NaN or a dropped row.
    tickets = _tickets_df()
    agents = pd.DataFrame([
        dict(agent_id="A1", team="Billing", to_date="2024-01-01"),  # inactive
    ])
    out = analyze.team_workload_summary(
        tickets[tickets.assigned_team == "Billing"], agents
    )
    assert out.loc[out.assigned_team == "Billing", "active_agents"].iloc[0] == 0


def test_tickets_per_agent_divides_correctly():
    out = analyze.tickets_per_agent(_tickets_df(), _agents_df())
    billing = out[out.assigned_team == "Billing"]
    assert billing["active_agents"].iloc[0] == 2
    assert billing["tickets_per_agent"].iloc[0] == 1.0  # 2 tickets / 2 agents


def test_sla_breach_analysis_flags_correct_tickets():
    result = analyze.sla_breach_analysis(_tickets_df())
    # TK-1: chat, 10 min response, target 15 -> not breached
    # TK-2: chat, 40 min response, target 15 -> breached
    # TK-3: email, 11h response, target 8h -> breached
    assert result["n_breached_in_window"] == 2
    assert result["n_tickets_in_window"] == 3


def test_sla_breach_liability_uses_policy_credit_amount():
    result = analyze.sla_breach_analysis(_tickets_df())
    assert result["sla_breach_credit_inr"] == config.SLA_BREACH_CREDIT_INR


def test_category_tag_audit_computes_billing_reclassification():
    audit = analyze.category_tag_audit(_tickets_df())
    assert audit["billing_tagged_total"] == 2
    assert audit["billing_confirmed_billing"] == 1
    assert audit["billing_reclassified_to_delivery_specifically"] == 1


def test_business_impact_scenario_math():
    sla = analyze.sla_breach_analysis(_tickets_df())
    scen = analyze.business_impact_scenarios(sla)
    s = scen["scenario_halve_breach_rate"]
    expected_savings = s["breaches_avoided_per_year"] * config.SLA_BREACH_CREDIT_INR
    assert s["annual_savings_inr"] == expected_savings


def _validation_frame(pred_overrides):
    """Build a categorized frame big enough to sample from, where every
    known-flagged ticket exists. pred_overrides: ticket_id -> predicted."""
    ids = list(validate.KNOWN_MISCLASSIFICATIONS)
    rows = []
    for tid in ids:
        true_cat = validate.KNOWN_MISCLASSIFICATIONS[tid][0]
        rows.append(dict(ticket_id=tid, category="Other",
                         predicted_category=pred_overrides.get(tid, "WRONG-" + true_cat)))
    return pd.DataFrame(rows)


def test_validation_counts_error_only_when_prediction_disagrees(monkeypatch):
    """Core regression test: a known-flagged ticket whose CURRENT prediction
    now equals its true label must NOT count as an error (the old code
    counted every id in the dict regardless)."""
    monkeypatch.setattr(config, "VALIDATION_SAMPLE_SIZE", len(validate.KNOWN_MISCLASSIFICATIONS))
    fixed = {"TK-247596": "Charging & Battery", "TK-241791": "Other"}
    rep = validate.run_validation(_validation_frame(fixed))
    n = len(validate.KNOWN_MISCLASSIFICATIONS)
    assert rep["sample_size"] == n
    assert rep["n_errors"] == n - 2
    assert rep["n_correct"] == 2
    assert sorted(rep["known_flagged_tickets_now_passing"]) == ["TK-241791", "TK-247596"]
    assert rep["accuracy"] == pytest.approx(2 / n, abs=1e-4)


def test_validation_all_wrong_and_all_right(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_SAMPLE_SIZE", len(validate.KNOWN_MISCLASSIFICATIONS))
    wrong = validate.run_validation(_validation_frame({}))
    assert wrong["n_errors"] == len(validate.KNOWN_MISCLASSIFICATIONS)
    right = validate.run_validation(_validation_frame(
        {t: v[0] for t, v in validate.KNOWN_MISCLASSIFICATIONS.items()}))
    assert right["n_errors"] == 0 and right["error_rate"] == 0


def test_validation_report_discloses_ai_reviewer_and_limits(monkeypatch):
    monkeypatch.setattr(config, "VALIDATION_SAMPLE_SIZE", len(validate.KNOWN_MISCLASSIFICATIONS))
    rep = validate.run_validation(_validation_frame({}))
    for key in ["sample_size", "random_seed", "taxonomy_version", "reviewer_type",
                "limitation", "failure_modes", "error_rate", "method"]:
        assert key in rep
    assert "not an independent human" in rep["reviewer_type"]
    assert "138" in rep["limitation"]


def test_sla_breach_is_strictly_greater_than_target():
    """Exactly-on-target response (chat, 15 min) is NOT a breach."""
    df = _tickets_df().iloc[:1].copy()
    df["first_response_at"] = df["created_at"] + pd.Timedelta(minutes=15)
    assert analyze.sla_breach_analysis(df)["n_breached_in_window"] == 0


def test_sla_excludes_legacy_and_rejects_missing_timestamps():
    df = _tickets_df()
    legacy = df.iloc[:1].copy()
    legacy["source_system"] = "legacy_fd"
    legacy["ticket_id"] = "TK-L"
    legacy["first_response_at"] = legacy["created_at"] + pd.Timedelta(days=3)
    res = analyze.sla_breach_analysis(pd.concat([df, legacy]))
    assert res["n_tickets_in_window"] == 3  # legacy row not counted
    bad = df.copy()
    bad.loc[0, "first_response_at"] = pd.NaT
    with pytest.raises(ValueError):
        analyze.sla_breach_analysis(bad)


def test_sla_annualization_arithmetic():
    res = analyze.sla_breach_analysis(_tickets_df())
    days = 31  # 2025-10-01 09:00 -> 2025-11-01 09:00
    expected_tickets = 3 / days * 365
    assert res["annualized_tickets"] == pytest.approx(round(expected_tickets), abs=1)
    assert res["current_annual_liability_inr"] == pytest.approx(
        expected_tickets * (2 / 3) * config.SLA_BREACH_CREDIT_INR, abs=1)