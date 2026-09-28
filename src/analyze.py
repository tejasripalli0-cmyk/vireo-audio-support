"""Monthly, team and business-impact analysis for Vireo Audio support tickets."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 9. Monthly analysis
# ---------------------------------------------------------------------------
def monthly_by_category(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = d["created_at"].dt.to_period("M").astype(str)
    out = (
        d.groupby(["month", "predicted_category"])
        .size()
        .reset_index(name="ticket_count")
        .sort_values(["month", "predicted_category"])
    )
    return out


def monthly_by_team(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = d["created_at"].dt.to_period("M").astype(str)
    out = (
        d.groupby(["month", "assigned_team"])
        .size()
        .reset_index(name="ticket_count")
        .sort_values(["month", "assigned_team"])
    )
    return out


# ---------------------------------------------------------------------------
# 8. Team / headcount analysis
# ---------------------------------------------------------------------------
def team_volume_summary(df: pd.DataFrame) -> pd.DataFrame:
    """The client's requested metric: raw ticket volume by assigned_team,
    using the bot-assigned (as-routed) team -- i.e. exactly what Priya's
    'whichever team has the most volume' rule would look at."""
    g = df.groupby("assigned_team").agg(
        ticket_count=("ticket_id", "count"),
        share_of_total=("ticket_id", lambda s: len(s) / len(df)),
    ).reset_index().sort_values("ticket_count", ascending=False)
    return g


def team_workload_summary(df: pd.DataFrame, agents_df: pd.DataFrame) -> pd.DataFrame:
    """Beyond raw volume: handle time (helpdesk-era rows only, since legacy
    resolved_at is flagged unreliable -- see docs/methodology.md), transfer
    rate, escalation/priority mix, AND active agent headcount, by team.

    FIX: this function used to accept only `df` and return workload metrics
    without headcount, even though its own comment claimed headcount was
    included ("agent headcount per team, from the current roster") -- it
    never actually merged agents_df in. Headcount was available elsewhere
    (tickets_per_agent.csv, via agents_per_team()), so the number itself
    was never wrong, but this file's own claim about its own contents was.
    It now takes agents_df and actually merges active_agents in, so the
    workload and headcount context live together in one place."""
    d = df[(df["source_system"] == "helpdesk") & (~df["invalid_date_order"])].copy()
    d = d.dropna(subset=["resolved_at"])
    d["handle_time_hours"] = (d["resolved_at"] - d["created_at"]).dt.total_seconds() / 3600

    g = d.groupby("assigned_team").agg(
        n_tickets=("ticket_id", "count"),
        median_handle_time_hours=("handle_time_hours", "median"),
        mean_handle_time_hours=("handle_time_hours", "mean"),
        transfer_rate=("transfers", lambda s: (s.fillna(0) > 0).mean()),
        high_priority_share=("priority", lambda s: (s == "High").mean()),
    ).reset_index()

    heads = agents_per_team(agents_df)
    g = g.merge(heads, left_on="assigned_team", right_on="team", how="left").drop(columns=["team"])
    g["active_agents"] = g["active_agents"].fillna(0).astype(int)

    return g.sort_values("median_handle_time_hours", ascending=False)


def agents_per_team(agents_df: pd.DataFrame) -> pd.DataFrame:
    active = agents_df[agents_df["to_date"].isna()]
    return (
        active.groupby("team")
        .agg(active_agents=("agent_id", "nunique"))
        .reset_index()
        .sort_values("active_agents", ascending=False)
    )


def tickets_per_agent(df: pd.DataFrame, agents_df: pd.DataFrame) -> pd.DataFrame:
    """Volume relative to headcount -- a fairer workload proxy than raw
    team volume, since teams differ in size."""
    vol = team_volume_summary(df)[["assigned_team", "ticket_count"]]
    heads = agents_per_team(agents_df).rename(columns={"team": "assigned_team"})
    merged = vol.merge(heads, on="assigned_team", how="left")
    merged["tickets_per_agent"] = merged["ticket_count"] / merged["active_agents"]
    return merged.sort_values("tickets_per_agent", ascending=False)


# ---------------------------------------------------------------------------
# 7. Business impact: first-response SLA breach (support-policy.pdf §3)
# ---------------------------------------------------------------------------
def sla_breach_analysis(df: pd.DataFrame) -> dict:
    """Current-era (helpdesk, post go-live) first-response breach rate and
    the associated automatic Rs 350 credit liability (support-policy.pdf §3).
    Legacy-era tickets are excluded: source_system=legacy_fd predates the
    current policy version and its timestamps are the ones flagged as
    unreliable in clean_data.py.
    """
    d = df[df["source_system"] == "helpdesk"].copy()

    # Defensive: a missing first_response_at would silently become
    # `NaN > target -> False` (i.e. "not breached"), understating the
    # breach rate for exactly the tickets most likely to actually be
    # breaches (no first response recorded at all). Verified against the
    # current data/tickets.csv that no helpdesk-era row is missing
    # first_response_at or created_at (0 of 7,728), so this does not
    # change today's numbers -- it fails loudly instead of silently
    # under-counting if a future data export does have gaps.
    missing_frt = d["first_response_at"].isna().sum()
    missing_created = d["created_at"].isna().sum()
    if missing_frt or missing_created:
        raise ValueError(
            f"sla_breach_analysis: {missing_frt} helpdesk tickets missing "
            f"first_response_at and {missing_created} missing created_at. "
            "These would silently count as 'not breached' if left in -- "
            "decide explicitly (exclude vs. treat as breached) before "
            "reporting this number."
        )

    d["frt_minutes"] = (d["first_response_at"] - d["created_at"]).dt.total_seconds() / 60
    d["target_minutes"] = d["channel"].map(config.FIRST_RESPONSE_TARGET_MIN)
    d["breached"] = d["frt_minutes"] > d["target_minutes"]

    by_channel = d.groupby("channel").agg(
        n_tickets=("ticket_id", "count"),
        breach_rate=("breached", "mean"),
        n_breached=("breached", "sum"),
    ).reset_index()
    by_channel["target_minutes"] = by_channel["channel"].map(config.FIRST_RESPONSE_TARGET_MIN)

    overall_rate = d["breached"].mean()
    n_breached = int(d["breached"].sum())
    n_tickets = len(d)

    # Run-rate: use the most recent 12 full calendar months of helpdesk data
    # to project an annual figure (helpdesk covers 2025-09-14 to 2026-06-30,
    # i.e. ~9.5 months -- not a full year, so we annualize the observed rate
    # rather than sum raw counts).
    date_span_days = (d["created_at"].max() - d["created_at"].min()).days
    tickets_per_year = n_tickets / date_span_days * 365 if date_span_days > 0 else np.nan
    breaches_per_year = tickets_per_year * overall_rate

    current_annual_cost = breaches_per_year * config.SLA_BREACH_CREDIT_INR
    current_quarterly_cost = current_annual_cost / 4

    return {
        "policy_reference": "support-policy.pdf §3: automatic Rs 350 store credit on any missed first-response target",
        "measured_vs_estimated": "breach rate and counts are MEASURED from tickets.csv timestamps; the Rs figures are ESTIMATED policy-based liability (breaches x Rs 350, annualized) -- NOT confirmed cash paid",
        "data_window": "helpdesk-era tickets only (source_system=helpdesk), "
                       f"{d['created_at'].min().date()} to {d['created_at'].max().date()} "
                       f"({date_span_days} days)",
        "overall_breach_rate": round(float(overall_rate), 4),
        "n_tickets_in_window": int(n_tickets),
        "n_breached_in_window": n_breached,
        "by_channel": by_channel.to_dict(orient="records"),
        "annualized_tickets": round(float(tickets_per_year), 0),
        "annualized_breaches": round(float(breaches_per_year), 0),
        "sla_breach_credit_inr": config.SLA_BREACH_CREDIT_INR,
        "current_annual_liability_inr": round(float(current_annual_cost), 0),
        "current_quarterly_liability_inr": round(float(current_quarterly_cost), 0),
    }


def business_impact_scenarios(sla_result: dict) -> dict:
    """Two improvement targets, both derived from the actual measured rate:
    (a) bring the worst channel (chat, by volume-weighted breach count) down
    to the current email-channel rate as a same-team benchmark, and
    (b) a flat 50%-relative-reduction scenario, since Vireo has not stated
    a target of its own. Both are shown; neither is invented in isolation
    from the measured current rate.
    """
    by_channel = {r["channel"]: r for r in sla_result["by_channel"]}
    current_rate = sla_result["overall_breach_rate"]
    annual_tickets = sla_result["annualized_tickets"]

    # Scenario A: halve the current breach rate (relative reduction)
    target_rate_a = round(current_rate / 2, 4)
    breaches_avoided_a = round(annual_tickets * (current_rate - target_rate_a), 0)
    annual_savings_a = breaches_avoided_a * config.SLA_BREACH_CREDIT_INR

    return {
        "current_rate": current_rate,
        "scenario_halve_breach_rate": {
            "label": "ILLUSTRATIVE SCENARIO ONLY -- not a measured outcome, not a prediction, not a client-stated target",
            "current_rate": current_rate,
            "target_rate": target_rate_a,
            "annualized_tickets": annual_tickets,
            "breaches_avoided_per_year": breaches_avoided_a,
            "annual_savings_inr": round(annual_savings_a, 0),
            "quarterly_savings_inr": round(annual_savings_a / 4, 0),
        },
        "assumption": (
            "No Vireo-stated SLA target exists in the supplied materials, so "
            "a halved (relative) breach rate is shown as an illustrative, "
            "achievable-looking target -- NOT a number from the data. The "
            "current rate and current liability above ARE measured directly "
            "from tickets.csv and are the figures we'd stand behind without "
            "qualification."
        ),
    }


# ---------------------------------------------------------------------------
# Category-tag audit: how much does re-categorization move team volume?
# ---------------------------------------------------------------------------
def category_tag_audit(df: pd.DataFrame) -> dict:
    """Quantifies the client's stated concern: does the volume-based
    headcount rule (based on the bot's `category`/`assigned_team` tag) hold
    up once tickets are re-categorized from their actual text?
    """
    changed_rate = (df["predicted_category"] != df["category"]).mean()

    # For the four specialist categories that map 1:1 to a team
    # (Billing, Logistics, Returns Desk, Escalations & Warranty), compute
    # the "true" volume if tickets were routed by predicted_category
    # instead of the bot's original category.
    orig_counts = df["category"].value_counts()
    pred_counts = df["predicted_category"].value_counts()
    delta = pd.DataFrame({
        "bot_tagged_count": orig_counts,
        "reclassified_count": pred_counts,
    }).fillna(0).astype(int)
    delta["delta"] = delta["reclassified_count"] - delta["bot_tagged_count"]
    delta["pct_change"] = (delta["delta"] / delta["bot_tagged_count"].replace(0, np.nan) * 100).round(1)

    # Specifically: how many Billing-tagged tickets were reclassified to a
    # non-Billing category (the client's exact "biggest queue" claim)?
    billing_tagged = df[df["category"] == "Billing & Payments"]
    billing_still_billing = (billing_tagged["predicted_category"] == "Billing & Payments").sum()
    billing_reclassified_away = len(billing_tagged) - billing_still_billing
    billing_to_delivery = (billing_tagged["predicted_category"] == "Delivery & Shipping").sum()

    return {
        "overall_tag_change_rate": round(float(changed_rate), 4),
        "category_delta_table": delta.reset_index().rename(columns={"index": "category"}).to_dict(orient="records"),
        "billing_tagged_total": int(len(billing_tagged)),
        "billing_confirmed_billing": int(billing_still_billing),
        "billing_reclassified_to_other_category": int(billing_reclassified_away),
        "billing_reclassified_to_delivery_specifically": int(billing_to_delivery),
        "billing_reclassified_pct": round(billing_reclassified_away / len(billing_tagged) * 100, 1) if len(billing_tagged) else None,
    }