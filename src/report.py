"""Generate output files (CSVs, JSON, charts) into outputs/."""
from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def write_categorized_tickets(df: pd.DataFrame) -> None:
    """Write the categorized ticket export. Customer message / agent notes
    are dropped from this export (support-policy §... data-minimization
    concern raised in the assignment: avoid unnecessary exposure of raw
    customer text in generated files). The full text stays only in the
    original data/tickets.csv, which is not shared beyond this project.
    """
    cols = [
        "ticket_id", "created_at", "first_response_at", "resolved_at", "status",
        "channel", "customer_id", "order_id", "product_sku", "category",
        "predicted_category", "categorization_score", "categorization_margin",
        "categorization_model", "taxonomy_version", "category_changed",
        "priority", "assigned_team", "agent_id", "transfers", "csat_score",
        "refund_amount_inr", "refund_reason_code", "replacement_issued",
        "source_system", "invalid_date_order",
    ]
    out = df[cols].copy()
    path = config.OUTPUT_DIR / "categorized_tickets.csv"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(out))


def write_csv(df: pd.DataFrame, filename: str) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))


def write_json(obj: dict, filename: str) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_monthly_category(monthly_cat_df: pd.DataFrame) -> None:
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    piv = monthly_cat_df.pivot(index="month", columns="predicted_category", values="ticket_count").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    piv.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Monthly ticket volume by category (predicted_category)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Tickets")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(config.CHARTS_DIR / "monthly_category.png", dpi=130)
    plt.close(fig)


def chart_monthly_team(monthly_team_df: pd.DataFrame) -> None:
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    piv = monthly_team_df.pivot(index="month", columns="assigned_team", values="ticket_count").fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    piv.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title("Monthly ticket volume by team (assigned_team)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Tickets")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    plt.xticks(rotation=90)
    plt.tight_layout()
    fig.savefig(config.CHARTS_DIR / "monthly_team.png", dpi=130)
    plt.close(fig)


def chart_team_volume_vs_workload(tickets_per_agent_df: pd.DataFrame) -> None:
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    d = tickets_per_agent_df.sort_values("tickets_per_agent", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(d["assigned_team"], d["tickets_per_agent"], color="#4C72B0")
    ax.set_xlabel("Tickets per active agent (all-time)")
    ax.set_title("Workload proxy: tickets per agent, by team")
    plt.tight_layout()
    fig.savefig(config.CHARTS_DIR / "tickets_per_agent.png", dpi=130)
    plt.close(fig)


def chart_billing_reclassification(audit: dict) -> None:
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Confirmed\nBilling & Payments", "Reclassified to\nDelivery & Shipping", "Reclassified to\nother category"]
    values = [
        audit["billing_confirmed_billing"],
        audit["billing_reclassified_to_delivery_specifically"],
        audit["billing_reclassified_to_other_category"] - audit["billing_reclassified_to_delivery_specifically"],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, values, color=["#55A868", "#C44E52", "#DD8452"])
    ax.set_title(f"What are the {audit['billing_tagged_total']:,} bot-tagged\n'Billing & Payments' tickets, really?")
    ax.set_ylabel("Tickets")
    for i, v in enumerate(values):
        ax.text(i, v + 10, str(v), ha="center")
    plt.tight_layout()
    fig.savefig(config.CHARTS_DIR / "billing_reclassification.png", dpi=130)
    plt.close(fig)


def chart_sla_breach_by_channel(sla_result: dict) -> None:
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = sla_result["by_channel"]
    channels = [r["channel"] for r in rows]
    rates = [r["breach_rate"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(channels, rates, color="#4C72B0")
    ax.set_ylabel("First-response breach rate (%)")
    ax.set_title("First-response SLA breach rate by channel\n(helpdesk-era tickets; each breach = Rs 350 credit per policy)")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.3, f"{r:.1f}%", ha="center")
    plt.tight_layout()
    fig.savefig(config.CHARTS_DIR / "sla_breach_by_channel.png", dpi=130)
    plt.close(fig)
