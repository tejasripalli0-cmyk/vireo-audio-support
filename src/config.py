"""
Central configuration for the Vireo Audio support-ticket pipeline.

All business constants below are taken directly from support-policy.pdf
(v3.2, effective 1 Apr 2025). Where a figure is an assumption rather than
a value read from a document, it is labelled as such.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"

TICKETS_CSV = DATA_DIR / "tickets.csv"
AGENTS_CSV = DATA_DIR / "agents.csv"
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
ORDERS_CSV = DATA_DIR / "orders.csv"
PRODUCTS_CSV = DATA_DIR / "products.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TAXONOMY_VERSION = "v1.0-2026-09"
VALIDATION_SAMPLE_SIZE = 150

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
# The 11 labels below are the SAME labels the intake bot already uses
# (see tickets.csv `category` column). We do not invent a new taxonomy:
# support-policy.pdf and the data give no evidence that the *label set* is
# wrong, only that individual *assignments* of ticket -> label are
# unreliable (the bot tags from the customer's opening message only, and
# policy states "Agents may correct the tag on closure" but the email
# thread confirms this "rarely" happens in practice). See docs/methodology.md.
CATEGORIES = [
    "Billing & Payments",
    "Delivery & Shipping",
    "Returns & Refunds",
    "Warranty & Repair",
    "Connectivity",
    "Charging & Battery",
    "Audio Quality",
    "App & Firmware",
    "Account & Login",
    "Product Enquiry",
    "Other",
]

# Category -> team is fully deterministic in the source data for the four
# "specialist" categories (verified in EDA: crosstab is diagonal for these).
# Generic categories are handled by whichever Frontline team matches the
# ticket's channel. This mapping is used only for the *rule-based-team*
# sanity check in analyze.py, not to assign teams to tickets (assigned_team
# from tickets.csv is always used as the actual historical routing).
SPECIALIST_CATEGORY_TEAM = {
    "Billing & Payments": "Billing",
    "Delivery & Shipping": "Logistics",
    "Returns & Refunds": "Returns Desk",
    "Warranty & Repair": "Escalations & Warranty",
}
FRONTLINE_TEAM_BY_CHANNEL = {
    "chat": "Chat Frontline",
    "email": "Email Frontline",
    "voice": "Voice Frontline",
    "social": "Chat Frontline",  # policy §2: social worked by Chat Frontline
}

# ---------------------------------------------------------------------------
# Cost standards (support-policy.pdf §4, FY26 planning figures)
# ---------------------------------------------------------------------------
CONTACT_COST_INR = {
    "chat": 210,
    "email": 260,
    "voice": 520,
    "social": 240,
}
BLENDED_CONTACT_COST_INR = 290
TRANSFER_COST_INR = 305
AGENT_HOUR_COST_INR = 165
SHIFT_HOURS = 8

# support-policy.pdf §3: first-response targets and automatic SLA breach credit
FIRST_RESPONSE_TARGET_MIN = {
    "chat": 15,
    "voice": 120,  # 2 hours, callback
    "social": 240,  # 4 hours
    "email": 480,  # 8 hours
}
SLA_BREACH_CREDIT_INR = 350

# support-policy.pdf §5
GOODWILL_CREDIT_CAP_INR = 500
REPLACEMENT_LOGISTICS_COST_INR = 340  # + product unit_cost_inr from products.csv

# support-policy.pdf §10: first-contact resolution / repeat-contact window
REPEAT_CONTACT_WINDOW_DAYS = 30

# support-policy.pdf §9: current helpdesk go-live date (source_system split)
HELPDESK_GO_LIVE = "2025-09-14"
