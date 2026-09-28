"""
Ticket re-categorization.

WHY RULE-BASED, NOT AN LLM OR TRAINED ML MODEL (see docs/methodology.md
section "Model/AI strategy" for the full reasoning):

  - No ground-truth labels exist anywhere in the supplied data, so a
    trained classifier (TF-IDF+LogReg, embeddings+kNN, etc.) would need
    labels manufactured by either an LLM or a human first. At ~11.8k
    tickets, an LLM-labelling pass would cost real money and be the
    single most expensive, least reproducible part of the whole pipeline
    for a problem that a transparent rule set solves adequately.
  - The bot's OWN existing tags were produced by a simple intake-keyword
    flow (support-policy.pdf §2), so the ticket text is short, templated,
    and keyword-dense (see the raw samples in docs/methodology.md) --
    exactly the regime where keyword rules are competitive with ML.
  - Rules are fully deterministic, free, auditable line-by-line, and can
    be corrected by a support-ops person with no ML background -- a
    material maintainability advantage for a 44-agent team.
  - The validation step (validate.py) exists specifically to catch cases
    where this assumption fails; see outputs/validation_report.json for
    the measured error rate and failure modes.

This module assigns each ticket a `predicted_category`, independent of
(and used to audit) the bot-assigned `category` tag already in the data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src import config

MODEL_NAME = "rule_based_keyword_v1"

# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------
# Each category maps to a list of (regex, weight) pairs. Patterns were
# derived by reading stratified samples of customer_message/agent_notes
# text per existing bot tag (see docs/methodology.md, section "EDA").
# Weight 2 = near-unambiguous signal for the category; weight 1 = supporting
# signal that often co-occurs with other categories.
_R = re.compile

RULES: dict[str, list[tuple[re.Pattern, int]]] = {
    "Delivery & Shipping": [
        (_R(r"(order|shipment|package|parcel|item)[^.\n]{0,25}(not deliver|not received|never (arriv|showed|came))", re.I), 2),
        (_R(r"(not deliver|not received)[^.\n]{0,25}(order|shipment|package|parcel|item)", re.I), 2),
        (_R(r"\bshipment|\bcourier|\btracking|\bawb|\brto\b|reshi?p|re-?ship", re.I), 2),
        (_R(r"\bstuck (on|in) (shipped|processing|transit)|order status stuck|marked delivered", re.I), 2),
        (_R(r"wrong item|wrong (color|colour)|address (update|change|wrong)", re.I), 1),
        (_R(r"no update on (my )?(order|shipment|delivery)", re.I), 1),
        (_R(r"lost in transit|delivery delay(ed)?|dlvry delay(ed)?", re.I), 2),
        (_R(r"parcel stuck|(sending|transferred|xfer|moved) to logistics", re.I), 2),
    ],
    "Returns & Refunds": [
        (_R(r"reverse pick ?up|pkp\b", re.I), 2),
        (_R(r"\bpickup\b.{0,25}(pending|resched|not (done|happened)|missed)|pickup (pending|scheduled|not)", re.I), 2),
        (_R(r"\breturn(ed|ing)? (accepted|received|process|status)|return\b.*(refund|amount)", re.I), 2),
        (_R(r"refund (not|pending|status|nowhere|initiat|process|reprocess|credited)", re.I), 2),
        (_R(r"\bwaiting for (my )?refund\b", re.I), 2),
        (_R(r"refund (was )?promised|money (hasn'?t|has not) come back|where is the money", re.I), 2),
    ],
    "Warranty & Repair": [
        (_R(r"\bwarranty\b", re.I), 2),
        (_R(r"\brepair\b|service cent(re|er)|\brma\b|claim number", re.I), 2),
        (_R(r"no (audio|sound) at all|left side (has )?no|one side (dead|silent)", re.I), 1),
        (_R(r"damaged? in transit|damage claim|\bdent\b|\bcracked\b|\bcrushed\b|dead on arrival|\bdoa\b", re.I), 3),
    ],
    "Connectivity": [
        (_R(r"\bpair(ing)?\b|bluetooth|not connect(ing)?|disconnect|discoverable", re.I), 2),
        (_R(r"connection (drop|issue|fail)|randoom? disconnect", re.I), 2),
    ],
    "Charging & Battery": [
        (_R(r"\bbattery\b|\bbackup\b.*(poor|terrible|bad|drain|almost nothing)", re.I), 2),
        (_R(r"\bcharg(e|ing)\b|battery (drain|backup|life)|dies by|rapid.*drain", re.I), 2),
        (_R(r"not charging|won'?t charge|charge case", re.I), 2),
    ],
    "Audio Quality": [
        (_R(r"\bmic\b|underwater|muffled|static|\bbuzz(ing)?\b|distort", re.I), 2),
        (_R(r"\baudio (quality|distortion)|sound (quality|issue)|no sound from", re.I), 2),
        (_R(r"\bcrackl|\bhiss(ing)?\b|frying sound|cannot hear me|can'?t hear me", re.I), 2),
    ],
    "App & Firmware": [
        (_R(r"\bapp\b.*(crash|close|open|won'?t)|firmware|\bfw\b\s*(update|version)", re.I), 2),
        (_R(r"update fail|bricked|progress bar|app closes|recovery mode", re.I), 2),
        (_R(r"\bbug\b", re.I), 1),
    ],
    "Account & Login": [
        (_R(r"\botp\b|log ?in|password|account (locked|unlock)|login code", re.I), 2),
    ],
    "Product Enquiry": [
        (_R(r"before i buy|does (this|it|orbit|nexa|pulse) work|compat(ible|ibility)", re.I), 2),
        (_R(r"\bwill (this|it) (talk|work|run)|pre-?sales|spec sheet", re.I), 2),
        (_R(r"\bcan i connect (two|2)|does .*support", re.I), 1),
    ],
    "Billing & Payments": [
        (_R(r"double payment|duplicate (payment|charge)|charged twice", re.I), 2),
        (_R(r"\binvoice\b|\bgst\b|payment (deduct|fail|issue)|paid.*nothing", re.I), 2),
        (_R(r"\bpaid\b|\bpayment\b|\bcharged\b", re.I), 1),
    ],
}

# Priority order used ONLY to break ties when two+ categories score equally.
# Ordered from most-specific/actionable to most-generic, so an ambiguous
# "paid but nothing delivered" ticket (Billing signal=1, Delivery signal=2)
# is decided by score first; priority only matters on an exact tie.
PRIORITY_ORDER = [
    "Returns & Refunds",
    "Delivery & Shipping",
    "Warranty & Repair",
    "Connectivity",
    "Charging & Battery",
    "Audio Quality",
    "App & Firmware",
    "Account & Login",
    "Product Enquiry",
    "Billing & Payments",
    "Other",
]

# Cancellation / pre-dispatch requests are operational "Other" cases (they
# are not about a product issue).
#
# FIX: the original single regex required "cancel" to appear BEFORE
# "shipment"/"before dispatch"/"order" in the string, i.e. it matched only
# one word order. Real tickets say "don't ship it ... tried cancel button".
# Now an unordered AND: mentions cancelling AND mentions order/shipment.
# This is a general regex-ordering fix, not tuned to any one ticket.
_CANCEL_WORD = _R(r"\bcancel(led|ling|lation)?\b", re.I)
_CANCEL_CONTEXT = _R(r"\border\b|\bship(ment|ping)?\b|\bdispatch(ed)?\b", re.I)
_CHANGE_OF_MIND = _R(r"change of mind", re.I)


def _is_cancellation_before_dispatch(text: str) -> bool:
    return bool(_CHANGE_OF_MIND.search(text)) or bool(
        _CANCEL_WORD.search(text) and _CANCEL_CONTEXT.search(text)
    )


# FIX: "Escalations & Warranty" is the literal name of the team agents
# route tickets to, and bare "warranty" is a Warranty & Repair keyword, so
# every ticket merely TRANSFERRED to that team scored as Warranty & Repair
# whatever the customer's stated problem was (~225 of 11,780 tickets'
# agent_notes contain such a routing phrase). We strip only the routing
# phrase "<transfer verb> (to) (the) (escalations &) warranty (team)" from
# agent_notes before scoring. The rest of agent_notes still counts (e.g.
# "parcel stuck, sending to logistics" is real signal), and genuine
# warranty content such as "warranty claim status" is left alone.
_ROUTING_NOISE_STRIP = _R(
    r"\b(transferred?|xfer|xefr|escalated?|esc|via)\.?\s*(to\s+)?(the\s+)?"
    r"(escalations?\s*(&|and)\s*warranty|warranty(\s+team)?)\b",
    re.I,
)


def _strip_routing_noise(agent_notes: str) -> str:
    return _ROUTING_NOISE_STRIP.sub(" ", agent_notes)


@dataclass
class CategoryResult:
    predicted_category: str
    score: int
    margin: int
    matched_categories: str


def categorize_text(customer_message: str, agent_notes: str) -> CategoryResult:
    # customer_message and agent_notes are scored together (see
    # docs/methodology.md §4): agent_notes often holds the clearest
    # statement of the real problem, so it is not discounted wholesale;
    # only routing boilerplate is removed.
    text = f"{customer_message}\n{_strip_routing_noise(agent_notes)}"

    if _is_cancellation_before_dispatch(text):
        return CategoryResult("Other", 2, 2, "Other")

    scores: dict[str, int] = {}
    for cat, patterns in RULES.items():
        s = 0
        for pattern, weight in patterns:
            if pattern.search(text):
                s += weight
        if s > 0:
            scores[cat] = s

    if not scores:
        return CategoryResult("Other", 0, 0, "")

    max_score = max(scores.values())
    top = [c for c, s in scores.items() if s == max_score]
    if len(top) > 1:
        top.sort(key=lambda c: PRIORITY_ORDER.index(c))
    winner = top[0]

    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0)
    matched = ",".join(sorted(scores.keys()))
    return CategoryResult(winner, max_score, margin, matched)


def categorize_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Add predicted_category / categorization_score / categorization_margin /
    categorization_model / category_changed columns to a tickets dataframe."""
    df = df.copy()
    results = df.apply(
        lambda r: categorize_text(r["customer_message"], r["agent_notes"]), axis=1
    )
    df["predicted_category"] = [r.predicted_category for r in results]
    df["categorization_score"] = [r.score for r in results]
    df["categorization_margin"] = [r.margin for r in results]
    df["categorization_model"] = MODEL_NAME
    df["taxonomy_version"] = config.TAXONOMY_VERSION
    df["category_changed"] = df["predicted_category"] != df["category"]
    return df