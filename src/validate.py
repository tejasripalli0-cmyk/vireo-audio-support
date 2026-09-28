"""
Validation for the rule-based categorizer.

METHODOLOGY (see docs/methodology.md for full detail):
  1. Draw a fixed random sample of config.VALIDATION_SAMPLE_SIZE tickets
     (seed = config.RANDOM_SEED, so the sample is reproducible).
  2. Each sampled ticket's customer_message + agent_notes was read and
     assigned a "true" category based on the customer's STATED problem
     (not the team that ultimately handled it, and not the bot's
     existing `category` tag, which is exactly what is under review).
  3. predicted_category is compared against that true category.

IMPORTANT, HONEST LIMITATION: the "true" label in step 2 was assigned
by the AI assistant (Claude, via Claude Code) that built this pipeline,
reading raw ticket text -- NOT by an independent human annotator or a
Vireo support agent. This is disclosed here and in docs/ai_usage.md.
A genuinely independent second read (ideally by a Vireo Tier-2 agent)
is listed as a follow-up in docs/methodology.md "Limitations".

SECOND, SEPARATE LIMITATION (fixed here, disclosed for transparency): an
earlier version of this file recorded, for each of the 12 tickets below,
only its ticket_id and true_category, and counted it as an error purely
because its id appeared in this dict -- it never actually compared the
CURRENT predicted_category against true_category. That meant a fix to
categorize.py that correctly resolved one of these 12 cases would still
have been reported as an error forever, and the README's claim that
"a code change that improves or breaks a validated case will show up as
n_errors moving" was not true of the shipped code. run_validation() below
now does that comparison live, every run, against whichever
categorize.py is currently checked in.

The true_category values themselves are still hard-coded (not re-derived
at runtime) -- ground truth should not change just because the code under
test changed -- but whether a given ticket currently counts as an error
is now computed, not assumed.

REMAINING LIMITATION, still real after that fix: true_category is only
recorded for these 12 tickets, i.e. the ones the original AI review found
wrong. The other 138 tickets in the fixed 150-ticket sample were reviewed
once and judged correct, but have no stored true_category, so this
function has no way to notice if a future code change makes one of THOSE
138 wrong -- only regressions/fixes among these 12 documented cases are
caught automatically. Catching a new failure mode outside this set
requires an actual re-review of the sample (or a larger one), not just
rerunning this file. See docs/methodology.md §10.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# ticket_id -> (true_category, failure_mode)
KNOWN_MISCLASSIFICATIONS: dict[str, tuple[str, str]] = {
    "TK-247411": ("Warranty & Repair", "no explicit warranty/damage keyword (broken part described only as 'strap pin came off')"),
    "TK-247540": ("Connectivity", "connectivity issue phrased without a pairing/bluetooth keyword ('keeps losing my phone')"),
    "TK-240882": ("Connectivity", "ticket mentions both a connectivity symptom and a DOA resolution; classifier keyed on the resolution ('doa'), not the stated symptom"),
    "TK-252568": ("Connectivity", "same pattern: resolution phrase ('reverse pkp') outweighed the stated pairing problem"),
    "TK-241791": ("Other", "cancellation-before-dispatch request; classifier keyed on an incidental logistics word"),
    "TK-251157": ("Billing & Payments", "promotional-discount dispute; taxonomy/keyword set has no dedicated phrase for coupon/discount issues"),
    "TK-247596": ("Charging & Battery", "agent note mentions the receiving team ('xfer escalations & warranty'); classifier keyed on that instead of the stated charging symptom"),
    "TK-245969": ("App & Firmware", "app-hang phrased as 'spinning circle' / 'Update hang' without the 'firmware' or 'update fail' keywords the rule set expects"),
    "TK-245700": ("Other", "cancellation-before-dispatch request; agent note's incidental 'RTO' mention outweighed the cancellation intent"),
    "TK-242051": ("Warranty & Repair", "transit-damage phrasing used 'a crack' (noun) rather than 'cracked' (the form the rule set matches)"),
    "TK-244751": ("Billing & Payments", "promotional-discount dispute, same taxonomy gap as TK-251157"),
    "TK-241805": ("Billing & Payments", "promotional-discount dispute, same taxonomy gap as TK-251157"),
}

# Coarse failure-mode bucket for each known-flagged ticket, used to build
# failure_modes in the report below. This groups the (still hard-coded)
# true_category/failure_mode text above; it does NOT determine whether a
# ticket is currently an error -- that is decided live in run_validation()
# by comparing predicted_category to true_category, so a ticket whose
# underlying bug has since been fixed drops out of the summary on its own.
_FAILURE_MODE_BUCKET: dict[str, str] = {
    "TK-247411": "vocabulary gap (real category, unmatched phrasing)",
    "TK-247540": "vocabulary gap (real category, unmatched phrasing)",
    "TK-240882": "resolution/handling-note keyword outweighs stated symptom",
    "TK-252568": "resolution/handling-note keyword outweighs stated symptom",
    "TK-241791": "cancellation-before-dispatch pulled toward another category",
    "TK-251157": "promotional discount/coupon dispute: no dedicated taxonomy category",
    "TK-247596": "stated symptom vs. handling-team keyword conflict",
    "TK-245969": "vocabulary gap (real category, unmatched phrasing)",
    "TK-245700": "cancellation-before-dispatch pulled toward another category",
    "TK-242051": "vocabulary gap (real category, unmatched phrasing)",
    "TK-244751": "promotional discount/coupon dispute: no dedicated taxonomy category",
    "TK-241805": "promotional discount/coupon dispute: no dedicated taxonomy category",
}


def run_validation(categorized_df: pd.DataFrame) -> dict:
    """Reproduce the fixed validation sample and score the CURRENT
    predicted_category against the AI-reviewed true_category labels above.

    Only the 12 tickets in KNOWN_MISCLASSIFICATIONS have a stored true
    label. A ticket counts as an error iff it is one of those 12 AND its
    live predicted_category still disagrees with the stored true_category
    -- not merely because its id appears in the dict (see the module
    docstring's "SECOND, SEPARATE LIMITATION" for why that distinction
    matters and what it does/doesn't catch)."""
    sample = categorized_df.sample(
        n=config.VALIDATION_SAMPLE_SIZE, random_state=config.RANDOM_SEED
    )

    n = len(sample)
    errors = []
    fixed_since_review = []  # known-flagged tickets the current code now gets right
    for _, row in sample.iterrows():
        tid = row["ticket_id"]
        if tid in KNOWN_MISCLASSIFICATIONS:
            true_cat, mode = KNOWN_MISCLASSIFICATIONS[tid]
            if row["predicted_category"] != true_cat:
                errors.append({
                    "ticket_id": tid,
                    "predicted_category": row["predicted_category"],
                    "true_category": true_cat,
                    "bot_original_category": row["category"],
                    "failure_mode": mode,
                })
            else:
                fixed_since_review.append(tid)

    n_errors = len(errors)
    accuracy = (n - n_errors) / n

    failure_modes: dict[str, int] = {}
    for e in errors:
        bucket = _FAILURE_MODE_BUCKET.get(e["ticket_id"], "unclassified")
        failure_modes[bucket] = failure_modes.get(bucket, 0) + 1

    # Per-category performance: for categories the classifier predicted at
    # least once in the sample, how many of those predictions were in the
    # error list.
    pred_counts = sample["predicted_category"].value_counts().to_dict()
    error_by_pred_cat: dict[str, int] = {}
    for e in errors:
        error_by_pred_cat[e["predicted_category"]] = error_by_pred_cat.get(e["predicted_category"], 0) + 1
    per_category = {
        cat: {
            "n_predicted_in_sample": count,
            "n_errors": error_by_pred_cat.get(cat, 0),
            "sample_precision": round(1 - error_by_pred_cat.get(cat, 0) / count, 4),
        }
        for cat, count in pred_counts.items()
    }

    report = {
        "method": (
            "Fixed random sample (n=150, seed=42). The 12 tickets in "
            "KNOWN_MISCLASSIFICATIONS were AI-reviewed against raw ticket "
            "text (not an independent human review -- see 'reviewer_type' "
            "and 'limitation'); their true_category is fixed, but whether "
            "each currently counts as an error is recomputed every run "
            "against the live predicted_category. The other 138 sampled "
            "tickets were reviewed once and judged correct at that time; "
            "they are assumed correct here and are NOT re-reviewed."
        ),
        "sample_size": n,
        "random_seed": config.RANDOM_SEED,
        "taxonomy_version": config.TAXONOMY_VERSION,
        "model": "rule_based_keyword_v1",
        "reviewer_type": "AI-assisted first-pass (Claude, via Claude Code) -- not an independent human annotator or Vireo support agent",
        "accuracy": round(accuracy, 4),
        "n_correct": n - n_errors,
        "n_errors": n_errors,
        "error_rate": round(1 - accuracy, 4),
        "known_flagged_tickets_now_passing": sorted(fixed_since_review),
        "per_category_sample_precision": per_category,
        "failure_modes": failure_modes,
        "errors_detail": errors,
        "limitation": (
            "This is an AI-reviewed first-pass agreement figure, not an "
            "independently audited accuracy figure: the true_category for "
            "each of the 12 originally-flagged tickets was assigned by the "
            "same AI assistant (Claude, via Claude Code) that built this "
            "pipeline, reading the raw ticket text -- not by an independent "
            "human annotator or a Vireo support agent. Separately, "
            "true_category is only stored for those 12 tickets; the other "
            "138 in the sample are assumed correct from that same original "
            "review and are not re-checked here, so this function cannot by "
            "itself catch a NEW misclassification introduced among them by "
            "a future code change. A second, independent human read "
            "(ideally by a Tier-2 agent who knows the products), ideally of "
            "the full 150 rather than just the 12 flagged ones, is "
            "recommended before this number is used in a business decision. "
            "See docs/ai_usage.md and docs/methodology.md."
        ),
    }
    logger.info("Validation: %d/%d correct (%.1f%% accuracy)", n - n_errors, n, accuracy * 100)
    return report


def save_validation_report(report: dict, path=config.OUTPUT_DIR / "validation_report.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report written to %s", path)