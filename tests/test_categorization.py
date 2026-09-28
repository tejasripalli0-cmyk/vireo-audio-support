"""Tests for categorize.py -- real behavior, not just 'does it run'."""
import pandas as pd

from src import categorize


def test_delivery_signal_beats_incidental_payment_word():
    """A ticket that mentions payment but whose real problem is a
    non-delivery should be categorized as Delivery, not Billing --
    this is the exact bot-mistagging pattern found in the source data."""
    result = categorize.categorize_text(
        "paid full amount on 30/05 - order still says processing VR892536",
        "cx paid fine, parcel stuck. sending to logistics team.",
    )
    assert result.predicted_category == "Delivery & Shipping"


def test_pure_billing_ticket_stays_billing():
    result = categorize.categorize_text(
        "double payment deducted, please refund the duplicate charge",
        "confirmed UTRs, duplicate refunded to source",
    )
    assert result.predicted_category == "Billing & Payments"


def test_login_ticket_not_confused_with_delivery():
    """Regression test for a real bug found in validation: 'the login code
    never arrives' was matching a delivery-tracking pattern."""
    result = categorize.categorize_text(
        "the login code never arrives, tried 6 times", "OTP resent, cx logged in."
    )
    assert result.predicted_category == "Account & Login"


def test_firmware_update_stuck_not_confused_with_shipment_stuck():
    """Regression test: 'firmware update stuck' should not match the
    'order status stuck' shipment pattern."""
    result = categorize.categorize_text(
        "firmware update stuck for 2 hours", "chk FW rollout. Recovery successful."
    )
    assert result.predicted_category == "App & Firmware"


def test_pickup_ticket_prefers_returns_over_generic_courier_mention():
    result = categorize.categorize_text(
        "return pickup has not happened", "checked pickup awb, pickup rescheduled"
    )
    assert result.predicted_category == "Returns & Refunds"


def test_no_signal_falls_back_to_other():
    result = categorize.categorize_text("hi", "resolved on call")
    assert result.predicted_category == "Other"
    assert result.score == 0


def test_cancellation_before_dispatch_is_other():
    result = categorize.categorize_text(
        "please cancel, ordered by mistake",
        "cancellation request. cancelled before dispatch. refund initiated.",
    )
    assert result.predicted_category == "Other"


def test_categorize_tickets_adds_expected_columns():
    df = pd.DataFrame([
        {"customer_message": "battery drains fast", "agent_notes": "escalated to wty", "category": "Charging & Battery"},
    ])
    out = categorize.categorize_tickets(df)
    for col in ["predicted_category", "categorization_score", "categorization_margin",
                "categorization_model", "taxonomy_version", "category_changed"]:
        assert col in out.columns
    assert out.loc[0, "predicted_category"] == "Charging & Battery"
    assert out.loc[0, "category_changed"] == False


def test_categorize_tickets_flags_change_when_tag_disagrees():
    df = pd.DataFrame([
        {"customer_message": "double payment deducted", "agent_notes": "refunded duplicate", "category": "Delivery & Shipping"},
    ])
    out = categorize.categorize_tickets(df)
    assert out.loc[0, "predicted_category"] == "Billing & Payments"
    assert out.loc[0, "category_changed"] == True


# ---------------------------------------------------------------------------
# Regression tests for the routing-noise / cancellation fixes found during
# this review (see docs/methodology.md §5 and outputs/validation_report.json
# failure modes). Real ticket text, trimmed.
# ---------------------------------------------------------------------------
def test_transfer_to_warranty_team_does_not_override_stated_symptom():
    """TK-247596 pattern: agent_notes names the receiving team
    ('xfer escalations & warranty'), which happens to contain the bare
    keyword 'warranty'. The customer's stated problem (charging) must win,
    not the routing destination."""
    result = categorize.categorize_text(
        "the left one stopped charging since last month - please call me",
        "left earbud not taking charge | advised to observe for 48h | xfer escalations & warranty",
    )
    assert result.predicted_category == "Charging & Battery"


def test_escalated_to_warranty_abbreviation_also_stripped():
    """Same routing-noise pattern, shorter abbreviation ('esc to warranty')
    -- must not spuriously win over a clearly-stated connectivity issue."""
    result = categorize.categorize_text(
        "bluetooth keeps dropping every few minutes",
        "issue: bt dropouts, checked interference | esc to warranty",
    )
    assert result.predicted_category == "Connectivity"


def test_genuine_warranty_claim_status_still_recognized():
    """Sanity check: stripping the routing phrase must not delete genuine
    warranty content elsewhere in the same note (e.g. checking on an
    existing claim, service centre follow-up)."""
    result = categorize.categorize_text(
        "any update on my warranty claim?",
        "cx reported warranty claim status. followed up with service centre. rma shared with cx.",
    )
    assert result.predicted_category == "Warranty & Repair"


def test_cancellation_before_dispatch_regardless_of_word_order():
    """TK-241791 / TK-245700 pattern: the original regex required 'cancel'
    to appear BEFORE 'order'/'shipment' in the text. Real customers phrase
    it the other way round ('...don't ship it...cancel button, greyed
    out...'). The check must be order-independent."""
    result = categorize.categorize_text(
        "ordered the wrong colour, don't ship it. i already tried cancel button, greyed out.",
        "resolved on call",
    )
    assert result.predicted_category == "Other"


def test_cancellation_request_with_order_mentioned_first():
    result = categorize.categorize_text(
        "with reference to my order placed last week. please cancel, ordered by mistake.",
        "cancellation request. cancelled before dispatch.",
    )
    assert result.predicted_category == "Other"