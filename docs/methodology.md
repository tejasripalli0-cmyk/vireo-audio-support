# Methodology

This document explains how the analysis and outputs in `outputs/` were produced from the supplied Vireo Audio data and policy documents. It is intended to make the analysis reproducible, auditable, and easy to challenge.

---

## 1. Data Inventory and Scope

| File | Rows | Used? | Purpose |
|---|---:|---|---|
| `tickets.csv` | 11,780 | Yes | Primary dataset for categorization, monthly trends, team analysis, and SLA analysis |
| `agents.csv` | 44 | Yes | Current active team headcount for the tickets-per-agent workload proxy |
| `support-policy.pdf` | — | Yes | Source for SLA targets, the ₹350 SLA-credit rule, and policy definitions |
| `email-thread.txt` | — | Yes, for context | Defines the client's original business question and the proposed hiring rule |
| `README.txt` | — | Yes, for context | Assignment and data context |
| `customers.csv` | 9,501 | Reviewed, not used | Not required for the final business questions |
| `orders.csv` | 15,000 | Reviewed, not used | Not required for the final business questions |
| `products.csv` | 15 | Reviewed, not used | Not required for the final business questions |

### Why customers, orders, and products were not used

The ticket data already contains `customer_id`, `order_id`, and `product_sku`, but the requested analysis does not require customer demographics, order value, or product cost.

These datasets remain useful for possible follow-up work. For example:

- `products.csv.unit_cost_inr` could support a replacement-cost analysis.
- `orders.csv` could support order-value-weighted analysis.
- `customers.csv` could support customer-segment or geography analysis.

They were intentionally left out of the final analysis to keep the scope focused on the assignment's stated questions.

---

## 2. Data Coverage and Initial Findings

The ticket dataset contains **11,780 tickets** covering:

**20 June 2024 – 30 June 2026**

The helpdesk system became operational on **14 September 2025**. Earlier records are identified as legacy data from the previous system.

### Basic data checks

The following checks were performed:

- Duplicate `ticket_id` checks
- Required-field and null checks
- Date-range checks
- Timestamp consistency checks
- Category and team-value checks
- Agent-roster consistency checks
- Distribution checks across categories and teams

There were no duplicate `ticket_id` values and no missing required ticket fields.

`order_id` is blank on 4,017 tickets. This is expected because customers do not always provide an order ID.

---

## 3. Legacy Timestamp Issue

A total of **2,379 tickets** have:

`resolved_at < created_at`

These records are all from the legacy system.

The supplied policy explains that legacy resolution timestamps were reconstructed from a legacy event log using UTC, while the helpdesk displays and exports timestamps in IST.

Because the exact historical correction procedure is not specified, the analysis does not attempt to manually reconstruct these timestamps.

Instead:

- The tickets remain in the dataset for ticket-count and category analysis.
- They are flagged using `invalid_date_order`.
- Time-based calculations exclude these records.
- SLA and handle-time analysis is therefore restricted to reliable helpdesk-era records.

This avoids creating artificial duration values from an uncertain timestamp correction.

---

## 4. Data Cleaning

The cleaning process is implemented in `src/clean_data.py`.

### Cleaning steps

1. Remove exact duplicate rows.
2. Stop the pipeline if duplicate `ticket_id` values contain conflicting ticket information.
3. Flag rows where `resolved_at < created_at`.
4. Preserve flagged rows for non-time-based analysis.
5. Normalize free-text fields by trimming whitespace.
6. Coerce numeric columns to numeric types.
7. Make blank `replacement_issued` values explicit as `"N"`.

No rows are silently removed because of the legacy timestamp issue.

---

# 5. Categorization Method

## 5.1 Approach

The final categorization method is a **deterministic rule-based keyword scoring system**.

It does not use:

- an LLM classification API,
- a trained machine-learning classifier,
- embeddings,
- or paid API calls.

The categorizer is implemented in:

`src/categorize.py`

The existing category taxonomy supplied by the ticket system is reused rather than creating a new taxonomy.

The categories are:

1. Delivery & Shipping
2. Returns & Refunds
3. Warranty & Repair
4. Connectivity
5. Charging & Battery
6. Audio Quality
7. App & Firmware
8. Account & Login
9. Product Enquiry
10. Billing & Payments
11. Other

---

## 5.2 Why a Rule-Based Approach Was Used

Three considerations drove the choice.

### 1. There is no complete ground-truth label set

The supplied data contains existing bot categories, but those labels are precisely what this analysis is checking.

Using those labels directly as training truth would risk reproducing the existing categorization errors.

### 2. Ticket text is relatively short and domain-specific

Many tickets contain recurring terminology such as:

- pairing
- battery backup
- reverse pickup
- shipment
- order
- refund
- payment
- firmware
- login
- warranty

This makes transparent keyword-based rules practical for a first-pass categorization system.

### 3. The method is easy to audit

Each rule can be inspected and changed directly.

This is useful for a support-operations workflow where a team lead should be able to understand why a ticket received a particular category.

The approach also has zero runtime API cost.

---

## 5.3 Scoring

Each category contains a set of regular-expression patterns.

Patterns receive different weights depending on how strongly they indicate a category.

For example:

- A specific phrase such as `double payment` receives a stronger Billing signal.
- A generic word such as `paid` receives a weaker signal because it can also occur in delivery-related complaints.

The system evaluates both:

- `customer_message`
- `agent_notes`

The category with the highest score is selected.

A fixed priority order is used to resolve ties.

Each prediction also records:

- `categorization_score`
- `categorization_margin`

in `outputs/categorized_tickets.csv`.

The margin can be used later to identify lower-confidence predictions for manual review.

---

## 5.4 Why Agent Notes Are Included

The customer's opening message does not always contain the complete issue.

Agent closing notes can contain information such as:

> parcel stuck. sending to logistics team.

That information can be useful when the opening message only mentions payment or another secondary detail.

Therefore, both `customer_message` and `agent_notes` are considered.

However, routing boilerplate can itself create false signals.

For example, the phrase:

`Escalations & Warranty`

contains the word `Warranty`.

If the entire routing phrase were scored literally, tickets transferred to that team could incorrectly become `Warranty & Repair`.

The categorizer therefore removes the specific routing phrase before scoring while retaining genuine warranty-related text.

This change is covered by regression tests.

---

## 5.5 Cancellation Handling

Cancellation requests do not map cleanly to a product-problem category.

The categorizer therefore detects cancellation language together with order, shipment, or dispatch language and routes these cases to `Other`.

The matching works regardless of the order in which the cancellation and order-related terms appear.

---

## 5.6 Multi-Issue Tickets

A ticket can mention multiple issues.

The categorizer does not simply use the first keyword found.

Instead, all relevant category signals are scored and the strongest category wins.

For example, a ticket containing both payment and delivery language can be classified based on the relative strength of the two signals.

Tickets for which no category receives a meaningful signal fall back to `Other`.

---

# 6. Categorization Validation

Validation is implemented in:

`src/validate.py`

The validation result currently reported by the pipeline is:

**94.0% agreement — 141/150 tickets**

This corresponds to a **6.0% error rate** on the reviewed sample.

### Important qualification

This is **not an independent human accuracy audit**.

The sample was reviewed using an AI-assisted first-pass review process, so the result should be treated as directional rather than as a production-grade accuracy guarantee.

---

## 6.1 Sampling

The validation sample contains:

- 150 tickets
- random seed: `42`
- sampled from the 11,780-ticket dataset
- taxonomy version: `v1.0-2026-09`

The sample was drawn uniformly rather than being stratified by category.

Therefore, small categories have only a few observations in the validation sample, and their individual precision estimates should be treated as directional.

---

## 6.2 Validation Labels

The validation process stores confirmed labels for the tickets identified during review as potential errors.

The current validation logic compares the live prediction against those stored labels.

This means the validation system can detect whether the previously identified errors remain fixed.

However, the other sampled tickets do not have independently stored labels.

Therefore, the 94.0% result should not be interpreted as a fully independent estimate of population accuracy.

A fresh independently labelled sample would be required for that.

---

## 6.3 Observed Failure Modes

The remaining validation errors include:

- vocabulary gaps such as `"strap pin came off"`
- phrases such as `"spinning circle"`
- damage descriptions such as `"a crack"`
- promotional discount/coupon disputes
- resolution-note terminology such as `"doa"` or `"reverse pkp"` outweighing the customer's stated symptom

Some earlier rule errors were also identified and converted into regression tests, including:

- `"never arrives"` incorrectly matching login OTP problems
- `"stuck"` incorrectly matching firmware-update problems
- payment language incorrectly overriding delivery-delay context

The remaining errors were deliberately not patched individually because doing so against a small validation sample would risk overfitting the rules to those examples.

---

# 7. Monthly and Team Analysis

The analysis is implemented in `src/analyze.py`.

## 7.1 Monthly Category Analysis

`outputs/monthly_category.csv`

contains ticket counts grouped by:

- calendar month
- predicted category

This is based on the categorization output rather than the original bot category.

---

## 7.2 Monthly Team Analysis

`outputs/monthly_team.csv`

contains ticket counts grouped by:

- calendar month
- historical assigned team

The historical `assigned_team` is retained rather than silently replacing it with a predicted routing decision.

This distinction is important because the project is analyzing historical routing rather than actually re-routing historical tickets.

---

## 7.3 Team Volume

`outputs/team_volume_summary.csv`

contains:

- team
- ticket count
- share of total tickets

This directly answers the client's original volume-based question.

The largest team by historical ticket volume is:

**Chat Frontline — 3,030 tickets (25.7%)**

Billing has:

**2,564 tickets (21.8%)**

Therefore, the supplied data does not support the statement that Billing is the largest team by raw ticket volume.

---

## 7.4 Tickets per Agent

`outputs/tickets_per_agent.csv`

calculates ticket volume divided by active agent headcount.

This is included because raw ticket volume alone does not account for differences in team size.

The resulting workload proxy is:

| Team | Tickets | Active Agents | Tickets / Agent |
|---|---:|---:|---:|
| Billing | 2,564 | 4 | 641 |
| Logistics | 1,905 | 5 | 381 |
| Returns Desk | 1,049 | 3 | 350 |
| Email Frontline | 1,807 | 7 | 258 |
| Voice Frontline | 900 | 4 | 225 |
| Chat Frontline | 3,030 | 15 | 202 |
| Escalations & Warranty | 525 | 6 | 88 |

This is only a workload proxy.

The ticket history spans June 2024–June 2026, while the roster represents the current active headcount. Therefore, this is not a time-matched historical utilization measure.

---

## 7.5 Handle Time and Team Workload

`outputs/team_workload_summary.csv` contains:

- median handle time
- mean handle time
- transfer rate
- high-priority share
- active headcount

Time-based fields are restricted to helpdesk-era tickets with valid timestamps.

Handle time is not treated as a direct headcount requirement because the data does not contain sufficient time-on-ticket information to convert handle time into staffing capacity.

---

# 8. Billing Tag Audit

The original bot tagged:

**2,564 tickets as Billing & Payments**

The rule-based re-categorization produces:

| Result | Tickets | Share |
|---|---:|---:|
| Remain Billing & Payments | 898 | 35.0% |
| Reclassified to Delivery & Shipping | 1,049 | 40.9% |
| Reclassified to other categories | 617 | 24.1% |
| Reclassified in total | 1,666 | 65.0% |

These figures are model outputs from the rule-based categorizer.

They should therefore be described as **reclassification findings**, not as verified ground truth.

In particular, the analysis should not claim that only 35% of Billing tickets are "actually" Billing without an independent review.

The finding is useful because it shows that the original Billing tag may contain substantial cross-category work and should be checked before being used as the sole basis for staffing decisions.

---

# 9. Business Impact: First-Response SLA

The selected financial metric is first-response SLA-breach exposure.

The reason for selecting this metric is that the supplied support policy provides:

- explicit first-response targets by channel
- an explicit ₹350 store-credit rule for missed first-response targets

This allows a measurable financial calculation directly from the ticket timestamps.

---

## 9.1 Scope

The SLA analysis uses only helpdesk-era tickets:

**14 September 2025 – 30 June 2026**

This excludes the legacy records with unreliable elapsed-time timestamps.

There are:

**7,728 helpdesk-era tickets**

available for this analysis.

No helpdesk-era rows are missing the required first-response timestamps.

---

## 9.2 Breach Definition

For each ticket:

`first_response_minutes = first_response_at - created_at`

A breach is recorded when:

`first_response_minutes > channel_target`

The supplied policy targets are:

| Channel | First-Response Target |
|---|---:|
| Chat | 15 minutes |
| Voice callback | 2 hours |
| Social | 4 hours |
| Email | 8 hours |

A response exactly at the target is therefore not considered a breach.

---

## 9.3 Current Breach Rate

The analysis finds:

**839 breaches out of 7,728 tickets**

or approximately:

**10.9% first-response breach rate**

The approximate channel-level rates are:

- Chat: ~12%
- Email: ~12%
- Voice: 5.4%

---

## 9.4 Annualized Financial Exposure

The observed helpdesk period is annualized using the observed ticket volume and breach rate.

The resulting estimate is:

**₹370,873 per year**

or approximately:

**₹92,718 per quarter**

The calculation is based on:

`annualized tickets × breach rate × ₹350`

The ₹350 amount comes from the supplied support policy.

The annualization is an estimate and assumes that the observed ticket volume and breach rate continue at approximately the same rate throughout a year.

It is **not confirmed cash paid** because the ticket export does not contain an explicit field confirming that an SLA credit was actually issued.

---

## 9.5 Illustrative Scenario

A simple illustrative scenario is also included:

**10.9% breach rate → 5.4% breach rate**

This would correspond to approximately:

**₹185,500 annual SLA-credit exposure**

or approximately:

**₹46,375 per quarter**

This is arithmetic only.

It is not a Vireo target, forecast, or measured outcome.

---

# 10. Headcount Question

The original client rule was:

> "Whichever team has the most volume gets the next two hires."

The analysis does not automatically apply this rule as a hiring recommendation.

Instead, it compares several pieces of evidence.

### Raw ticket volume

Chat Frontline has the highest historical volume:

**3,030 tickets / 25.7%**

Billing is second:

**2,564 tickets / 21.8%**

### Volume per active agent

Billing has the highest tickets-per-agent ratio:

**641 tickets per active agent**

followed by Logistics at:

**381 tickets per active agent**

### Other workload indicators

The teams differ substantially in:

- team size
- handle time
- transfer rate
- high-priority share
- type of work handled

For example, Escalations & Warranty has lower ticket volume but substantially longer median handle time.

Therefore, ticket count alone is not sufficient to convert the analysis into a defensible staffing requirement.

The project provides the evidence needed for the staffing discussion but does not claim that the data alone proves which team should receive the next two hires.

A stronger staffing decision would require additional information such as:

- time-on-ticket data
- backlog
- queue wait time
- staffing by historical period
- capacity/utilization
- independently validated category reclassification

---

# 11. Source Facts, Derived Metrics, and Assumptions

| Item | Type | Source |
|---|---|---|
| Ticket count and dates | Source fact | `tickets.csv` |
| Historical team assignment | Source fact | `tickets.csv` |
| Agent roster | Source fact | `agents.csv` |
| ₹350 SLA-credit rule | Policy fact | `support-policy.pdf` |
| Channel SLA targets | Policy fact | `support-policy.pdf` |
| 10.9% breach rate | Measured result | Ticket timestamps |
| 3,030 Chat Frontline tickets | Measured result | `tickets.csv` |
| 2,564 Billing tickets | Measured result | `tickets.csv` |
| 65.0% Billing reclassification | Rule-based model output | `src/categorize.py` |
| 94.0% validation agreement | AI-assisted first-pass review | `src/validate.py` |
| ₹370,873 annual exposure | Estimate | Measured breach rate × policy credit × annualization |
| ₹92,718 quarterly exposure | Estimate | Annual estimate / 4 |
| Halved-breach scenario | Illustrative scenario | Arithmetic assumption |
| Tickets per agent | Derived metric | Ticket volume / current active roster |

---

# 12. Key Assumptions

The analysis uses the following assumptions:

1. The existing 11-category taxonomy is retained.
2. Historical `assigned_team` values represent the actual routing recorded in the source data.
3. `predicted_category` is used to audit the existing categorization rather than silently rewrite historical routing.
4. Rule-based categorization is used instead of an LLM or trained ML classifier.
5. Legacy tickets with invalid elapsed-time timestamps are excluded from time-based calculations.
6. Current active agent headcount is used for the tickets-per-agent workload proxy.
7. The SLA annualization assumes that observed ticket volume and breach rates remain approximately stable.
8. The ₹350 SLA-credit amount is taken from the supplied policy.
9. The annual SLA-credit figure is an estimated liability, not confirmed cash paid.
10. The validation result is an AI-assisted first-pass review and is not an independent human audit.

---

# 13. Limitations

### Categorization validation

The 94.0% result is based on a 150-ticket sample and AI-assisted labels rather than an independent human audit.

A fresh, independently labelled sample is needed before the classifier is used for high-stakes automated routing or staffing decisions.

### Taxonomy limitations

`Other` and promotional discount/coupon issues are weaker parts of the current taxonomy.

A dedicated `Promotions / Pricing` category could be considered in a future iteration, but changing the taxonomy was outside the scope of this pass.

### Legacy timestamps

2,379 legacy tickets have unreliable elapsed-time timestamps.

They remain available for ticket-count and category analysis but are excluded from SLA and handle-time calculations.

### SLA-credit verification

The dataset does not contain a field confirming that a ₹350 credit was actually issued for every SLA breach.

Therefore, the ₹370,873 annual figure represents estimated policy-based exposure rather than confirmed cash expenditure.

### Staffing analysis

Tickets per agent is a useful comparison metric but is not a complete staffing-capacity model.

The current roster is not time-matched to the entire historical ticket period, and the dataset does not contain sufficient utilization or time-on-ticket information to determine exact hiring requirements.

### Additional datasets

`customers.csv`, `orders.csv`, and `products.csv` were reviewed but were not required for the final analysis.

They can support future analyses such as customer segmentation, order-value weighting, and replacement-cost exposure.

---

# 14. Reproducibility

The complete analysis can be reproduced from the project root with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py