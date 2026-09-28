# Submission Form

## What did you build, and what business outcome does it move? State the number and the money.

A reproducible pipeline (`python run.py`) that re-categorizes all 11,780 Vireo support tickets from their actual text rather than relying only on the intake tag. It also validates the categorization on a sample, produces the requested monthly-by-category and monthly-by-team analysis, and tests whether the client's proposed "most volume gets the next two hires" rule is supported by the available workload data.

Two key numbers:

1. Of the 2,564 tickets tagged `Billing & Payments` by the intake bot, the rule-based re-read assigns:
   - 898 (35.0%) to Billing & Payments
   - 1,049 (40.9%) to Delivery & Shipping
   - 617 (24.1%) to other categories

   Therefore, 1,666 tickets (65.0%) differ from the bot's Billing tag. These are rule-based predictions, not verified ground truth. They indicate that the Billing tag may overstate Billing-related work, so raw volume alone is an incomplete basis for the two-hire decision.

2. First-response SLA breaches are 10.9% of helpdesk-era tickets (839 of 7,728). The support policy specifies an automatic ₹350 credit for a missed target. Annualizing the observed rate gives an estimated **₹370,873/year (approximately ₹92,718/quarter)** in policy-based liability.

   This is an estimate, not confirmed cash paid. Vireo has not provided a target breach rate, so no target was invented.

## What does one run cost, and what would a month cost at Vireo's volume (roughly 650 tickets a week)? Show the arithmetic. If you used no paid calls, say so.

**No paid API calls were used. One run costs ₹0 / $0.**

The shipped classifier in `src/categorize.py` is a local, deterministic keyword/regex classifier. It makes no network calls and has no per-ticket API charge.

At Vireo's stated volume:

- Approximately 650 tickets/week
- Approximately 2,600 tickets/month
- Production categorization cost: **₹0 / $0 per month**

No LLM or paid API is used by the production pipeline.

## How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

A fixed 150-ticket random sample was used (uniform sample, seed 42, taxonomy v1.0-2026-09).

The sample received an AI-assisted first-pass review against the customer's stated problem.

**Result: 94.0% agreement (141/150), 9 disagreements, 6.0% error rate.**

This is **not an independent human audit**. Only the 12 originally flagged tickets have stored true labels; the other 138 were judged during the review but were not independently re-checked.

After fixing two general bugs found during review — the routing phrase "Escalations & Warranty" affecting category scoring and a word-order-dependent cancellation rule — the validation result improved to 94.0%.

Observed failure types include:

- Promotional/discount disputes with no clean category in the existing taxonomy
- Unmatched phrasings such as "strap pin came off", "spinning circle", "a crack", and "keeps losing my phone"
- Cases where a resolution keyword such as "DOA" or "reverse pickup" outweighs the customer's main symptom

Details are recorded in `outputs/validation_report.json`.

## Did you change, narrow, or push back on the client's ask? What, when, and why?

Yes.

The client proposed that the team with the most volume should receive the next two hires. The analysis does not automatically accept that rule because the available data shows that raw ticket volume and workload are not the same measure.

The analysis therefore compares:

- Raw team volume
- Tickets per active agent
- Handling-time information
- Transfers
- Channel mix
- SLA performance

The analysis does **not** select a team for the two hires. Instead, it shows why the staffing decision should be validated using more than raw ticket count.

Separately, several possible business metrics were considered. SLA-breach cost was selected as the main quantified business metric because it has a direct policy-defined monetary value in the provided support policy.

## What is wrong with what you are handing us? Be specific: bugs, shortcuts, things you know are off.

- Categorization agreement is 94% on 150 tickets, not 100%; 9 sampled cases disagreed with the reviewed labels.
- The validation review was AI-assisted and was not an independent human audit.
- Only 12 tickets have stored true labels, so the validation set is limited.
- Promotional/discount disputes do not have a clean category in the existing taxonomy and may land inconsistently.
- 2,379 legacy-era tickets have unusable `resolved_at` timestamp ordering. Affected time-based calculations exclude invalid timestamps rather than attempting an unverified timezone correction.
- The SLA figure of ₹370,873/year is an estimated policy-based liability, not confirmed cash paid. It assumes the observed helpdesk-era rate continues for a year.
- `team_workload_summary.csv` handle times are workload indicators, not direct measures of staffing capacity.
- `tickets_per_agent.csv` divides historical ticket volume by the current roster, so it should not be interpreted as historical staffing-adjusted productivity.
- `team_volume_summary.csv` uses `assigned_team` as the historical routing field. The pipeline does not silently rewrite historical team assignments.
- There is no CI/pre-commit setup; tests must be run manually with `python -m pytest -q`.
- Some vocabulary gaps, including discount/coupon language and several uncommon phrasings, were deliberately not tuned to the validation sample.

## What did you deliberately leave out, and why that rather than something else?

- **No re-routing of historical tickets.**
  `predicted_category` is used to audit the existing routing rather than silently overwrite `assigned_team`. Changing historical routing would hide the routing problem being investigated.

- **`customers.csv`, `orders.csv`, and `products.csv` were not joined into the final analysis.**
  The required outputs — categorization, monthly breakdown, validation, and a quantified business number — could be produced without these joins. Adding them would increase complexity without a clear analytical payoff within the assignment's time limit. A future replacement-cost analysis using `products.csv.unit_cost_inr` would be a logical extension.

- **Repeat-contact rate was not used as the final business metric.**
  It was considered, but SLA-breach cost was chosen because the support policy provides a direct monetary definition.

- **No LLM/embedding classifier was used in production.**
  A deterministic rule-based approach was chosen because the task has no large verified training set, the ticket text is relatively short and keyword-dense, and the production requirement benefits from a simple, reproducible classifier.

## Anything you built or found that nobody asked for?

- The `invalid_date_order` flag and investigation of the legacy timestamp issue. This was important because invalid timestamps affect the reliability of time-based calculations.
- The observation that the provided `refund_reason_code` values do not contain a specific SLA-breach-credit code. This is left as an open question rather than assuming whether credits were actually issued.
- `tickets_per_agent.csv` and the related chart, to provide another workload view for the staffing question.
- Regression tests for classification edge cases and data-quality calculations.

## What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.

AI agents were used selectively as a development and review assistant rather than as the production classifier.

Main uses included:

- Reviewing categorization rules and identifying edge cases.
- Helping debug validation and analysis logic.
- Suggesting regression-test cases for discovered bugs.
- Reviewing wording and structure of the analysis documentation.
- Assisting with interpretation of classification failures.

The shipped production pipeline itself is deterministic Python code and makes no LLM or paid API calls.

AI-assisted review was also used for the 150-ticket validation sample, so that validation should be treated as a first-pass review rather than an independent human audit.

Discarded work included overly broad keyword rules that produced misleading matches. The final rules were narrowed to require more context rather than simply maximizing agreement with the existing bot tag.

Screen recording:
[ADD SCREEN RECORDING LINK]

## Your Public Google Drive Link

[ADD PUBLIC GOOGLE DRIVE LINK]

## Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **The Billing-tag audit is the key finding.**

   `outputs/business_impact.json` → `category_tag_audit` contains the exact figures:

   - 898 retained as Billing & Payments
   - 1,049 reclassified to Delivery & Shipping
   - 617 classified elsewhere
   - 2,564 originally tagged Billing & Payments

   These are rule-based predictions, not verified ground truth.

2. **The 94% validation result has an important limitation.**

   The review was AI-assisted rather than an independent human audit, and only 12 tickets have stored true labels. Before using the categorization to make an irreversible staffing or routing decision, an independent reviewer should check a new sample.

3. **Everything regenerates from `data/`.**

   Run:

   `python run.py`

   Do not hand-edit files in `outputs/`. If a number looks wrong, inspect the relevant logic in `src/categorize.py`, `src/analyze.py`, or the tests, then rerun the pipeline.

## Honest hours spent. One number.

[ADD ACTUAL HOURS SPENT — 6-7]

## Github Repo Link

[ADD GITHUB URL]