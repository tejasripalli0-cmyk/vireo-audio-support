# Submission Form — Vireo Audio, Support Tickets (Set E)

## 1. What did you build, and what business outcome does it move? State the number and the money.

I built a small, deterministic Python pipeline (`python run.py`) that:

- reads all 11,780 tickets and re-categorises each one from its text (customer message first, agent note as weaker secondary evidence) using transparent keyword rules, with no LLM and no paid API;
- produces the monthly breakdowns by category and by team that Priya asked for, plus workload views (tickets per active agent, handle time, transfer rate);
- tests the "biggest team gets the two hires" rule against the data;
- quantifies one policy-backed business number.

**The business number.** The support policy (§3) says every ticket that misses its first-response target automatically earns the customer a ₹350 credit.

| Item | Value | Status |
|---|---|---|
| Tickets in scope (helpdesk era, 14 Sep 2025 – 30 Jun 2026) | 7,728 | measured |
| Missed first-response target | 839 = **10.9%** | measured |
| Annualised estimate (7,728 tickets over 289 days → ~9,760 a year; ~1,060 breaches) | 1,060 × ₹350 = **₹370,873 a year, ≈ ₹92,718 a quarter** | estimate |

Goal stated as a number: **cut first-response breaches from 10.9% to 5.4% (illustrative target: half the current rate), worth about ₹46,375 a quarter (₹185,500 a year).** The 5.4% target is *my illustrative assumption*, not a Vireo target and not a prediction. Vireo gave no target anywhere in the pack. Only the 10.9% and the ₹92,718 baseline are measured or estimated from the data.

I cannot confirm from the data that the ₹350 credits are actually being paid. There is no "SLA credit" refund reason code in `tickets.csv`. So ₹370,873 is a **policy-implied liability**, not confirmed cash paid.

**The headcount finding.** Of the 2,564 tickets the intake bot tagged "Billing & Payments", my rule-based re-read reassigns 1,655 (64.5%) to another category, 1,062 of them (41.4%) to Delivery & Shipping. Only 909 (35.5%) stay Billing. This is a rule-based reclassification, not verified ground truth (see Q3). It means raw ticket volume overstates how much genuine billing work Billing has.

## 2. What does one run cost, and what would a month cost at Vireo's volume (roughly 650 tickets a week)? Show the arithmetic.

**No paid calls were used. One run costs ₹0 / $0.** The classifier is local regex scoring in pandas. It makes no network calls, so cost does not depend on volume. A full run over 11,780 tickets takes a few seconds on a laptop.

For reference only, here is what an LLM classifier *would* have cost. These rates are my **assumed, illustrative** small-model prices, not a quote:

- 650 tickets/week × ~200 input tokens = 130,000 tokens; 650 × ~50 output tokens = 32,500 tokens.
- At an assumed $0.25 per million input and $1.25 per million output: $0.0325 + $0.0406 ≈ $0.07/week ≈ **$0.31/month**.

That is trivial, so I did not avoid an LLM because of cost. I avoided it because there are no ground-truth labels to check it against, the text is short and keyword-dense, and rules can be read and edited by an ops person.

Note: the data actually runs at about 730–830 tickets a *month* in the last six months (roughly 170–190 a week), well below 650 a week. I used the client's figure only for this cost question.

## 3. How do you know it works? Sample size, how you checked, error rate, and the kind of case it gets wrong.

- **Sample:** 150 tickets, a fixed random sample (`random_state=42`), reproducible.
- **How I checked:** each ticket's text was read and given a "true" category based on the customer's stated problem. Then I compared it with the rule-based prediction.
- **Result: 141 of 150 agree = 94.0% AI-reviewed agreement; 9 errors = 6.0% error rate.**
- **This is not an independent human audit.** The same AI session (Claude) that wrote the rules also assigned the reference labels. I call the figure "AI-reviewed first-pass agreement", not "accuracy".
- **The 94% is optimistic.** I fixed rules after reading errors in this same 150-ticket sample, so it is not a held-out estimate. Two extra tickets (TK-253080, TK-242694) are counted as correct because their customer messages alone read as delivery waits, but a different reviewer could disagree.
- **What it gets wrong (the 9 actual errors):**
  - Promotional discount or coupon disputes have no home in the 11-category taxonomy (3).
  - The customer describes a real problem in wording the rules don't recognise, e.g. "keeps losing my phone" (3).
  - Cancellation intent appears only in the agent note (1).
  - "a crack" versus "cracked" in a transit-damage ticket (1).
  - A real pairing complaint outweighed by resolution wording in the note (1).
- Per-category precision is on tiny sub-samples and is only directional. Details are in `outputs/validation_report.json`.

## 4. Did you change, narrow, or push back on the client's ask? What, when, and why?

**Pushed back, with evidence, on the headcount rule.** Priya asked for volume by category and team and expected "two hires to Billing". In EDA I found that four categories route 1:1 to a team, so a wrong bot tag puts the ticket in the wrong queue. The Billing reclassification then showed that raw volume overstates Billing's genuine work. That fits Arjun's request to see the case in writing and his preference to "fix a process".

I did **not** conclude that Billing shouldn't get the hires, and I did **not** conclude that Logistics should. The memo says the volume rule is an incomplete proxy and recommends re-running the case on corrected numbers.

Other findings that cut both ways:
- Raw volume: Chat Frontline is largest (3,030, 25.7%), then Billing (2,564, 21.8%), then Logistics (1,905, 16.2%). That matches Priya's 22% and 16%.
- Billing has the most tickets per active agent (641 all-time; 4 agents).
- Billing's median handle time is short (about 2 hours), but its transfer rate is 31.5% against 5.6–11.3% elsewhere.
- Logistics' median handle time is about 26 hours, which is consistent with Neha's "a day plus".
- Escalations & Warranty is Tier 2 and should not be compared with Tier 1 on volume (policy §6).

**Narrowed:** I picked SLA breach as the money metric and set repeat contacts and refunds aside (Q6). I did not re-route historical tickets.

## 5. What is wrong with what you are handing us? Be specific: bugs, shortcuts, things you know are off.

- The 94.0% is AI-reviewed, not independent, and partly tuned on its own sample. There is no held-out set.
- The customer-message vs agent-note weighting (1.0 vs 0.5) and every regex weight are judgment calls, not tuned or cross-validated.
- The Billing headline (64.5% reclassified) depends on rules with a roughly 6% error rate on a small sample. The direction is clear, but treat the exact percentages as approximate.
- "Waiting for your courier" tickets can be either a delivery wait or an awaited return pickup. The classifier now reads the customer's words, so some pickup tickets land in Delivery.
- The taxonomy lacks a Promotions/Pricing category, so discount disputes land inconsistently.
- 2,379 legacy tickets (20%) have `resolved_at` earlier than `created_at`, most likely a UTC/IST artefact (policy §9). I excluded them from all time-based metrics rather than guess a correction. SLA and handle-time results therefore use only 7,728 helpdesk-era tickets.
- The ₹370,873 assumes the policy's ₹350 credit is applied on every breach. The data cannot confirm that.
- The data-pack README says tickets start 1 Jan 2025, but the file starts 20 Jun 2024. I left it as is.
- `team_workload_summary` handle times cover resolved helpdesk-era tickets only.
- `docs/methodology.md` still says the roster has "45 rows"; `agents.csv` has 44 rows, one per agent. [FIX BEFORE SUBMITTING]
- No CI. Tests run manually with `python -m pytest -q` (32 tests pass).

## 6. What did you deliberately leave out, and why that rather than something else?

- **LLM or trained classifier.** No labels to train or check against, and rules are auditable. Cheap to add later behind the same function.
- **Repeat-contact cost (policy §10).** It needs "same customer, same issue within 30 days", which would stack a second imperfect signal on top of the categoriser.
- **Refund and replacement cost.** These are mostly legitimate policy outcomes, not an obvious lever.
- **`customers.csv`, `orders.csv`, `products.csv`.** Reviewed, not joined. None was needed for categorisation, monthly views, headcount or the SLA number. Product unit cost would matter for a replacement-cost case.
- **Re-routing history and a UI, database or dashboard.** Not asked for, and out of proportion for a 5-hour task.

## 7. Anything you built or found that nobody asked for?

- **Bot mistags caught in agent notes.** Some closing notes say outright that the bot tagged the ticket wrong (e.g. "Bot tagged as billing, actually shipment issue. Moved to Logistics"). This is direct evidence for the Billing finding.
- **The legacy timestamp problem** (Q5) and the exclusion it forced.
- **No SLA-credit reason code** in `refund_reason_code`, so it is unclear whether the credit is tracked or paid.
- **A validation guard.** The validator raises if the sample no longer contains the reviewed tickets, so it can't silently under-report errors.
- **A bug fix.** `team_workload_summary` claimed to include headcount but didn't; it now merges active agents.

## 8. What did you use AI for? Which tools and models, where they helped, where they wasted your time, what you threw away. Link your three-minute screen recording here.

- **Tool:** Claude (Anthropic) in a chat session with code execution, used for data inspection, rule design and iteration, all source code and tests, the analysis, charts, and drafting these documents. [ADD ANY OTHER TOOL YOU ACTUALLY USED, e.g. ChatGPT for review. Do not list one you did not use.]
- **Paid API calls:** none. Pipeline cost ₹0.
- **Helped:** fast profiling of every file; testing each rule against all 11,780 tickets and reading where it disagreed; turning each bug into a regression test.
- **Wasted time / threw away:** an early rule set with broad words ("package", "delay", "stuck") reached about 58% agreement with the bot's tags but pulled in unrelated tickets, so I discarded it. Rules that scored the customer message and agent note equally let a note like "xfer escalations & warranty" override a charging complaint. I replaced that with message-first weighting. Even my own validation labels were reconsidered mid-way.
- **Disclosure:** the AI wrote the classifier and also produced the validation labels. That conflict is stated in Q3.


## 10. Someone picks this up on Monday and you are unreachable. The three things they need to know.

1. **The headline is a rule-based inference, not proof.** "64.5% of Billing-tagged tickets reclassified" comes from `predicted_category`, checked only by an AI-reviewed first pass (94.0%, 9 errors of 150). Before anyone changes headcount, get one Tier-2 agent to independently label a fresh sample. Do not present the numbers as verified.
2. **Everything regenerates from `data/` with `python run.py`; never hand-edit `outputs/`.** Fix rules in `src/categorize.py`, then rerun and run `python -m pytest -q`. The 9 reviewed errors live in `src/validate.py` and will raise an error if the sample changes, so re-review rather than editing the list to make it pass.
3. **The ₹ number is a policy-implied estimate, not cash paid.** ₹370,873 a year (₹92,718 a quarter) comes from 10.9% breaches × ₹350 and assumes the credit is applied. Ask Finance whether it is. The 5.4% target and ₹46,375 a quarter are illustrative only.

## 11 Hours spent.

[6-7]

## 12. Github Repo Link

[https://github.com/tejasripalli0-cmyk/vireo-audio-support]