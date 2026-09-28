# AI Usage

## Tools and models

AI was used selectively as an engineering assistant during the development and review of this project.

The main AI tools used were:

- **Claude (Anthropic)** — used for implementation assistance, debugging, code iteration, and review of the project against the assignment requirements.
- **ChatGPT (OpenAI)** — used for additional review, clarification, and quality checking of the implementation and submission materials.

The submitted pipeline itself does **not** call an LLM or any paid AI API. Ticket categorization is performed locally using Python and deterministic keyword/regex rules.

**Pipeline API cost: ₹0.**

---

## Where AI was used

### 1. Initial implementation

AI assistance was used to help structure and implement the main components of the project, including:

- data loading and cleaning
- ticket categorization
- validation
- monthly/category/team analysis
- SLA analysis
- business-impact calculations
- output generation
- automated tests

The implementation was then run locally against the supplied dataset and checked through the project's test suite.

### 2. Categorization design and debugging

AI was used to develop and refine the rule-based ticket categorization approach.

The rules were tested against the actual **11,780-ticket dataset**, and problematic patterns were investigated and narrowed when they produced false matches.

Examples of issues identified during iteration included:

- delivery rules incorrectly matching login-code messages
- generic "stuck" wording matching firmware issues
- payment-related wording incorrectly triggering Billing classification
- routing phrases such as "escalations & warranty" overriding the customer's actual issue
- cancellation wording being missed when the words appeared in a different order

These issues were corrected and regression tests were added for the identified cases.

### 3. Validation design and review

AI was used to review a fixed sample of **150 tickets** and compare the predicted category with the reviewed category.

The final result was:

**141/150 agreement = 94.0%**

This is an **AI-reviewed first-pass agreement figure**, not an independent human audit.

The validation sample uses a fixed seed and is intended to provide evidence about the behavior of the categorization approach on a reviewed sample. It should not be interpreted as a definitive production accuracy estimate.

### 4. Project quality review

AI was also used for a final quality review of:

- categorization logic
- validation methodology
- SLA calculations
- team/headcount analysis
- automated tests
- documentation
- submission completeness

This review identified and corrected several implementation and documentation issues before the final pipeline run.

---

## What was not AI-powered

The submitted ticket-processing pipeline does **not** send ticket data to an external LLM.

The production categorization flow is:

```text
Ticket text
    ↓
Local Python processing
    ↓
Keyword / regex rules
    ↓
Category assignment
    ↓
Local analysis and reporting