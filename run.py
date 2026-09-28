"""
Vireo Audio support-ticket pipeline.

Usage:
    python run.py
    python run.py --input data/tickets.csv --agents data/agents.csv
    python run.py --help
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import (
    analyze,
    categorize,
    clean_data,
    config,
    load_data,
    report,
    validate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


def parse_args():
    p = argparse.ArgumentParser(
        description="Vireo Audio support-ticket categorization, validation and business-impact pipeline."
    )
    p.add_argument("--input", type=Path, default=config.TICKETS_CSV, help="Path to tickets.csv")
    p.add_argument("--agents", type=Path, default=config.AGENTS_CSV, help="Path to agents.csv")
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR, help="Directory for outputs")
    p.add_argument("--skip-charts", action="store_true", help="Skip chart generation (faster)")
    return p.parse_args()


def main():
    args = parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("STEP 1/9: Loading data")
    try:
        tickets = load_data.load_tickets(args.input)
        agents = load_data.load_agents(args.agents)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("STEP 2/9: Validating & cleaning tickets")
    tickets = clean_data.clean_tickets(tickets)

    logger.info("STEP 3/9: Categorizing tickets (rule_based_keyword_v1, zero API cost)")
    tickets = categorize.categorize_tickets(tickets)

    logger.info("STEP 4/9: Running categorization validation")
    val_report = validate.run_validation(tickets)
    validate.save_validation_report(val_report, args.output_dir / "validation_report.json")
    logger.info("Validation accuracy: %.1f%% (%d/%d)", val_report["accuracy"] * 100,
                val_report["n_correct"], val_report["sample_size"])

    logger.info("STEP 5/9: Monthly + team analysis")
    monthly_cat = analyze.monthly_by_category(tickets)
    monthly_team = analyze.monthly_by_team(tickets)
    team_volume = analyze.team_volume_summary(tickets)
    team_workload = analyze.team_workload_summary(tickets, agents)
    per_agent = analyze.tickets_per_agent(tickets, agents)
    audit = analyze.category_tag_audit(tickets)

    logger.info("STEP 6/9: Business impact (SLA breach cost)")
    sla_result = analyze.sla_breach_analysis(tickets)
    scenarios = analyze.business_impact_scenarios(sla_result)
    business_impact = {**sla_result, "scenarios": scenarios, "category_tag_audit": audit}

    logger.info("STEP 7/9: Writing output files")
    report.write_categorized_tickets(tickets)
    report.write_csv(monthly_cat, "monthly_category.csv")
    report.write_csv(monthly_team, "monthly_team.csv")
    report.write_csv(team_volume, "team_volume_summary.csv")
    report.write_csv(team_workload, "team_workload_summary.csv")
    report.write_csv(per_agent, "tickets_per_agent.csv")
    report.write_json(business_impact, "business_impact.json")

    if not args.skip_charts:
        logger.info("STEP 8/9: Generating charts")
        report.chart_monthly_category(monthly_cat)
        report.chart_monthly_team(monthly_team)
        report.chart_team_volume_vs_workload(per_agent)
        report.chart_billing_reclassification(audit)
        report.chart_sla_breach_by_channel(sla_result)
    else:
        logger.info("STEP 8/9: Skipped (--skip-charts)")

    logger.info("STEP 9/9: Done. Outputs in %s", args.output_dir)
    print("\n=== SUMMARY ===")
    print(f"Tickets processed: {len(tickets)}")
    print(f"Categorization validation accuracy: {val_report['accuracy']*100:.1f}% "
          f"({val_report['n_correct']}/{val_report['sample_size']})")
    print(f"Current-era first-response breach rate: {sla_result['overall_breach_rate']*100:.1f}%")
    print(f"Estimated current annual SLA-credit liability: Rs {sla_result['current_annual_liability_inr']:,.0f}")
    print(f"Estimated current quarterly SLA-credit liability: Rs {sla_result['current_quarterly_liability_inr']:,.0f}")
    print(f"Of {audit['billing_tagged_total']} bot-tagged 'Billing & Payments' tickets, "
          f"{audit['billing_reclassified_to_other_category']} ({audit['billing_reclassified_pct']}%) "
          f"were reclassified to a different category on re-reading the ticket text.")
    print(f"\nFull outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()