import json
import os
from datetime import datetime
from agents.data_understanding import run_data_understanding_agent
from agents.analytics import run_analytics_agent
from agents.insight_generation import run_insight_generation_agent
from agents.recommendation import run_recommendation_agent
from agents.critic import run_critic_agent


def run_pipeline(data_filepath: str, scenario_name: str) -> dict:
    """
    Full 5-agent pipeline for SME business insight generation.
    """

    print("\n" + "#"*60)
    print(f"  SME INSIGHT AUTOMATION PIPELINE")
    print(f"  Scenario: {scenario_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)

    # Create scenario-specific output folder
    scenario_slug = scenario_name.replace(" ", "_").lower()
    output_dir = f"outputs/{scenario_slug}"
    os.makedirs(output_dir, exist_ok=True)

    agent1_output = run_data_understanding_agent(data_filepath)
    agent2_output = run_analytics_agent(data_filepath, agent1_output)
    agent3_output = run_insight_generation_agent(agent2_output)
    agent4_output = run_recommendation_agent(agent2_output, agent3_output)
    agent5_output = run_critic_agent(agent2_output, agent3_output, agent4_output)

    # Save individual outputs to scenario folder
    for filename, data in [
        ("agent1_data_understanding.json", agent1_output),
        ("agent2_analytics.json", agent2_output),
        ("agent3_insights.json", agent3_output),
        ("agent4_recommendations.json", agent4_output),
        ("agent5_critic.json", agent5_output),
    ]:
        with open(f"{output_dir}/{filename}", "w") as f:
            json.dump(data, f, indent=2, default=str)

    # Final report
    final_report = {
        "scenario": scenario_name,
        "data_file": data_filepath,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_status": "completed",
        "agents_run": 5,
        "kpis": agent2_output["analytics"]["kpis"],
        "insights": agent3_output["insights"],
        "recommendations": agent4_output["recommendations"],
        "quality_review": agent5_output["review"],
        "ground_truth": agent5_output["ground_truth_used"]
    }

    report_path = f"{output_dir}/final_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {scenario_name}")
    print(f"{'='*60}")
    print(f"Total Revenue     : £{agent2_output['analytics']['kpis']['total_revenue']:,}")
    print(f"Total Transactions: {agent2_output['analytics']['kpis']['total_transactions']:,}")
    print(f"Insights Generated: 8")
    print(f"Recommendations   : 6")
    print(f"Report saved to   : {report_path}")

    return final_report


def run_all_scenarios():
    """Run the full pipeline across all three SME scenarios."""

    scenarios = [
        ("data/retail_sme_data.csv",       "Retail SME"),
        ("data/food_service_data.csv",      "Food Service SME"),
        ("data/distribution_data.csv",      "Distribution SME"),
    ]

    all_results = []
    summary = []

    print("\n" + "#"*60)
    print("  RUNNING ALL THREE SME SCENARIOS")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)

    for filepath, scenario_name in scenarios:
        result = run_pipeline(filepath, scenario_name)
        all_results.append(result)
        summary.append({
            "scenario": scenario_name,
            "total_revenue": result["kpis"]["total_revenue"],
            "total_transactions": result["kpis"]["total_transactions"],
            "gross_margin_pct": result["kpis"]["overall_gross_margin_pct"],
            "timestamp": result["timestamp"]
        })

    # Cross-scenario summary
    print("\n" + "#"*60)
    print("  ALL SCENARIOS COMPLETE — CROSS-SCENARIO SUMMARY")
    print("#"*60)
    for s in summary:
        print(f"\n  {s['scenario']}")
        print(f"    Revenue      : £{s['total_revenue']:,}")
        print(f"    Transactions : {s['total_transactions']:,}")
        print(f"    Gross Margin : {s['gross_margin_pct']}%")

    with open("outputs/all_scenarios_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: outputs/all_scenarios_summary.json")

    return all_results


if __name__ == "__main__":
    run_all_scenarios()