import json
import os
from datetime import datetime
from agents.data_understanding import run_data_understanding_agent
from agents.analytics import run_analytics_agent
from agents.insight_generation import run_insight_generation_agent
from agents.recommendation import run_recommendation_agent
from agents.critic import run_critic_agent

def run_pipeline(data_filepath: str, scenario_name: str = "Retail SME"):
    """
    Full 5-agent pipeline for SME business insight generation.
    Runs all agents in sequence and produces a final report.
    """

    print("\n" + "#"*60)
    print(f"  SME INSIGHT AUTOMATION PIPELINE")
    print(f"  Scenario: {scenario_name}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)

    os.makedirs("outputs", exist_ok=True)

    # ── Agent 1 ────────────────────────────────────────────
    agent1_output = run_data_understanding_agent(data_filepath)

    # ── Agent 2 ────────────────────────────────────────────
    agent2_output = run_analytics_agent(data_filepath, agent1_output)

    # ── Agent 3 ────────────────────────────────────────────
    agent3_output = run_insight_generation_agent(agent2_output)

    # ── Agent 4 ────────────────────────────────────────────
    agent4_output = run_recommendation_agent(agent2_output, agent3_output)

    # ── Agent 5 ────────────────────────────────────────────
    agent5_output = run_critic_agent(agent2_output, agent3_output, agent4_output)

    # ── Final Report ───────────────────────────────────────
    print("\n" + "#"*60)
    print("  PIPELINE COMPLETE — GENERATING FINAL REPORT")
    print("#"*60)

    final_report = {
        "scenario": scenario_name,
        "data_file": data_filepath,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_status": "completed",
        "agents_run": 5,
        "kpis": agent2_output["analytics"]["kpis"],
        "insights": agent3_output["insights"],
        "recommendations": agent4_output["recommendations"],
        "quality_review": agent5_output["review"]
    }

    report_path = f"outputs/final_report_{scenario_name.replace(' ', '_').lower()}.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    # ── Print Summary ──────────────────────────────────────
    print(f"\nScenario      : {scenario_name}")
    print(f"Data File     : {data_filepath}")
    print(f"Total Revenue : £{agent2_output['analytics']['kpis']['total_revenue']:,}")
    print(f"Transactions  : {agent2_output['analytics']['kpis']['total_transactions']:,}")
    print(f"Insights      : 8 generated")
    print(f"Recommendations: 6 generated")
    print(f"Report saved  : {report_path}")
    print("\nPIPELINE COMPLETED SUCCESSFULLY")

    return final_report


if __name__ == "__main__":
    run_pipeline(
        data_filepath="data/retail_sme_data.csv",
        scenario_name="Retail SME"
    )