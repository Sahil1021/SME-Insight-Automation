from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

def run_critic_agent(analytics_output: dict,
                     insights_output: dict,
                     recommendations_output: dict) -> dict:
    """
    Agent 5: Critic Agent
    Reviews all pipeline outputs for factual consistency,
    hallucinations, completeness, and actionability.
    """

    print("\n" + "="*60)
    print("AGENT 5: CRITIC AGENT")
    print("="*60)

    analytics = analytics_output["analytics"]
    kpis = analytics["kpis"]
    pareto = analytics["pareto_analysis"]
    monthly = analytics["monthly_sales_trend"]
    inventory = analytics["inventory_analysis"]

    # Ground truth from Python computation
    ground_truth = {
        "total_revenue": kpis["total_revenue"],
        "total_gross_profit": kpis["total_gross_profit"],
        "gross_margin_pct": kpis["overall_gross_margin_pct"],
        "total_transactions": kpis["total_transactions"],
        "avg_transaction_value": kpis["avg_transaction_value"],
        "peak_month": monthly["peak_month"],
        "lowest_month": monthly["lowest_month"],
        "peak_revenue": monthly["peak_revenue"],
        "top_products": pareto["top_20pct_products"],
        "top_products_revenue_share": pareto["top_20pct_revenue_share"],
        "slow_moving_products": inventory["slow_moving_products"],
        "stockout_products": inventory["stockout_products"],
        "best_region": kpis["best_performing_region"],
        "top_supplier": kpis["top_supplier_by_revenue"]
    }

    prompt = f"""
You are a critical quality reviewer for an AI business analytics system.
Your job is to check whether the generated insights and recommendations
are factually consistent with the computed analytical ground truth.

GROUND TRUTH (computed by Python, 100% accurate):
{json.dumps(ground_truth, indent=2)}

GENERATED INSIGHTS:
{insights_output['insights']}

GENERATED RECOMMENDATIONS:
{recommendations_output['recommendations']}

Perform a thorough quality review covering these five dimensions:

1. FACTUAL CONSISTENCY CHECK:
List any specific facts in the insights or recommendations that contradict
the ground truth. Quote the exact text and state what the correct value is.
If all facts are consistent, state "All facts verified as consistent."

2. HALLUCINATION CHECK:
Identify any claims, statistics, or figures mentioned that do not appear
in the ground truth data and cannot be verified. If none, state "No
hallucinations detected."

3. COMPLETENESS CHECK:
Are all major findings from the analytics covered in the insights?
List any significant analytical finding that was missed.

4. ACTIONABILITY CHECK:
Are the recommendations specific and actionable for a non-technical
SME owner? Flag any recommendation that is too vague or generic.

5. OVERALL QUALITY SCORE:
Rate the overall output quality on a scale of 1 to 10 for each dimension:
- Factual Accuracy: X/10
- Completeness: X/10
- Actionability: X/10
- Clarity: X/10
- Overall: X/10

Provide a one paragraph FINAL VERDICT summarising the reliability of
this output for a non-technical business owner.
"""

    print("\nRunning quality review...")
    response = llm.invoke(prompt)

    # Parse overall score from response
    output = {
        "agent": "Critic Agent",
        "status": "completed",
        "ground_truth_used": ground_truth,
        "review": response.content
    }

    print("\n--- AGENT 5 OUTPUT ---")
    print(response.content)

    with open("outputs/agent5_critic.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nOutput saved to outputs/agent5_critic.json")

    return output


if __name__ == "__main__":
    with open("outputs/agent2_analytics.json") as f:
        analytics_output = json.load(f)
    with open("outputs/agent3_insights.json") as f:
        insights_output = json.load(f)
    with open("outputs/agent4_recommendations.json") as f:
        recommendations_output = json.load(f)

    result = run_critic_agent(
        analytics_output,
        insights_output,
        recommendations_output
    )