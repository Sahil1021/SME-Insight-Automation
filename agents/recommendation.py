from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

def run_recommendation_agent(analytics_output: dict, insights_output: dict) -> dict:
    """
    Agent 4: Recommendation Agent
    Produces prioritised, actionable recommendations grounded
    in the analytical findings and insights.
    """

    print("\n" + "="*60)
    print("AGENT 4: RECOMMENDATION AGENT")
    print("="*60)

    analytics = analytics_output["analytics"]
    insights = insights_output["insights"]
    kpis = analytics["kpis"]
    pareto = analytics["pareto_analysis"]
    inventory = analytics["inventory_analysis"]
    monthly = analytics["monthly_sales_trend"]

    prompt = f"""
You are a business consultant writing prioritised recommendations
for the owner of a small retail paint and hardware business.

The recommendations must be based strictly on the insights and
analytics below. Do not invent new data. Use £ for currency.

BUSINESS INSIGHTS GENERATED:
{insights}

KEY ANALYTICS:
- Total Revenue: £{kpis['total_revenue']:,}
- Gross Margin: {kpis['overall_gross_margin_pct']}%
- Peak Month: {monthly['peak_month']}
- Lowest Month: {monthly['lowest_month']}
- Top Products (20%) driving {pareto['top_20pct_revenue_share']}% revenue: {pareto['top_20pct_products']}
- Slow Moving Products: {inventory['slow_moving_products']}
- Best Region: {kpis['best_performing_region']}
- Top Supplier: {kpis['top_supplier_by_revenue']}

Generate exactly 6 prioritised recommendations.
Number them 1 to 6 from highest to lowest priority.

Each recommendation must follow this structure:
PRIORITY [N] - [BOLD TITLE]
Action: [Specific action the owner should take]
Rationale: [Why this is important, referencing specific data]
Expected Impact: [What improvement this could drive]
Timeline: [When to implement: Immediate / Short-term (1-3 months) / Medium-term (3-6 months)]

Write for a non-technical business owner.
Be specific, direct, and practical.
Do not use generic business jargon.
"""

    print("\nGenerating recommendations...")
    response = llm.invoke(prompt)

    output = {
        "agent": "Recommendation Agent",
        "status": "completed",
        "recommendations": response.content
    }

    print("\n--- AGENT 4 OUTPUT ---")
    print(response.content)

    with open("outputs/agent4_recommendations.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nOutput saved to outputs/agent4_recommendations.json")

    return output


if __name__ == "__main__":
    with open("outputs/agent2_analytics.json") as f:
        analytics_output = json.load(f)
    with open("outputs/agent3_insights.json") as f:
        insights_output = json.load(f)

    result = run_recommendation_agent(analytics_output, insights_output)