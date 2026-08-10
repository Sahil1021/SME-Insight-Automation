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

def run_insight_generation_agent(analytics_output: dict) -> dict:
    """
    Agent 3: Insight Generation Agent
    Converts analytical outputs into clear, actionable business
    insights written for a non-technical SME owner.
    """

    print("\n" + "="*60)
    print("AGENT 3: INSIGHT GENERATION AGENT")
    print("="*60)

    analytics = analytics_output["analytics"]

    # Extract key findings computed by Agent 2
    kpis = analytics["kpis"]
    pareto = analytics["pareto_analysis"]
    monthly = analytics["monthly_sales_trend"]
    category = analytics["category_performance"]
    inventory = analytics["inventory_analysis"]
    regional = analytics["regional_performance"]

    prompt = f"""
You are a business analyst writing a professional insight report for the owner
of a small retail paint and hardware business in India. The owner is not a
data expert. Write clearly, specifically, and practically.

All numbers below were computed by a Python analytics engine. Use them exactly
as provided. Do not invent or estimate any figures.

KEY PERFORMANCE DATA:
- Total Annual Revenue: £{kpis['total_revenue']:,}
- Total Gross Profit: £{kpis['total_gross_profit']:,}
- Overall Gross Margin: {kpis['overall_gross_margin_pct']}%
- Total Transactions: {kpis['total_transactions']:,}
- Average Transaction Value: £{kpis['avg_transaction_value']}
- Daily Revenue Average: £{kpis['revenue_per_day']}
- Total Units Sold: {kpis['total_units_sold']:,}

SALES TREND:
- Peak Month: {monthly['peak_month']} (Revenue: £{monthly['peak_revenue']:,})
- Lowest Month: {monthly['lowest_month']} (Revenue: £{monthly['lowest_revenue']:,})
- Revenue growth from January to peak: {monthly['revenue_growth_jan_to_peak']}%

PARETO ANALYSIS:
- Top 20% of products: {pareto['top_20pct_products']}
- These products generate: {pareto['top_20pct_revenue_share']}% of total revenue
- Bottom 20% of products: {pareto['bottom_20pct_products']}
- Bottom products generate: {pareto['bottom_20pct_revenue_share']}% of total revenue

CATEGORY PERFORMANCE:
- Top Category: {category['top_category']}
- Highest Margin Category: {category['highest_margin_category']}
- Lowest Margin Category: {category['lowest_margin_category']}

INVENTORY:
- Products with stockout risk: {inventory['stockout_products'] if inventory['stockout_products'] else 'None identified'}
- Slow-moving products: {inventory['slow_moving_products']}

REGIONAL PERFORMANCE:
- Best Region: {kpis['best_performing_region']} (£{kpis['best_region_revenue']:,})
- Top Supplier: {kpis['top_supplier_by_revenue']}

Generate exactly 8 business insights numbered 1 to 8.
Each insight must:
- Start with a bold one-line headline summarising the insight
- Follow with 2-3 sentences explaining what it means and why it matters
- Be written in plain English for a non-technical business owner
- Use the exact figures provided above
- Use £ for currency
- Be specific and actionable, not generic

Do not use phrases like "it is recommended" or "the data suggests".
State insights directly and confidently.
"""

    print("\nGenerating business insights...")
    response = llm.invoke(prompt)

    output = {
        "agent": "Insight Generation Agent",
        "status": "completed",
        "insights": response.content
    }

    print("\n--- AGENT 3 OUTPUT ---")
    print(response.content)

    with open("outputs/agent3_insights.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nOutput saved to outputs/agent3_insights.json")

    return output


if __name__ == "__main__":
    with open("outputs/agent2_analytics.json") as f:
        analytics_output = json.load(f)

    result = run_insight_generation_agent(analytics_output)