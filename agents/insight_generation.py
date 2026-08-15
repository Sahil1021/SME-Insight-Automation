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
    Covers all six analytical dimensions equally.
    """

    print("\n" + "="*60)
    print("AGENT 3: INSIGHT GENERATION AGENT")
    print("="*60)

    analytics = analytics_output["analytics"]

    kpis = analytics["kpis"]
    pareto = analytics["pareto_analysis"]
    monthly = analytics["monthly_sales_trend"]
    category = analytics["category_performance"]
    inventory = analytics["inventory_analysis"]
    regional = analytics["regional_performance"]

    # Build regional data string
    regional_data = regional["data"][:3] if regional["data"] else []
    regional_str = json.dumps(regional_data, indent=2)

    # Build category data string
    category_data = category["data"][:5] if category["data"] else []
    category_str = json.dumps(category_data, indent=2)

    prompt = f"""
You are a business analyst writing a professional insight report for the owner
of a small business. The owner is not a data expert.
Write clearly, specifically, and practically.

All numbers below were computed by a Python analytics engine.
Use them exactly as provided. Do not invent or estimate any figures.
Always use £ for currency.

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

PRODUCT REVENUE DISTRIBUTION:
- Top 20% of products: {pareto['top_20pct_products']}
- These products generate: {pareto['top_20pct_revenue_share']}% of total revenue
- Bottom 20% of products: {pareto['bottom_20pct_products']}
- Bottom products generate: {pareto['bottom_20pct_revenue_share']}% of total revenue
- Total products: {pareto['total_products']}

CATEGORY PERFORMANCE:
- Top Revenue Category: {category['top_category']}
- Highest Margin Category: {category['highest_margin_category']}
- Lowest Margin Category: {category['lowest_margin_category']}
- Category breakdown:
{category_str}

INVENTORY:
- Products with stockout risk: {inventory['stockout_products'] if inventory['stockout_products'] else 'None identified'}
- Slow-moving products: {inventory['slow_moving_products']}
- Average stock to reorder ratio: {inventory['avg_stock_to_reorder_ratio']}

REGIONAL PERFORMANCE:
- Best Region: {kpis['best_performing_region']} (£{kpis['best_region_revenue']:,})
- Top Supplier: {kpis['top_supplier_by_revenue']}
- Regional breakdown (top 3):
{regional_str}

Generate exactly 8 business insights numbered 1 to 8.
Cover all six analytical dimensions across the 8 insights:
- Sales performance and seasonal trends (at least 1 insight)
- Revenue KPIs and overall profitability (at least 1 insight)
- Product revenue distribution and concentration (1 insight)
- Category performance and margin differences (1 insight)
- Inventory management findings including slow movers (1 insight)
- Regional performance and supplier relationships (1 insight)
- A forward-looking business implication drawn from the data (1 insight)
- One additional insight on whichever dimension has the most interesting finding

Each insight must:
- Start with a bold one-line headline summarising the insight
- Follow with 2-3 sentences explaining what it means and why it matters
- Be written in plain English for a non-technical business owner
- Use the exact figures provided above
- Use £ for all currency values
- Be specific and actionable, not generic

Do not use phrases like "it is recommended" or "the data suggests".
State insights directly and confidently.
Do not give disproportionate attention to any single analysis.
Give equal weight to sales trends, margins, inventory, categories, and regional data.
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