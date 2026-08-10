import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatGroq(
    model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

def run_analytics_agent(filepath: str, data_understanding_output: dict) -> dict:
    """
    Agent 2: Analytics Agent
    Performs all numerical analysis using Python tools.
    LLM is only used for interpretation, never for computation.
    """

    print("\n" + "="*60)
    print("AGENT 2: ANALYTICS AGENT")
    print("="*60)

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")
    df["week"] = df["date"].dt.isocalendar().week

    analytics_results = {}

    # ── Analysis 1: Sales Trend (Monthly) ──────────────────
    print("\nRunning: Sales trend analysis...")
    monthly = df.groupby("month_name").agg(
        total_revenue=("revenue", "sum"),
        total_transactions=("transaction_id", "count"),
        avg_transaction_value=("revenue", "mean"),
        total_units=("quantity_sold", "sum")
    ).round(2)

    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    monthly = monthly.reindex([m for m in month_order if m in monthly.index])

    analytics_results["monthly_sales_trend"] = {
        "data": monthly.reset_index().to_dict(orient="records"),
        "peak_month": monthly["total_revenue"].idxmax(),
        "lowest_month": monthly["total_revenue"].idxmin(),
        "peak_revenue": round(monthly["total_revenue"].max(), 2),
        "lowest_revenue": round(monthly["total_revenue"].min(), 2),
        "revenue_growth_jan_to_peak": round(
            ((monthly["total_revenue"].max() - monthly["total_revenue"].iloc[0])
             / monthly["total_revenue"].iloc[0]) * 100, 1)
    }

    # ── Analysis 2: Pareto Analysis ─────────────────────────
    print("Running: Pareto analysis...")
    product_revenue = df.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
    total_rev = product_revenue.sum()
    product_revenue_pct = (product_revenue / total_rev * 100).round(2)
    cumulative_pct = product_revenue_pct.cumsum()

    pareto_df = pd.DataFrame({
        "product": product_revenue.index,
        "revenue": product_revenue.values.round(2),
        "revenue_pct": product_revenue_pct.values,
        "cumulative_pct": cumulative_pct.values.round(2)
    })

    top_20_pct_count = max(1, int(len(pareto_df) * 0.2))
    top_products = pareto_df.head(top_20_pct_count)

    analytics_results["pareto_analysis"] = {
        "data": pareto_df.to_dict(orient="records"),
        "top_20pct_products": top_products["product"].tolist(),
        "top_20pct_revenue_share": round(top_products["revenue_pct"].sum(), 1),
        "bottom_20pct_products": pareto_df.tail(top_20_pct_count)["product"].tolist(),
        "bottom_20pct_revenue_share": round(pareto_df.tail(top_20_pct_count)["revenue_pct"].sum(), 1),
        "total_products": len(pareto_df)
    }

    # ── Analysis 3: Category Performance ───────────────────
    print("Running: Category performance analysis...")
    category = df.groupby("category").agg(
        total_revenue=("revenue", "sum"),
        total_units=("quantity_sold", "sum"),
        avg_margin=("gross_margin_pct", "mean"),
        transaction_count=("transaction_id", "count")
    ).round(2).sort_values("total_revenue", ascending=False)

    category["revenue_share_pct"] = (
        category["total_revenue"] / category["total_revenue"].sum() * 100
    ).round(2)

    analytics_results["category_performance"] = {
        "data": category.reset_index().to_dict(orient="records"),
        "top_category": category["total_revenue"].idxmax(),
        "top_category_revenue_share": round(
            category["revenue_share_pct"].max(), 1),
        "lowest_margin_category": category["avg_margin"].idxmin(),
        "highest_margin_category": category["avg_margin"].idxmax()
    }

    # ── Analysis 4: Inventory Analysis ─────────────────────
    print("Running: Inventory analysis...")
    inventory = df.groupby("product_name").agg(
        avg_stock=("stock_level", "mean"),
        min_stock=("stock_level", "min"),
        reorder_point=("reorder_point", "first"),
        total_units_sold=("quantity_sold", "sum")
    ).round(2)

    inventory["stockout_risk"] = inventory["min_stock"] == 0
    inventory["stock_to_reorder_ratio"] = (
        inventory["avg_stock"] / inventory["reorder_point"]
    ).round(2)

    slow_movers = inventory[
        inventory["total_units_sold"] < inventory["total_units_sold"].quantile(0.25)
    ]

    analytics_results["inventory_analysis"] = {
        "data": inventory.reset_index().to_dict(orient="records"),
        "stockout_products": inventory[inventory["stockout_risk"]].index.tolist(),
        "slow_moving_products": slow_movers.index.tolist(),
        "avg_stock_to_reorder_ratio": round(
            inventory["stock_to_reorder_ratio"].mean(), 2)
    }

    # ── Analysis 5: Revenue and Margin KPIs ────────────────
    print("Running: KPI calculations...")
    analytics_results["kpis"] = {
        "total_revenue": round(df["revenue"].sum(), 2),
        "total_gross_profit": round(df["gross_profit"].sum(), 2),
        "overall_gross_margin_pct": round(df["gross_margin_pct"].mean(), 1),
        "total_transactions": len(df),
        "avg_transaction_value": round(df["revenue"].mean(), 2),
        "total_units_sold": int(df["quantity_sold"].sum()),
        "revenue_per_day": round(df["revenue"].sum() / 365, 2),
        "best_performing_region": df.groupby("region")["revenue"].sum().idxmax(),
        "best_region_revenue": round(
            df.groupby("region")["revenue"].sum().max(), 2),
        "top_supplier_by_revenue": df.groupby("supplier")["revenue"].sum().idxmax()
    }

    # ── Analysis 6: Regional Performance ───────────────────
    print("Running: Regional analysis...")
    regional = df.groupby("region").agg(
        total_revenue=("revenue", "sum"),
        total_transactions=("transaction_id", "count"),
        avg_margin=("gross_margin_pct", "mean")
    ).round(2).sort_values("total_revenue", ascending=False)

    regional["revenue_share_pct"] = (
        regional["total_revenue"] / regional["total_revenue"].sum() * 100
    ).round(2)

    analytics_results["regional_performance"] = {
        "data": regional.reset_index().to_dict(orient="records")
    }

    # ── LLM Interprets Results ─────────────────────────────
    print("\nLLM interpreting analytical results...")

    summary_for_llm = {
        "kpis": analytics_results["kpis"],
        "peak_month": analytics_results["monthly_sales_trend"]["peak_month"],
        "lowest_month": analytics_results["monthly_sales_trend"]["lowest_month"],
        "pareto_top_products": analytics_results["pareto_analysis"]["top_20pct_products"],
        "pareto_top_revenue_share": analytics_results["pareto_analysis"]["top_20pct_revenue_share"],
        "top_category": analytics_results["category_performance"]["top_category"],
        "stockout_products": analytics_results["inventory_analysis"]["stockout_products"],
        "slow_moving_products": analytics_results["inventory_analysis"]["slow_moving_products"],
        "best_region": analytics_results["kpis"]["best_performing_region"]
    }

    prompt = f"""
You are a business analyst interpreting analytical results for an SME retail business.
All numbers below were computed using Python. Do not change or question them.
Interpret what these results mean for the business owner.

ANALYTICAL RESULTS SUMMARY:
{json.dumps(summary_for_llm, indent=2)}

Provide a structured analytical interpretation covering:

SALES PERFORMANCE:
[Interpret revenue, transactions, and seasonal trends]

PRODUCT INSIGHTS:
[Interpret the Pareto findings and what they mean]

INVENTORY FINDINGS:
[Interpret stockout risks and slow movers]

REGIONAL INSIGHTS:
[Interpret regional performance]

Keep each section to 3-4 sentences. Be specific with numbers. Write for a non-technical business owner.
"""

    response = llm.invoke(prompt)
    analytics_results["llm_interpretation"] = response.content

    print("\n--- AGENT 2 OUTPUT ---")
    print(response.content)

    output = {
        "agent": "Analytics Agent",
        "status": "completed",
        "analytics": analytics_results
    }

    with open("outputs/agent2_analytics.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nOutput saved to outputs/agent2_analytics.json")

    return output


if __name__ == "__main__":
    with open("outputs/agent1_data_understanding.json") as f:
        agent1_output = json.load(f)

    result = run_analytics_agent("data/retail_sme_data.csv", agent1_output)