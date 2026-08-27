import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
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

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})

BLUE       = "#3B3B8A"
RED        = "#C0392B"
GREEN      = "#1D9E75"
LIGHT_BLUE = "#AAAACC"


# ── VISUALISATION FUNCTIONS ────────────────────────────────

def _chart1_monthly(df, monthly, scenario_name, charts_dir):
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly_rev = df.groupby(
        df["date"].dt.strftime("%b"))["revenue"].sum()
    monthly_rev = monthly_rev.reindex(month_order)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(month_order, monthly_rev.values, color=BLUE,
            linewidth=2.5, marker="o", markersize=7,
            markerfacecolor="white", markeredgewidth=2.5)
    ax.fill_between(month_order, monthly_rev.values,
                    alpha=0.08, color=BLUE)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.set_xlabel("Month", labelpad=8)
    ax.set_ylabel("Total Revenue (£)", labelpad=8)
    ax.set_title(f"Monthly Revenue Trend — {scenario_name}",
                 fontsize=13, fontweight="bold", pad=14)
    peak = monthly["peak_month"]
    peak_val = monthly["peak_revenue"]
    peak_short = peak[:3]
    if peak_short in month_order and peak_short in monthly_rev.index:
        pidx = month_order.index(peak_short)
        offset = min(pidx + 1.5, 10)
        ax.annotate(f"Peak: £{peak_val:,.0f}\n({peak})",
                    xy=(peak_short, monthly_rev[peak_short]),
                    xytext=(month_order[int(offset)],
                            monthly_rev[peak_short] * 1.05),
                    arrowprops=dict(arrowstyle="->", color="#555"),
                    fontsize=9, color="#333")
    plt.tight_layout()
    path = os.path.join(charts_dir, "chart1_monthly_trend.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _chart2_pareto(df, pareto, scenario_name, charts_dir):
    product_rev = df.groupby("product_name")["revenue"].sum()\
                    .sort_values(ascending=False)
    total = product_rev.sum()
    pct   = (product_rev / total * 100).round(2)
    cum   = pct.cumsum()
    top_n = max(1, int(len(product_rev) * 0.2))
    colors = [BLUE if i < top_n else LIGHT_BLUE
              for i in range(len(product_rev))]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(range(len(pct)), pct.values, color=colors,
            width=0.65, zorder=3)
    ax1.set_ylabel("Revenue Share (%)", labelpad=8)
    ax1.set_xticks(range(len(pct)))
    ax1.set_xticklabels([p[:14] for p in pct.index],
                        rotation=38, ha="right", fontsize=8.5)
    ax1.set_ylim(0, max(pct.values) * 1.3)

    ax2 = ax1.twinx()
    ax2.plot(range(len(cum)), cum.values, color=RED,
             linewidth=2, marker="o", markersize=4)
    ax2.axhline(80, color=RED, linestyle="--",
                linewidth=1, alpha=0.4)
    ax2.set_ylabel("Cumulative Revenue (%)", labelpad=8)
    ax2.set_ylim(0, 115)
    ax2.spines["right"].set_visible(True)

    ax1.axvline(top_n - 0.5, color="#555",
                linestyle=":", linewidth=1, alpha=0.5)
    top_share = pareto["top_20pct_revenue_share"]
    ax1.text(top_n / 2 - 0.5, max(pct.values) * 1.18,
             f"Top 20% products\n({top_share}% of revenue)",
             ha="center", fontsize=9, color=BLUE)
    ax1.set_title(
        f"Pareto Analysis: Product Revenue Distribution — {scenario_name}",
        fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    path = os.path.join(charts_dir, "chart2_pareto.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _chart3_category(df, scenario_name, charts_dir):
    cat_df = df.groupby("category").agg(
        revenue=("revenue", "sum"),
        margin=("gross_margin_pct", "mean")
    ).sort_values("revenue", ascending=True)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    shades = [BLUE if i == len(cat_df) - 1
              else LIGHT_BLUE for i in range(len(cat_df))]
    ax1.barh(cat_df.index, cat_df["revenue"],
             color=shades, height=0.55, zorder=3)
    ax1.set_xlabel("Total Revenue (£)", labelpad=8)
    ax1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    for i, rev in enumerate(cat_df["revenue"]):
        ax1.text(rev * 0.02, i, f"  £{rev:,.0f}",
                 va="center", fontsize=8.5, color="#333")

    ax2 = ax1.twiny()
    ax2.plot(cat_df["margin"], cat_df.index, color=RED,
             linewidth=2, marker="D", markersize=7,
             markerfacecolor="white", markeredgewidth=2)
    ax2.set_xlabel("Gross Margin (%)", labelpad=8, color=RED)
    ax2.tick_params(axis="x", colors=RED)
    ax2.spines["top"].set_edgecolor(RED)
    margin_range = max(cat_df["margin"]) - min(cat_df["margin"])
    ax2.set_xlim(min(cat_df["margin"]) - margin_range * 0.5,
                 max(cat_df["margin"]) + margin_range * 0.5)

    ax1.set_title(
        f"Category Revenue and Gross Margin — {scenario_name}",
        fontsize=13, fontweight="bold", pad=28)
    plt.tight_layout()
    path = os.path.join(charts_dir, "chart3_category.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _chart4_regional(regional, scenario_name, charts_dir):
    reg_df = pd.DataFrame(regional["data"])\
               .sort_values("total_revenue", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_reg = [BLUE if i == len(reg_df) - 1
                  else LIGHT_BLUE for i in range(len(reg_df))]
    ax.barh(reg_df["region"], reg_df["total_revenue"],
            color=colors_reg, height=0.55, zorder=3)
    ax.set_xlabel("Total Revenue (£)", labelpad=8)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    for i, rev in enumerate(reg_df["total_revenue"]):
        ax.text(rev * 0.02, i, f"  £{rev:,.0f}",
                va="center", fontsize=9, color="#333")
    ax.set_title(
        f"Regional Revenue Performance — {scenario_name}",
        fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    path = os.path.join(charts_dir, "chart4_regional.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _chart5_inventory(df, inventory, scenario_name, charts_dir):
    inv_df = pd.DataFrame(inventory["data"])\
               .sort_values("total_units_sold", ascending=True)
    slow = set(inventory["slow_moving_products"])

    fig, ax = plt.subplots(figsize=(11, 6))
    bar_colors = [RED if p in slow else GREEN
                  for p in inv_df["product_name"]]
    ax.barh(inv_df["product_name"], inv_df["total_units_sold"],
            color=bar_colors, height=0.6, zorder=3)
    ax.set_xlabel("Total Units Sold", labelpad=8)
    ax.set_title(
        f"Inventory Performance: Units Sold by Product — {scenario_name}",
        fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(axis="y", labelsize=8.5)
    legend = [Patch(color=RED, label="Slow-moving product"),
              Patch(color=GREEN, label="Normal performer")]
    ax.legend(handles=legend, frameon=False, fontsize=9,
              loc="lower right")
    plt.tight_layout()
    path = os.path.join(charts_dir, "chart5_inventory.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _chart6_kpi(kpis, scenario_name, charts_dir):
    kpi_items = [
        ("Total Revenue",    f"£{kpis['total_revenue']:,.0f}"),
        ("Gross Profit",     f"£{kpis['total_gross_profit']:,.0f}"),
        ("Gross Margin",     f"{kpis['overall_gross_margin_pct']}%"),
        ("Transactions",     f"{kpis['total_transactions']:,}"),
        ("Avg Transaction",  f"£{kpis['avg_transaction_value']:,.2f}"),
        ("Daily Revenue",    f"£{kpis['revenue_per_day']:,.2f}"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(14, 3))
    fig.patch.set_facecolor("white")
    for ax, (label, value) in zip(axes, kpi_items):
        ax.set_facecolor("#F4F4F9")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.62, value, ha="center", va="center",
                fontsize=13, fontweight="bold", color=BLUE,
                transform=ax.transAxes)
        ax.text(0.5, 0.28, label, ha="center", va="center",
                fontsize=8.5, color="#555",
                transform=ax.transAxes)
        for spine in ["top","bottom","left","right"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#CCCCDD")
            ax.spines[spine].set_linewidth(1.2)

    fig.suptitle(f"Key Performance Indicators — {scenario_name}",
                 fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    path = os.path.join(charts_dir, "chart6_kpi_dashboard.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def generate_visualisations(filepath, analytics_results,
                             output_dir, scenario_name):
    """Generate all 6 analytical charts and save to output directory."""
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])

    kpis      = analytics_results["kpis"]
    pareto    = analytics_results["pareto_analysis"]
    monthly   = analytics_results["monthly_sales_trend"]
    inventory = analytics_results["inventory_analysis"]
    regional  = analytics_results["regional_performance"]

    saved = []
    saved.append(_chart1_monthly(df, monthly, scenario_name, charts_dir))
    saved.append(_chart2_pareto(df, pareto, scenario_name, charts_dir))
    saved.append(_chart3_category(df, scenario_name, charts_dir))
    saved.append(_chart4_regional(regional, scenario_name, charts_dir))
    saved.append(_chart5_inventory(df, inventory, scenario_name, charts_dir))
    saved.append(_chart6_kpi(kpis, scenario_name, charts_dir))

    print(f"\n   6 charts saved to: {charts_dir}")
    return saved


# ── ANALYTICS AGENT ───────────────────────────────────────

def run_analytics_agent(filepath: str,
                        data_understanding_output: dict,
                        output_dir: str = "outputs",
                        scenario_name: str = "SME") -> dict:
    """
    Agent 2: Analytics Agent
    Performs all numerical analysis using Python tools.
    LLM is only used for interpretation, never for computation.
    Also generates 6 visual charts saved to the output directory.
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

    # ── Analysis 1: Sales Trend ────────────────────────────
    print("\nRunning: Sales trend analysis...")
    monthly = df.groupby("month_name").agg(
        total_revenue=("revenue", "sum"),
        total_transactions=("transaction_id", "count"),
        avg_transaction_value=("revenue", "mean"),
        total_units=("quantity_sold", "sum")
    ).round(2)

    month_order = ["January","February","March","April","May",
                   "June","July","August","September","October",
                   "November","December"]
    monthly = monthly.reindex(
        [m for m in month_order if m in monthly.index])

    jan_revenue  = monthly["total_revenue"].iloc[0]
    peak_revenue = monthly["total_revenue"].max()
    growth = round(
        ((peak_revenue - jan_revenue) / jan_revenue) * 100, 1)

    analytics_results["monthly_sales_trend"] = {
        "data": monthly.reset_index().to_dict(orient="records"),
        "peak_month":   monthly["total_revenue"].idxmax(),
        "lowest_month": monthly["total_revenue"].idxmin(),
        "peak_revenue":   round(peak_revenue, 2),
        "lowest_revenue": round(monthly["total_revenue"].min(), 2),
        "revenue_growth_jan_to_peak": growth
    }

    # ── Analysis 2: Pareto ─────────────────────────────────
    print("Running: Pareto analysis...")
    product_revenue = df.groupby("product_name")["revenue"]\
                        .sum().sort_values(ascending=False)
    total_rev = product_revenue.sum()
    pct = (product_revenue / total_rev * 100).round(2)
    cum = pct.cumsum()

    pareto_df = pd.DataFrame({
        "product":        product_revenue.index,
        "revenue":        product_revenue.values.round(2),
        "revenue_pct":    pct.values,
        "cumulative_pct": cum.values.round(2)
    })

    top_n = max(1, int(len(pareto_df) * 0.2))
    top_p = pareto_df.head(top_n)
    bot_p = pareto_df.tail(top_n)

    analytics_results["pareto_analysis"] = {
        "data":                    pareto_df.to_dict(orient="records"),
        "top_20pct_products":      top_p["product"].tolist(),
        "top_20pct_revenue_share": round(top_p["revenue_pct"].sum(), 1),
        "bottom_20pct_products":   bot_p["product"].tolist(),
        "bottom_20pct_revenue_share": round(
            bot_p["revenue_pct"].sum(), 1),
        "total_products": len(pareto_df)
    }

    # ── Analysis 3: Category Performance ──────────────────
    print("Running: Category performance analysis...")
    category = df.groupby("category").agg(
        total_revenue=("revenue", "sum"),
        total_units=("quantity_sold", "sum"),
        avg_margin=("gross_margin_pct", "mean"),
        transaction_count=("transaction_id", "count")
    ).round(2).sort_values("total_revenue", ascending=False)

    category["revenue_share_pct"] = (
        category["total_revenue"] /
        category["total_revenue"].sum() * 100
    ).round(2)

    analytics_results["category_performance"] = {
        "data": category.reset_index().to_dict(orient="records"),
        "top_category":
            category["total_revenue"].idxmax(),
        "top_category_revenue_share":
            round(category["revenue_share_pct"].max(), 1),
        "lowest_margin_category":
            category["avg_margin"].idxmin(),
        "highest_margin_category":
            category["avg_margin"].idxmax()
    }

    # ── Analysis 4: Inventory ──────────────────────────────
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

    threshold = inventory["total_units_sold"].quantile(0.25)
    slow_movers = inventory[inventory["total_units_sold"] < threshold]

    analytics_results["inventory_analysis"] = {
        "data": inventory.reset_index().to_dict(orient="records"),
        "stockout_products":
            inventory[inventory["stockout_risk"]].index.tolist(),
        "slow_moving_products": slow_movers.index.tolist(),
        "avg_stock_to_reorder_ratio":
            round(inventory["stock_to_reorder_ratio"].mean(), 2)
    }

    # ── Analysis 5: KPIs ──────────────────────────────────
    print("Running: KPI calculations...")
    region_revenue = df.groupby("region")["revenue"].sum()

    analytics_results["kpis"] = {
        "total_revenue":
            round(df["revenue"].sum(), 2),
        "total_gross_profit":
            round(df["gross_profit"].sum(), 2),
        "overall_gross_margin_pct":
            round(df["gross_margin_pct"].mean(), 1),
        "total_transactions": len(df),
        "avg_transaction_value":
            round(df["revenue"].mean(), 2),
        "total_units_sold":
            int(df["quantity_sold"].sum()),
        "revenue_per_day":
            round(df["revenue"].sum() / 365, 2),
        "best_performing_region":
            region_revenue.idxmax(),
        "best_region_revenue":
            round(region_revenue.max(), 2),
        "top_supplier_by_revenue":
            df.groupby("supplier")["revenue"].sum().idxmax()
    }

    # ── Analysis 6: Regional ──────────────────────────────
    print("Running: Regional analysis...")
    regional = df.groupby("region").agg(
        total_revenue=("revenue", "sum"),
        total_transactions=("transaction_id", "count"),
        avg_margin=("gross_margin_pct", "mean")
    ).round(2).sort_values("total_revenue", ascending=False)

    regional["revenue_share_pct"] = (
        regional["total_revenue"] /
        regional["total_revenue"].sum() * 100
    ).round(2)

    analytics_results["regional_performance"] = {
        "data": regional.reset_index().to_dict(orient="records")
    }

    # ── Generate Visualisations ────────────────────────────
    print("\nGenerating analytical visualisations...")
    chart_paths = generate_visualisations(
        filepath, analytics_results, output_dir, scenario_name)
    analytics_results["chart_paths"] = chart_paths

    # ── LLM Interprets Results ─────────────────────────────
    print("\nLLM interpreting analytical results...")
    kpis     = analytics_results["kpis"]
    monthly_r = analytics_results["monthly_sales_trend"]
    pareto_r  = analytics_results["pareto_analysis"]
    cat_r     = analytics_results["category_performance"]
    inv_r     = analytics_results["inventory_analysis"]

    summary = {
        "kpis": kpis,
        "peak_month":   monthly_r["peak_month"],
        "lowest_month": monthly_r["lowest_month"],
        "peak_revenue": monthly_r["peak_revenue"],
        "pareto_top_products":      pareto_r["top_20pct_products"],
        "pareto_top_revenue_share": pareto_r["top_20pct_revenue_share"],
        "top_category":             cat_r["top_category"],
        "highest_margin_category":  cat_r["highest_margin_category"],
        "lowest_margin_category":   cat_r["lowest_margin_category"],
        "stockout_products":        inv_r["stockout_products"],
        "slow_moving_products":     inv_r["slow_moving_products"],
        "best_region":  kpis["best_performing_region"],
        "best_region_revenue": kpis["best_region_revenue"],
        "top_supplier": kpis["top_supplier_by_revenue"]
    }

    prompt = f"""
You are a business analyst interpreting analytical results for an SME.
All numbers below were computed using Python. Do not change or question them.
Always use £ (British Pound) for all currency values.

ANALYTICAL RESULTS SUMMARY:
{json.dumps(summary, indent=2)}

Provide a structured analytical interpretation covering:

SALES PERFORMANCE:
[Interpret revenue, transactions, and seasonal trends using exact figures]

PRODUCT INSIGHTS:
[Interpret the product revenue distribution findings]

INVENTORY FINDINGS:
[Interpret stockout risks and slow movers]

REGIONAL INSIGHTS:
[Interpret regional performance using exact figures]

Keep each section to 3-4 sentences. Be specific with numbers.
Write for a non-technical business owner. Use £ for all currency.
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

    out_path = os.path.join(output_dir, "agent2_analytics.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nOutput saved to {out_path}")

    return output


if __name__ == "__main__":
    with open("outputs/retail_sme/agent1_data_understanding.json") as f:
        agent1_output = json.load(f)
    run_analytics_agent(
        "data/retail_sme_data.csv",
        agent1_output,
        "outputs/retail_sme",
        "Retail SME"
    )