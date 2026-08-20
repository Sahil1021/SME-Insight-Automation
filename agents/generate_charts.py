import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os

os.makedirs("outputs/charts", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

# ── Load retail data ───────────────────────────────────────
df = pd.read_csv("data/retail_sme_data.csv")
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b")

month_order = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Chart 1: Monthly Revenue Trend ────────────────────────
monthly = df.groupby("month_name")["revenue"].sum()
monthly = monthly.reindex(month_order)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(month_order, monthly.values, color="#3B3B8A",
        linewidth=2.5, marker="o", markersize=6, markerfacecolor="white",
        markeredgewidth=2)
ax.fill_between(month_order, monthly.values, alpha=0.08, color="#3B3B8A")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"£{x:,.0f}"))
ax.set_xlabel("Month (2024)", labelpad=8)
ax.set_ylabel("Total Revenue (£)", labelpad=8)
ax.set_title("Figure 5.1: Monthly Revenue Trend — Retail SME Scenario (2024)",
             fontsize=12, pad=14)
peak_idx = list(month_order).index("Apr")
ax.annotate(f"Peak: £{monthly['Apr']:,.0f}",
            xy=("Apr", monthly["Apr"]),
            xytext=("May", monthly["Apr"] * 1.06),
            arrowprops=dict(arrowstyle="->", color="#555"),
            fontsize=9, color="#555")
plt.tight_layout()
plt.savefig("outputs/charts/fig5_1_monthly_revenue.png", dpi=200, bbox_inches="tight")
plt.close()
print("Chart 1 saved.")

# ── Chart 2: Pareto Analysis ───────────────────────────────
product_rev = df.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
total = product_rev.sum()
pct = (product_rev / total * 100).round(2)
cum_pct = pct.cumsum()

fig, ax1 = plt.subplots(figsize=(12, 5))
bars = ax1.bar(range(len(product_rev)), pct.values,
               color=["#3B3B8A" if i < 4 else "#AAAACC"
                      for i in range(len(product_rev))],
               width=0.6, zorder=3)
ax1.set_ylabel("Revenue Share (%)", labelpad=8)
ax1.set_xticks(range(len(product_rev)))
ax1.set_xticklabels([p[:16] for p in product_rev.index],
                    rotation=35, ha="right", fontsize=9)
ax1.set_ylim(0, max(pct.values) * 1.2)

ax2 = ax1.twinx()
ax2.plot(range(len(cum_pct)), cum_pct.values,
         color="#C0392B", linewidth=2, marker="o",
         markersize=4, label="Cumulative %")
ax2.axhline(80, color="#C0392B", linestyle="--",
            linewidth=1, alpha=0.5)
ax2.set_ylabel("Cumulative Revenue (%)", labelpad=8)
ax2.set_ylim(0, 110)
ax2.spines["right"].set_visible(True)

ax1.set_title("Figure 5.2: Pareto Analysis — Product Revenue Share, Retail SME Scenario",
              fontsize=12, pad=14)
ax1.axvline(3.5, color="#555", linestyle=":", linewidth=1, alpha=0.6)
ax1.text(1.5, max(pct.values) * 1.12,
         "Top 20% of products\n(62.8% of revenue)",
         ha="center", fontsize=8.5, color="#3B3B8A")
plt.tight_layout()
plt.savefig("outputs/charts/fig5_2_pareto.png", dpi=200, bbox_inches="tight")
plt.close()
print("Chart 2 saved.")

# ── Chart 3: Category Performance ─────────────────────────
cat = df.groupby("category").agg(
    revenue=("revenue", "sum"),
    margin=("gross_margin_pct", "mean")
).sort_values("revenue", ascending=True)

fig, ax1 = plt.subplots(figsize=(9, 5))
colors = ["#3B3B8A", "#5B5BAA", "#7B7BCA", "#9B9BDA", "#BBBBEA"]
bars = ax1.barh(cat.index, cat["revenue"],
                color=colors, height=0.5, zorder=3)
ax1.set_xlabel("Total Revenue (£)", labelpad=8)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"£{x:,.0f}"))

ax2 = ax1.twiny()
ax2.plot(cat["margin"], cat.index, color="#C0392B",
         linewidth=2, marker="D", markersize=6,
         markerfacecolor="white", markeredgewidth=2,
         label="Gross Margin %")
ax2.set_xlabel("Gross Margin (%)", labelpad=8)
ax2.spines["top"].set_visible(True)
ax2.set_xlim(40, 75)

for i, (rev, margin) in enumerate(zip(cat["revenue"], cat["margin"])):
    ax1.text(rev + 2000, i, f"£{rev:,.0f}",
             va="center", fontsize=8.5, color="#333")
    ax2.text(margin + 0.5, i, f"{margin:.1f}%",
             va="center", fontsize=8.5, color="#C0392B")

ax1.set_title("Figure 5.3: Category Revenue and Gross Margin — Retail SME Scenario",
              fontsize=12, pad=14)
plt.tight_layout()
plt.savefig("outputs/charts/fig5_3_category.png", dpi=200, bbox_inches="tight")
plt.close()
print("Chart 3 saved.")

# ── Chart 4: Cross-Scenario Evaluation Scores ─────────────
dimensions = ["Factual\nAccuracy", "Completeness",
              "Actionability", "Clarity", "Overall"]
retail      = [10, 8, 8, 9, 8.5]
food        = [10, 8, 9, 9, 9.0]
distrib     = [10, 8, 8, 9, 8.5]

x = range(len(dimensions))
w = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([i - w for i in x], retail,  width=w, label="Retail SME",
       color="#3B3B8A", zorder=3)
ax.bar([i     for i in x], food,    width=w, label="Food Service SME",
       color="#1D9E75", zorder=3)
ax.bar([i + w for i in x], distrib, width=w, label="Distribution SME",
       color="#C0392B", zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels(dimensions, fontsize=10)
ax.set_ylabel("Score (out of 10)", labelpad=8)
ax.set_ylim(0, 11)
ax.axhline(10, color="#555", linestyle="--",
           linewidth=0.8, alpha=0.4)
ax.legend(frameon=False, fontsize=10)
ax.set_title("Figure 5.4: Cross-Scenario Evaluation Scores",
             fontsize=12, pad=14)
plt.tight_layout()
plt.savefig("outputs/charts/fig5_4_scores.png", dpi=200, bbox_inches="tight")
plt.close()
print("Chart 4 saved.")

print("\nAll charts saved to outputs/charts/")
print("Embed these in the dissertation as Figures 5.1 to 5.4")