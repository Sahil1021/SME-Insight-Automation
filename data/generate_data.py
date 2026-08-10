import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ── Configuration ──────────────────────────────────────────
START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "retail_sme_data.csv")

# Product catalogue — Pareto distribution built in
# Top 5 products (25%) will drive ~75% of revenue
PRODUCTS = [
    {"id": "P001", "name": "Premium Paint 20L",     "category": "Paint",       "unit_price": 85.00, "cost_price": 42.00, "reorder_point": 20, "supplier": "ColorCo"},
    {"id": "P002", "name": "Standard Paint 10L",    "category": "Paint",       "unit_price": 45.00, "cost_price": 22.00, "reorder_point": 30, "supplier": "ColorCo"},
    {"id": "P003", "name": "Wood Primer 5L",         "category": "Primer",      "unit_price": 32.00, "cost_price": 15.00, "reorder_point": 25, "supplier": "PrimePro"},
    {"id": "P004", "name": "Wall Filler 2kg",        "category": "Accessories", "unit_price": 18.00, "cost_price": 8.00,  "reorder_point": 40, "supplier": "FixIt"},
    {"id": "P005", "name": "Gloss Paint 5L",         "category": "Paint",       "unit_price": 38.00, "cost_price": 18.00, "reorder_point": 25, "supplier": "ColorCo"},
    {"id": "P006", "name": "Paint Brush Set",        "category": "Tools",       "unit_price": 22.00, "cost_price": 10.00, "reorder_point": 50, "supplier": "ToolMart"},
    {"id": "P007", "name": "Paint Roller Kit",       "category": "Tools",       "unit_price": 28.00, "cost_price": 13.00, "reorder_point": 40, "supplier": "ToolMart"},
    {"id": "P008", "name": "Masking Tape 50m",       "category": "Accessories", "unit_price": 8.00,  "cost_price": 3.00,  "reorder_point": 80, "supplier": "FixIt"},
    {"id": "P009", "name": "Anti-Damp Paint 5L",     "category": "Specialist",  "unit_price": 55.00, "cost_price": 28.00, "reorder_point": 15, "supplier": "PrimePro"},
    {"id": "P010", "name": "Exterior Paint 10L",     "category": "Paint",       "unit_price": 62.00, "cost_price": 31.00, "reorder_point": 20, "supplier": "ColorCo"},
    {"id": "P011", "name": "Sandpaper Pack",         "category": "Accessories", "unit_price": 6.00,  "cost_price": 2.50,  "reorder_point": 100,"supplier": "FixIt"},
    {"id": "P012", "name": "Paint Stripper 1L",      "category": "Specialist",  "unit_price": 24.00, "cost_price": 11.00, "reorder_point": 20, "supplier": "PrimePro"},
    {"id": "P013", "name": "Ceiling Paint 10L",      "category": "Paint",       "unit_price": 48.00, "cost_price": 23.00, "reorder_point": 20, "supplier": "ColorCo"},
    {"id": "P014", "name": "Paint Tray Set",         "category": "Tools",       "unit_price": 12.00, "cost_price": 5.00,  "reorder_point": 60, "supplier": "ToolMart"},
    {"id": "P015", "name": "Tile Grout 5kg",         "category": "Specialist",  "unit_price": 35.00, "cost_price": 16.00, "reorder_point": 15, "supplier": "PrimePro"},
    {"id": "P016", "name": "Undercoat 5L",           "category": "Primer",      "unit_price": 30.00, "cost_price": 14.00, "reorder_point": 25, "supplier": "PrimePro"},
    {"id": "P017", "name": "Varnish 2.5L",           "category": "Specialist",  "unit_price": 28.00, "cost_price": 13.00, "reorder_point": 20, "supplier": "PrimePro"},
    {"id": "P018", "name": "Paint Thinner 1L",       "category": "Accessories", "unit_price": 10.00, "cost_price": 4.00,  "reorder_point": 40, "supplier": "FixIt"},
    {"id": "P019", "name": "Texture Paint 5L",       "category": "Specialist",  "unit_price": 42.00, "cost_price": 20.00, "reorder_point": 15, "supplier": "ColorCo"},
    {"id": "P020", "name": "Drop Cloth 3x4m",        "category": "Accessories", "unit_price": 15.00, "cost_price": 6.00,  "reorder_point": 30, "supplier": "FixIt"},
]

# Pareto weights — top 5 get ~75% of sales volume
WEIGHTS = [0.18, 0.16, 0.12, 0.10, 0.09,
           0.06, 0.06, 0.05, 0.04, 0.04,
           0.02, 0.02, 0.02, 0.01, 0.01,
           0.01, 0.01, 0.00, 0.00, 0.00]

REGIONS = ["North", "South", "East", "West", "Central"]

# ── Helper: seasonal multiplier ────────────────────────────
def seasonal_multiplier(date):
    month = date.month
    # Spring/summer peak (April-August), winter dip (Nov-Feb)
    if month in [4, 5, 6, 7, 8]:
        return round(random.uniform(1.3, 1.6), 2)
    elif month in [11, 12, 1, 2]:
        return round(random.uniform(0.6, 0.85), 2)
    else:
        return round(random.uniform(0.95, 1.15), 2)

# ── Generate transactions ──────────────────────────────────
records = []
transaction_id = 1000
stock_levels = {p["id"]: random.randint(50, 150) for p in PRODUCTS}

date = START_DATE
while date <= END_DATE:
    # 8-18 transactions per day depending on season
    seasonal = seasonal_multiplier(date)
    daily_transactions = int(random.uniform(8, 18) * seasonal)

    for _ in range(daily_transactions):
        product = random.choices(PRODUCTS, weights=WEIGHTS, k=1)[0]
        qty = random.randint(1, 6)
        region = random.choice(REGIONS)

        # Occasional stockout — stock drops to 0
        current_stock = stock_levels[product["id"]]
        if current_stock < qty:
            qty = max(1, current_stock)
            stock_levels[product["id"]] = 0
        else:
            stock_levels[product["id"]] -= qty

        # Restock trigger
        if stock_levels[product["id"]] <= product["reorder_point"]:
            stock_levels[product["id"]] += random.randint(40, 100)

        revenue        = round(product["unit_price"] * qty, 2)
        cost           = round(product["cost_price"] * qty, 2)
        gross_profit   = round(revenue - cost, 2)
        gross_margin   = round((gross_profit / revenue) * 100, 1) if revenue > 0 else 0

        records.append({
            "transaction_id":   transaction_id,
            "date":             date.strftime("%Y-%m-%d"),
            "product_id":       product["id"],
            "product_name":     product["name"],
            "category":         product["category"],
            "quantity_sold":    qty,
            "unit_price":       product["unit_price"],
            "cost_price":       product["cost_price"],
            "revenue":          revenue,
            "gross_profit":     gross_profit,
            "gross_margin_pct": gross_margin,
            "stock_level":      stock_levels[product["id"]],
            "reorder_point":    product["reorder_point"],
            "supplier":         product["supplier"],
            "region":           region,
        })
        transaction_id += 1

    date += timedelta(days=1)

# ── Save ───────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset generated successfully.")
print(f"Total transactions : {len(df):,}")
print(f"Date range         : {df['date'].min()} to {df['date'].max()}")
print(f"Total revenue      : £{df['revenue'].sum():,.2f}")
print(f"Products           : {df['product_name'].nunique()}")
print(f"Saved to           : {OUTPUT_PATH}")