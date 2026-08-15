import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(456)
random.seed(456)

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "distribution_data.csv")

PRODUCTS = [
    {"id": "D001", "name": "Industrial Lubricant 20L",   "category": "Lubricants",  "unit_price": 120.00, "cost_price": 65.00, "reorder_point": 15, "supplier": "ChemTech"},
    {"id": "D002", "name": "Safety Gloves (Box 100)",    "category": "Safety",      "unit_price": 45.00,  "cost_price": 18.00, "reorder_point": 30, "supplier": "SafetyPro"},
    {"id": "D003", "name": "Hydraulic Oil 10L",          "category": "Lubricants",  "unit_price": 85.00,  "cost_price": 42.00, "reorder_point": 20, "supplier": "ChemTech"},
    {"id": "D004", "name": "Packing Tape (Pack 6)",      "category": "Packaging",   "unit_price": 18.00,  "cost_price": 7.00,  "reorder_point": 80, "supplier": "PackMaster"},
    {"id": "D005", "name": "Bubble Wrap 50m",            "category": "Packaging",   "unit_price": 22.00,  "cost_price": 9.00,  "reorder_point": 40, "supplier": "PackMaster"},
    {"id": "D006", "name": "Warehouse Shelving Unit",    "category": "Equipment",   "unit_price": 280.00, "cost_price": 140.00,"reorder_point": 5,  "supplier": "StoreCo"},
    {"id": "D007", "name": "Forklift Pallet",            "category": "Equipment",   "unit_price": 55.00,  "cost_price": 25.00, "reorder_point": 20, "supplier": "StoreCo"},
    {"id": "D008", "name": "Hard Hat",                   "category": "Safety",      "unit_price": 28.00,  "cost_price": 10.00, "reorder_point": 25, "supplier": "SafetyPro"},
    {"id": "D009", "name": "Corrugated Box Large",       "category": "Packaging",   "unit_price": 8.00,   "cost_price": 3.00,  "reorder_point": 150,"supplier": "PackMaster"},
    {"id": "D010", "name": "Steel Strapping Roll",       "category": "Packaging",   "unit_price": 35.00,  "cost_price": 14.00, "reorder_point": 30, "supplier": "PackMaster"},
    {"id": "D011", "name": "Cleaning Solvent 5L",        "category": "Lubricants",  "unit_price": 42.00,  "cost_price": 18.00, "reorder_point": 20, "supplier": "ChemTech"},
    {"id": "D012", "name": "Safety Boots (Pair)",        "category": "Safety",      "unit_price": 65.00,  "cost_price": 28.00, "reorder_point": 15, "supplier": "SafetyPro"},
    {"id": "D013", "name": "Stretch Film 500m",          "category": "Packaging",   "unit_price": 28.00,  "cost_price": 11.00, "reorder_point": 40, "supplier": "PackMaster"},
    {"id": "D014", "name": "Barcode Scanner",            "category": "Equipment",   "unit_price": 180.00, "cost_price": 90.00, "reorder_point": 5,  "supplier": "TechSupply"},
    {"id": "D015", "name": "Anti-Static Wrap",           "category": "Packaging",   "unit_price": 32.00,  "cost_price": 13.00, "reorder_point": 25, "supplier": "PackMaster"},
    {"id": "D016", "name": "First Aid Kit",              "category": "Safety",      "unit_price": 38.00,  "cost_price": 15.00, "reorder_point": 20, "supplier": "SafetyPro"},
    {"id": "D017", "name": "Conveyor Belt Lubricant",    "category": "Lubricants",  "unit_price": 95.00,  "cost_price": 48.00, "reorder_point": 10, "supplier": "ChemTech"},
    {"id": "D018", "name": "Label Printer Ribbon",       "category": "Equipment",   "unit_price": 22.00,  "cost_price": 8.00,  "reorder_point": 30, "supplier": "TechSupply"},
    {"id": "D019", "name": "Pallet Wrap Machine",        "category": "Equipment",   "unit_price": 450.00, "cost_price": 220.00,"reorder_point": 2,  "supplier": "StoreCo"},
    {"id": "D020", "name": "Foam Corner Protectors",     "category": "Packaging",   "unit_price": 15.00,  "cost_price": 5.00,  "reorder_point": 60, "supplier": "PackMaster"},
]

WEIGHTS = [0.19, 0.15, 0.12, 0.10, 0.08,
           0.07, 0.06, 0.05, 0.04, 0.03,
           0.03, 0.02, 0.02, 0.01, 0.01,
           0.01, 0.01, 0.00, 0.00, 0.00]

REGIONS = ["North Hub", "South Hub", "East Hub", "West Hub", "Central Hub"]

def seasonal_multiplier(date):
    month = date.month
    # Distribution: peak Q4 (Oct-Dec) pre-Christmas logistics
    if month in [10, 11, 12]:
        return round(random.uniform(1.5, 1.9), 2)
    elif month in [1, 2]:
        return round(random.uniform(0.60, 0.80), 2)
    else:
        return round(random.uniform(0.90, 1.15), 2)

records = []
transaction_id = 3000
stock_levels = {p["id"]: random.randint(30, 100) for p in PRODUCTS}

date = START_DATE
while date <= END_DATE:
    seasonal = seasonal_multiplier(date)
    daily_transactions = int(random.uniform(10, 25) * seasonal)
    for _ in range(daily_transactions):
        product = random.choices(PRODUCTS, weights=WEIGHTS, k=1)[0]
        qty = random.randint(1, 8)
        region = random.choice(REGIONS)
        current_stock = stock_levels[product["id"]]
        if current_stock < qty:
            qty = max(1, current_stock)
            stock_levels[product["id"]] = 0
        else:
            stock_levels[product["id"]] -= qty
        if stock_levels[product["id"]] <= product["reorder_point"]:
            stock_levels[product["id"]] += random.randint(30, 80)
        revenue = round(product["unit_price"] * qty, 2)
        cost = round(product["cost_price"] * qty, 2)
        gross_profit = round(revenue - cost, 2)
        gross_margin = round((gross_profit / revenue) * 100, 1) if revenue > 0 else 0
        records.append({
            "transaction_id": transaction_id,
            "date": date.strftime("%Y-%m-%d"),
            "product_id": product["id"],
            "product_name": product["name"],
            "category": product["category"],
            "quantity_sold": qty,
            "unit_price": product["unit_price"],
            "cost_price": product["cost_price"],
            "revenue": revenue,
            "gross_profit": gross_profit,
            "gross_margin_pct": gross_margin,
            "stock_level": stock_levels[product["id"]],
            "reorder_point": product["reorder_point"],
            "supplier": product["supplier"],
            "region": region,
        })
        transaction_id += 1
    date += timedelta(days=1)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Distribution dataset generated.")
print(f"Total transactions : {len(df):,}")
print(f"Date range         : {df['date'].min()} to {df['date'].max()}")
print(f"Total revenue      : £{df['revenue'].sum():,.2f}")
print(f"Products           : {df['product_name'].nunique()}")

product_rev = df.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
total = product_rev.sum()
n_top = max(1, int(len(product_rev) * 0.2))
top_share = (product_rev.head(n_top).sum() / total * 100).round(1)
df["date_dt"] = pd.to_datetime(df["date"])
monthly = df.groupby(df["date_dt"].dt.month)["revenue"].sum()
months = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
print(f"\nPattern verification:")
print(f"  Pareto: top {n_top} products = {top_share}% revenue")
print(f"  Peak month: {months[monthly.idxmax()]}")
print(f"  Lowest month: {months[monthly.idxmin()]}")
print(f"Saved to: {OUTPUT_PATH}")