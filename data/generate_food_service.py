import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(123)
random.seed(123)

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "food_service_data.csv")

PRODUCTS = [
    {"id": "F001", "name": "Grilled Chicken Burger",  "category": "Mains",      "unit_price": 8.50,  "cost_price": 2.80,  "reorder_point": 50, "supplier": "FreshFoods"},
    {"id": "F002", "name": "Margherita Pizza",         "category": "Mains",      "unit_price": 9.00,  "cost_price": 2.50,  "reorder_point": 40, "supplier": "ItalianDeli"},
    {"id": "F003", "name": "Veggie Wrap",              "category": "Mains",      "unit_price": 6.50,  "cost_price": 1.80,  "reorder_point": 40, "supplier": "FreshFoods"},
    {"id": "F004", "name": "Fish and Chips",           "category": "Mains",      "unit_price": 10.00, "cost_price": 3.50,  "reorder_point": 30, "supplier": "SeaFresh"},
    {"id": "F005", "name": "Beef Burger",              "category": "Mains",      "unit_price": 9.50,  "cost_price": 3.20,  "reorder_point": 35, "supplier": "FreshFoods"},
    {"id": "F006", "name": "Caesar Salad",             "category": "Starters",   "unit_price": 5.50,  "cost_price": 1.50,  "reorder_point": 30, "supplier": "FreshFoods"},
    {"id": "F007", "name": "Garlic Bread",             "category": "Starters",   "unit_price": 3.50,  "cost_price": 0.80,  "reorder_point": 50, "supplier": "ItalianDeli"},
    {"id": "F008", "name": "Soup of the Day",          "category": "Starters",   "unit_price": 4.00,  "cost_price": 1.00,  "reorder_point": 40, "supplier": "FreshFoods"},
    {"id": "F009", "name": "Coca-Cola 330ml",          "category": "Drinks",     "unit_price": 2.50,  "cost_price": 0.60,  "reorder_point": 100,"supplier": "DrinksCo"},
    {"id": "F010", "name": "Fresh Orange Juice",       "category": "Drinks",     "unit_price": 3.00,  "cost_price": 0.80,  "reorder_point": 80, "supplier": "FreshFoods"},
    {"id": "F011", "name": "Coffee",                   "category": "Drinks",     "unit_price": 2.80,  "cost_price": 0.50,  "reorder_point": 100,"supplier": "CoffeePro"},
    {"id": "F012", "name": "Chocolate Cake",           "category": "Desserts",   "unit_price": 4.50,  "cost_price": 1.20,  "reorder_point": 20, "supplier": "BakeryCo"},
    {"id": "F013", "name": "Ice Cream Sundae",         "category": "Desserts",   "unit_price": 4.00,  "cost_price": 1.00,  "reorder_point": 25, "supplier": "BakeryCo"},
    {"id": "F014", "name": "Cheesecake",               "category": "Desserts",   "unit_price": 4.80,  "cost_price": 1.40,  "reorder_point": 15, "supplier": "BakeryCo"},
    {"id": "F015", "name": "Kids Meal",                "category": "Mains",      "unit_price": 5.50,  "cost_price": 1.80,  "reorder_point": 20, "supplier": "FreshFoods"},
    {"id": "F016", "name": "Sparkling Water",          "category": "Drinks",     "unit_price": 2.00,  "cost_price": 0.40,  "reorder_point": 80, "supplier": "DrinksCo"},
    {"id": "F017", "name": "Chicken Tikka Masala",     "category": "Specials",   "unit_price": 11.00, "cost_price": 3.80,  "reorder_point": 20, "supplier": "SpiceWorld"},
    {"id": "F018", "name": "Mushroom Risotto",         "category": "Specials",   "unit_price": 10.50, "cost_price": 3.20,  "reorder_point": 15, "supplier": "ItalianDeli"},
    {"id": "F019", "name": "Prawn Cocktail",           "category": "Starters",   "unit_price": 6.00,  "cost_price": 2.00,  "reorder_point": 20, "supplier": "SeaFresh"},
    {"id": "F020", "name": "Mixed Grill",              "category": "Specials",   "unit_price": 14.00, "cost_price": 5.00,  "reorder_point": 10, "supplier": "FreshFoods"},
]

# Pareto weights — top 20% drive ~65% of revenue
WEIGHTS = [0.20, 0.17, 0.12, 0.09, 0.08,
           0.06, 0.06, 0.05, 0.04, 0.03,
           0.02, 0.02, 0.01, 0.01, 0.01,
           0.01, 0.01, 0.00, 0.00, 0.00]

REGIONS = ["Dine-In", "Takeaway", "Delivery", "Catering", "Online"]

def seasonal_multiplier(date):
    month = date.month
    # Peak: summer and Christmas, low: January post-Christmas
    if month in [6, 7, 8, 12]:
        return round(random.uniform(1.4, 1.7), 2)
    elif month in [1, 2]:
        return round(random.uniform(0.55, 0.75), 2)
    else:
        return round(random.uniform(0.90, 1.15), 2)

records = []
transaction_id = 2000
stock_levels = {p["id"]: random.randint(60, 180) for p in PRODUCTS}

date = START_DATE
while date <= END_DATE:
    seasonal = seasonal_multiplier(date)
    # Food service: higher volume, smaller transactions
    daily_transactions = int(random.uniform(25, 60) * seasonal)
    for _ in range(daily_transactions):
        product = random.choices(PRODUCTS, weights=WEIGHTS, k=1)[0]
        qty = random.randint(1, 4)
        region = random.choice(REGIONS)
        current_stock = stock_levels[product["id"]]
        if current_stock < qty:
            qty = max(1, current_stock)
            stock_levels[product["id"]] = 0
        else:
            stock_levels[product["id"]] -= qty
        if stock_levels[product["id"]] <= product["reorder_point"]:
            stock_levels[product["id"]] += random.randint(60, 150)
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

print(f"Food Service dataset generated.")
print(f"Total transactions : {len(df):,}")
print(f"Date range         : {df['date'].min()} to {df['date'].max()}")
print(f"Total revenue      : £{df['revenue'].sum():,.2f}")
print(f"Products           : {df['product_name'].nunique()}")

# Verify patterns
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