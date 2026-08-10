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

def run_data_understanding_agent(filepath: str) -> dict:
    """
    Agent 1: Data Understanding Agent
    Reads the SME dataset, profiles it, identifies key variables,
    detects quality issues, and returns a structured summary.
    """

    print("\n" + "="*60)
    print("AGENT 1: DATA UNDERSTANDING AGENT")
    print("="*60)

    # ── Step 1: Load data ──────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

    # ── Step 2: Programmatic profiling ─────────────────────
    profile = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "date_range": {
            "start": df["date"].min(),
            "end": df["date"].max()
        },
        "missing_values": df.isnull().sum().to_dict(),
        "total_revenue": round(df["revenue"].sum(), 2),
        "total_transactions": len(df),
        "unique_products": df["product_name"].nunique(),
        "unique_categories": df["category"].nunique(),
        "unique_suppliers": df["supplier"].nunique(),
        "unique_regions": df["region"].nunique(),
        "avg_transaction_value": round(df["revenue"].mean(), 2),
        "avg_gross_margin_pct": round(df["gross_margin_pct"].mean(), 1),
        "numeric_summary": df[["quantity_sold","revenue",
                                "gross_profit","gross_margin_pct",
                                "stock_level"]].describe().round(2).to_dict(),
        "category_breakdown": df.groupby("category")["revenue"].sum().round(2).to_dict(),
        "supplier_breakdown": df.groupby("supplier")["revenue"].sum().round(2).to_dict(),
        "region_breakdown": df.groupby("region")["revenue"].sum().round(2).to_dict(),
        "products_list": df["product_name"].unique().tolist(),
        "data_quality": {
            "duplicate_rows": int(df.duplicated().sum()),
            "negative_revenue": int((df["revenue"] < 0).sum()),
            "negative_stock": int((df["stock_level"] < 0).sum()),
            "zero_quantity": int((df["quantity_sold"] == 0).sum()),
        }
    }

    # ── Step 3: LLM interprets the profile ─────────────────
    prompt = f"""
You are a business data analyst reviewing an SME operational dataset.
Based on the dataset profile below, provide a clear structured summary
that will guide downstream analytical agents.

DATASET PROFILE:
{json.dumps(profile, indent=2)}

Provide your response in this exact structure:

DATASET OVERVIEW:
[2-3 sentences describing what this dataset contains and its scope]

KEY VARIABLES:
[List the most analytically important columns and what they represent]

DATA QUALITY ASSESSMENT:
[Comment on completeness, any issues found, and data reliability]

BUSINESS CONTEXT:
[What type of SME is this? What does the data suggest about the business?]

ANALYTICAL OPPORTUNITIES:
[What key analyses should be performed on this data?]
"""

    print("\nLLM interpreting dataset profile...")
    response = llm.invoke(prompt)

    # ── Step 4: Compile output ──────────────────────────────
    output = {
        "agent": "Data Understanding Agent",
        "status": "completed",
        "profile": profile,
        "interpretation": response.content
    }

    print("\n--- AGENT 1 OUTPUT ---")
    print(response.content)

    # Save output
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/agent1_data_understanding.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nOutput saved to outputs/agent1_data_understanding.json")

    return output


if __name__ == "__main__":
    result = run_data_understanding_agent("data/retail_sme_data.csv")