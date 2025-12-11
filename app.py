import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import altair as alt
from typing import Dict, Any

st.set_page_config(
    page_title="Zarai Mandi Sales Dashboard",
    layout="wide",
)

st.title("Zarai Mandi Sales Dashboard")
st.caption("Transaction and Commodity-level Sales Intelligence.")

# ---------- CONFIG ----------

# EXPLICIT CSV EXPORT LINK for 'Master for SalesOps' (GID=1105756916)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q/gviz/tq?tqx=out:csv&gid=1105756916"

# Define the expected final column names after cleaning/renaming
COMMODITIES_COL = "commodities_list"
AMOUNT_COL = "amount_pkr"
CUSTOMER_COL = "customer_name"
GROSS_AMOUNT_PER_COMMODITY_COL = "gross_amount_per_commodity"


# ---------- DATA TRANSFORMATIONS (Explode Logic) ----------

def explode_commodities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe where the COMMODITIES_COL contains a comma-separated
    list of commodity names and amounts and explodes this into one row per commodity.
    """
    df_temp = df.copy()
    
    # 1. Rename columns to the names expected by this function (for compatibility)
    df_temp.rename(
        columns={
            "commodity": COMMODITIES_COL,
            "amount": AMOUNT_COL,
            "customer": CUSTOMER_COL,
        },
        inplace=True,
    )
    
    original_cols = list(df_temp.columns)

    required_cols = [COMMODITIES_COL, AMOUNT_COL]
    if not all(col in df_temp.columns for col in required_cols):
        return pd.DataFrame(columns=original_cols + ["commodity", GROSS_AMOUNT_PER_COMMODITY_COL])

    if df_temp[COMMODITIES_COL].isna().all():
        return pd.DataFrame(columns=original_cols + ["commodity", GROSS_AMOUNT_PER_COMMODITY_COL])

    df_temp[COMMODITIES_COL] = df_temp[COMMODITIES_COL].fillna("")

    # 2. Split the list by comma and explode
    df_temp["commodity_items"] = df_temp[COMMODITIES_COL].str.split(",").apply(
        lambda x: [item.strip() for item in x if str(item).strip()]
    )
    df_exploded = df_temp.explode("commodity_items")
    
    # 3. Handle cases where the commodity list is not split by amount
    
    # Check if any item contains digits at the start 
    has_split_amounts = df_exploded["commodity_items"].str.match(r"^\s*\d+").any()

    if has_split_amounts:
        # Complex logic (for "200 Edible oil, 300 Paddy")
        pattern = r"^\s*(\d+)\s*(.*)"
        df_exploded[["amount_split", "commodity"]] = df_exploded["commodity_items"].str.extract(pattern)
        df_exploded["amount_split"] = pd.to_numeric(df_exploded["amount_split"], errors="coerce").fillna(0)
        
        sum_split_per_txn = df_exploded.groupby(df_exploded.index)["amount_split"].transform("sum")
        count_items_per_txn = df_exploded.groupby(df_exploded.index)["commodity_items"].transform("count")

        df_exploded[GROSS_AMOUNT_PER_COMMODITY_COL] = np.where(
            (sum_split_per_txn > 0) & (df_exploded[AMOUNT_COL] > 0),
            df_exploded["amount_split"] / sum_split_per_txn * df_exploded[AMOUNT_COL],
            np.where(
                count_items_per_txn > 0,
                df_exploded[AMOUNT_COL] / count_items_per_txn,
                0,
            ),
        )
        # Final commodity name is the text part of the split
        df_exploded["commodity"] = df_exploded["commodity"].astype(str).str.strip()
        
    else:
        # Simple logic (for "Wheat, Cotton, Paddy") - split transaction amount equally
        df_exploded["commodity"] = df_exploded["commodity_items"].astype(str).str.strip()
        count_items_per_txn = df_exploded.groupby(df_exploded.index)["commodity_items"].transform("count")
        
        df_exploded[GROSS_AMOUNT_PER_COMMODITY_COL] = np.where(
             count_items_per_txn > 0,
             df_exploded[AMOUNT_COL] / count_items_per_txn,
             0,
        )

    # 5. Filter out empty commodities and keep necessary columns
    df_final = df_exploded[df_exploded["commodity"] != ""].copy()

    final_cols = [col for col in original_cols if col not in [COMMODITIES_COL, AMOUNT_COL]] 
    final_cols += ["commodity", GROSS_AMOUNT_PER_COMMODITY_COL]
    
    # Ensure customer is retained for unique count
    if CUSTOMER_COL not in final_cols:
         final_cols.append(CUSTOMER_COL)

    return df_final[final_cols].reset_index(drop=True)


# ---------- DATA LOADING ----------

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(SHEET_URL)

    # Standardise column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    # Expecting: date, customer, customer_type, commodity, amount, duration, ...

    # Parse date
    if "date" in df.columns:
        # Using dayfirst=True to handle common non-US date formats
        df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.date

    # Clean numeric amount
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Drop fully empty rows
    df = df.dropna(how="all")

    return df


# --- EXECUTION ---
raw_df = load_data()

# If nothing loaded, show a clear message
if raw_df.empty:
    st.error("No data loaded from Google Sheets. Check sharing permissions and the URL.")
    st.stop()

# Explode the commodity data for accurate sales breakdown
exploded_df = explode_commodities(raw_df)

# --- START FIX FOR TYPE ERROR ON DATE MIN/MAX ---

# Filter out NaT values (which occur on failed date conversions)
valid_dates = raw_df["date"].dropna()

if valid_dates.empty:
    # If no valid dates are found, set defaults safely
    min_data_date = date(2020, 1, 1)
    max_data_date = date.today()
else:
    # Find the minimum and maximum of the valid date objects
    # This avoids the TypeError that occurs when mixing date objects and NaT/NaN during min()
    min_data_date = valid_dates.min()
    max_data_date = valid_dates.max()

# --- END FIX FOR TYPE ERROR ON DATE MIN/MAX ---


st.write("---")
st.write("Preview of Raw Data (First 5 Transactions):")
st.dataframe(raw_df.head(5), use_container_width=True)
st.write("Preview of Exploded Commodity Data (First 10 Rows):")
st.dataframe(exploded_df.head(10), use_container_width=True)
st.write("---")


# ---------- SIMPLE KPIs (Using Raw Data for Total Transactions/Customers) ----------

total_amount = raw_df["amount"].sum() if "amount" in raw_df.columns else 0
total_txns = len(raw_df)
unique_customers = raw_df["customer"].nunique() if "customer" in raw_df.columns else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales (PKR)", f"{total_amount:,.0f}")
col2.metric("Total Transactions", f"{total_txns:,}")
col3.metric("Unique Customers (by Phone)", f"{unique_customers:,}")

# ---------- SALES BY COMMODITY (Using Exploded Data for Accuracy) ----------

if "commodity" in exploded_df.columns:
    st.subheader("Sales by Commodity (Accurate Breakdown)")

    commodity_sales = (
        exploded_df.groupby("commodity")[GROSS_AMOUNT_PER_COMMODITY_COL] # Use the proportional amount
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(commodity_sales)
else:
    st.info("No 'commodity' column found to plot after processing.")

# --- Placeholder for future filtering logic that requires min/max dates ---
st.subheader("Reporting Period:")
st.caption(f"Data available from {min_data_date} to {max_data_date}")
