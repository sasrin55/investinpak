import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import altair as alt

st.set_page_config(
    page_title="Zarai Mandi Focused Dashboard",
    layout="wide",
)

st.title("Zarai Mandi Sales Dashboard (Focused Metrics)")
st.caption("Displaying Total Sales, Total Customers, and Return Customers.")

# --- CONSTANTS ---
CUSTOMER_TYPE_COL = "customer_type"
AMOUNT_COL = "amount"
CUSTOMER_COL = "customer"
COMMODITY_COL = "commodity"
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f" 

# ---------- CONFIG ----------

# EXPLICIT CSV EXPORT LINK for 'Master for SalesOps' (GID=1105756916)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q/gviz/tq?tqx=out:csv&gid=1105756916"


# ---------- DATA LOADING (Clean & Simple) ----------

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(SHEET_URL)

        # 1. Standardise column names
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )

        # 2. Parse date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.date

        # 3. Clean numeric amount
        if AMOUNT_COL in df.columns:
            df[AMOUNT_COL] = (
                df[AMOUNT_COL]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            df[AMOUNT_COL] = pd.to_numeric(df[AMOUNT_COL], errors="coerce").fillna(0)
        
        # 4. FIX: UNIFY CUSTOMER TYPE
        if CUSTOMER_TYPE_COL in df.columns:
            df[CUSTOMER_TYPE_COL] = df[CUSTOMER_TYPE_COL].str.strip().str.title()

        # 5. Drop fully empty rows
        df = df.dropna(how="all")

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# --- EXECUTION ---
raw_df = load_data()

if raw_df.empty:
    st.error("No data loaded from Google Sheets. Check permissions and URL.")
    st.stop()

# --- METRIC CALCULATIONS ---

# 1. Total Sales
total_amount = raw_df[AMOUNT_COL].sum() if AMOUNT_COL in raw_df.columns else 0

# 2. Total Customers (Unique phone numbers)
total_customers = raw_df[CUSTOMER_COL].nunique() if CUSTOMER_COL in raw_df.columns else 0

# 3. Return Customers
if CUSTOMER_TYPE_COL in raw_df.columns and CUSTOMER_COL in raw_df.columns:
    return_customers = raw_df[raw_df[CUSTOMER_TYPE_COL] == "Return"][CUSTOMER_COL].nunique()
else:
    return_customers = 0

# 4. Sales by Commodity (Grouped by the raw, full string in the Commodity column)
if COMMODITY_COL in raw_df.columns and AMOUNT_COL in raw_df.columns:
    commodity_sales_df = (
        raw_df.groupby(COMMODITY_COL)[AMOUNT_COL] 
        .sum()
        .rename("Total Sales")
        .reset_index()
    )
    # Filter out empty/invalid commodity names
    commodity_sales_df = commodity_sales_df[commodity_sales_df[COMMODITY_COL].notna()].head(15)
else:
    commodity_sales_df = pd.DataFrame()


st.write("---")

# ---------- KPI BLOCKS (Total Sales, Total Customers, Return Customers) ----------

st.header("Key Performance Indicators")

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales (PKR)", f"{total_amount:{CURRENCY_FORMAT}}")
col2.metric("Total Customers (by Phone)", f"{total_customers:,}")
col3.metric("Return Customers", f"{return_customers:,}")

st.markdown("---")

# ---------- SALES BY COMMODITY (Chart Format) ----------

if not commodity_sales_df.empty:
    st.header("Total Sales by Commodity")
    st.caption("Note: Commodities are grouped by the full string found in the data (e.g., 'Wheat, Paddy, & Cotton' is one bar).")

    # Create Altair Bar Chart
    commodity_chart = alt.Chart(commodity_sales_df).mark_bar().encode(
        x=alt.X('Total Sales', title=f'Total Sales ({CURRENCY_CODE})', axis=alt.Axis(format=CURRENCY_FORMAT)),
        y=alt.Y(COMMODITY_COL, sort='-x', title='Commodity Group'),
        tooltip=[COMMODITY_COL, alt.Tooltip('Total Sales', format=f"{CURRENCY_CODE} {CURRENCY_FORMAT}", title='Total Sales')]
    ).properties(title="Sales Volume by Commodity String").interactive()
    
    st.altair_chart(commodity_chart, use_container_width=True)
else:
    st.info("Commodity data is unavailable for plotting.")

st.markdown("---")

# ---------- DATA PREVIEW ----------
st.header("Raw Data Preview")
st.dataframe(raw_df.head(10), use_container_width=True)
