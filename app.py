import pandas as pd
import streamlit as st
from datetime import date, timedelta
import altair as alt
import numpy as np
import re # Import regex for advanced cleaning

# ==============================================================================
# 1. CONFIGURATION AND INITIAL SETUP
# ==============================================================================

# --- SETTINGS ---
SHEET_ID = "1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q"
TAB_NAME = "Master"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={TAB_NAME}"
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = "$,.0f"

# Configure the page layout (Standard, stable arguments)
st.set_page_config(
    page_title="Zaraimandi Sales Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Title and Header ---
st.title("Zaraimandi Sales Dashboard")
st.markdown("Transaction and Commodity-level Sales Intelligence.")
st.markdown("---")


# ==============================================================================
# 2. DATA LOADING AND CLEANUP
# ==============================================================================

@st.cache_data(show_spinner="Connecting to Data Source and Loading...")
def load_data():
    """Reads data, renames columns, and cleans data types."""
    try:
        df = pd.read_csv(CSV_URL)
    except Exception as e:
        st.error(f"Error connecting to Google Sheet. Check ID, tab name, and permissions. Details: {e}")
        st.stop()
        return pd.DataFrame()

    cols = df.columns
    if len(cols) < 6:
        st.error(f"Data structure error: Expected at least 6 columns, found {len(cols)}. Check the '{TAB_NAME}' tab.")
        st.stop()

    rename_map = {
        cols[0]: "date", cols[1]: "customer_name", cols[2]: "phone",
        cols[3]: "txn_type", cols[4]: "commodities", cols[5]: "amount_pkr",
    }
    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["amount_pkr"] = pd.to_numeric(df["amount_pkr"], errors="coerce").fillna(0)
    df["txn_type"] = df["txn_type"].astype(str).str.strip().str.title()
    df["customer_name"] = df["customer_name"].astype(str).str.strip()
    df = df.dropna(subset=["date"])

    return df


def explode_commodities(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits transaction rows into one row per commodity, fairly allocating the total amount,
    with enhanced cleaning for de-duplication.
    """
    if base_df.empty or "commodities" not in base_df.columns:
        return base_df.copy()

    temp = base_df.copy()
    temp["commodities"] = temp["commodities"].fillna("").astype(str)

    # Normalize separators to commas
    temp["commodities_clean"] = (
        temp["commodities"]
        .str.replace(r"[\s]*[&\/]| and ", ",", regex=True)
        .str.strip()
    )

    # Build list of commodities
    temp["commodity_list"] = temp["commodities_clean"].apply(
        lambda s: [x.strip() for x in s.split(",") if x.strip() != ""]
    )
    
    # --- AGGRESSIVE COMMODITY CLEANING/NORMALIZATION ---
    def normalize_commodity_name(name):
        name = name.lower().strip()
        if not name:
            return None
        # Remove common extraneous words
        name = re.sub(r' (data|group|s|\.)$', '', name).strip()
        
        # Consolidation mapping for common misspellings/variants
        mapping = {
            'cotton': 'Cotton',
            'coton': 'Cotton',
            'paddy': 'Paddy',
            'padd': 'Paddy',
            'wheat': 'Wheat',
            'wheat and paddy': 'Wheat & Paddy',
            'edible oil': 'Edible Oil',
            'edibleoil': 'Edible Oil',
            'fertilizers': 'Fertilizer',
            'fertilizer': 'Fertilizer',
            'pulses': 'Pulses',
            'daal': 'Pulses',
            'bajra': 'Bajra',
            'lm': 'Livestock', # Assuming 'Lm' is short for Livestock marketing
            'livestock': 'Livestock'
        }
        
        # Apply mapping or title case if no map found
        for key, value in mapping.items():
            if key in name:
                return value
        
        return name.title()

    temp["commodity_list"] = temp["commodity_list"].apply(
        lambda lst: [normalize_commodity_name(item) for item in lst if normalize_commodity_name(item) is not None]
    )
    # ---------------------------------------------------

    temp["n_commodities"] = temp["commodity_list"].apply(
        lambda lst: len(lst) if len(lst) > 0 else np.nan
    )

    temp = temp.explode("commodity_list")
    temp = temp[temp["commodity_list"].notna() & (temp["n_commodities"].notna())]
    temp = temp.rename(columns={"commodity_list": "commodity"})
    
    # Re-calculate amount per commodity after cleaning/explosion
    temp["amount_per_commodity"] = temp["amount_pkr"] / temp["n_commodities"]
    
    return temp[["date", "customer_name", "txn_type", "commodity", "amount_per_commodity", "amount_pkr"]]


def sum_between(df, start, end, amount_col="amount_pkr"):
    """Calculates the sum of amount between two dates (inclusive)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask, amount_col].sum()

def count_transactions(df, start, end):
    """Counts the number of unique transactions (rows in the raw data)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].shape[0]

# ==============================================================================
# 3. DATA LOADING, FILTERING, AND PRE-CALCULATIONS
# ==============================================================================

raw_df = load_data()
exploded_df = explode_commodities(raw_df)

if raw_df.empty:
    st.stop()

# Date Calculations
today = date.today()
min_data_date = raw_df["date"].min()
max_data_date = raw_df["date"].max()

start_of_year = date(today.year, 1, 1)
last_30_days_start = today - timedelta(days=29) # 30 days including today

def metric_format(value):
    return f"{CURRENCY_CODE} {value:,.0f}"

## 📊 Top-Level Filters
st.subheader("Reporting Filters")
filter_cols = st.columns([1, 4])

# 1. Date range selector (Only filter remaining)
with filter_cols[0]:
    date_range = st.date_input(
        "Reporting Period",
        value=(min_data_date, today),
        min_value=min_data_date,
        max_value=max_data_date,
        key="top_date_filter"
    )

filter_start_date = min(date_range)
filter_end_date = max(date_range) if len(date_range) == 2 else date_range[0]

# Apply date filter to both dataframes
raw_df_filtered = raw_df[(raw_df["date"] >= filter_start_date) & (raw_df["date"] <= filter_end_date)]
exploded_df_filtered = exploded_df[(exploded_df["date"] >= filter_start_date) & (exploded_df["date"] <= filter_end_date)]


if raw_df_filtered.empty or exploded_df_filtered.empty:
    st.warning("No data matches the current date filter criteria. Please adjust your selections.")
    st.stop()

st.markdown("---")

# ==============================================================================
# 4. KEY PERFORMANCE INDICATORS (KPIs) - STRUCTURED TABLES
# ==============================================================================

st.header("Key Performance Indicators (KPIs)")

## KPI Section 1: YTD Amount (On Top)
st.subheader("Year-To-Date Cumulative Sales")
ytd_amount = sum_between(raw_df, start_of_year, today)

kpi_col_ytd, _, _, _ = st.columns(4)
with kpi_col_ytd:
    st.metric("**YTD Total Sales**", metric_format(ytd_amount))

st.markdown("---")


## KPI Section 2: Detailed Transaction Summaries

def create_summary_table(df, period_start, period_end, title):
    """Generates a detailed table for a specific period."""
    
    # Filter the exploded data for the specific period
    period_mask = (df["date"] >= period_start) & (df["date"] <= period_end)
    summary_df = exploded_df.loc[period_mask].copy()
    
    # Group by customer and commodity to summarize the transaction
    summary_df = summary_df.groupby(["customer_name", "commodity"])["amount_per_commodity"].sum().reset_index()
    
    summary_df = summary_df.rename(columns={
        "customer_name": "Customer",
        "commodity": "Commodity",
        "amount_per_commodity": "Amount"
    })
    
    # Sort and format the output
    summary_df = summary_df.sort_values("Amount", ascending=False)
    
    # Apply styling for better presentation
    styled_df = summary_df.style.format({
        "Amount": f"{CURRENCY_CODE} {{:,.0f}}",
    })

    st.subheader(f"{title} (Transactions: {count_transactions(raw_df, period_start, period_end)})")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


col_today, col_30days, col_year = st.columns(3)

with col_today:
    create_summary_table(exploded_df, today, today, "Today's Sales Breakdown")
    
with col_30days:
    create_summary_table(exploded_df, last_30_days_start, today, "Last 30 Days Sales Breakdown")

with col_year:
    # Use YTD for the third column, as Today and 30 days are already covered
    create_summary_table(exploded_df, start_of_year, today, "YTD Sales Breakdown")
    

st.markdown("---")
# ==============================================================================
# 5. VISUALIZATION: COMMODITY BREAKDOWN (Original chart kept, trend removed)
# ==============================================================================

st.header("Commodity Performance & Mix")

commodity_summary = (
    exploded_df_filtered.groupby("commodity")["amount_per_commodity"]
    .sum()
    .reset_index()
    .rename(columns={"amount_per_commodity": "Amount"})
    .sort_values("Amount", ascending=False)
)

col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.subheader("Top Selling Commodities (Volume)")

    max_commodities = len(commodity_summary)
    top_n_default = min(10, max_commodities)
    
    top_n = st.slider(
        "Select Top N Commodities to Display", 
        1, 
        max_commodities, 
        top_n_default, 
        key="top_n_slider"
    )

    top_commodity_summary = commodity_summary.head(top_n)

    # Altair Bar Chart
    chart_bar = alt.Chart(top_commodity_summary).mark_bar().encode(
        x=alt.X("Amount:Q", title=f"Total Sales ({CURRENCY_CODE})", axis=alt.Axis(format=CURRENCY_FORMAT)),
        y=alt.Y("commodity:N", sort="-x", title="Commodity"),
        tooltip=[
            alt.Tooltip("commodity:N", title="Commodity"),
            alt.Tooltip("Amount:Q", title="Sales Amount", format=CURRENCY_FORMAT)
        ]
    ).properties(
        title=f"Top {top_n} Commodities by Sales Amount"
    )
    st.altair_chart(chart_bar, use_container_width=True)

with col_table:
    st.subheader("Summary Table")
    styled_df = commodity_summary.style.format({
        "Amount": f"{CURRENCY_CODE} {{:,.0f}}",
    })

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==============================================================================
# 6. DATA EXPLORER
# ==============================================================================

st.header("Data Explorer: Transaction and Commodity Detail")
st.caption(f"Showing data for period: {filter_start_date.strftime('%b %d, %Y')} to {filter_end_date.strftime('%b %d, %Y')}")

data_choice = st.selectbox(
    "Select Data View:",
    options=["Raw Transactions (Total Amount)", "Exploded Data (Commodity Level Amount)"]
)

if data_choice == "Raw Transactions (Total Amount)":
    st.subheader("Raw Transaction Data")
    df_display = raw_df_filtered.sort_values("date", ascending=False).drop(columns=['phone'], errors='ignore')

    styled_df = df_display.style.format({
        "amount_pkr": f"{CURRENCY_CODE} {{:,.0f}}",
    })
    st.dataframe(styled_df, use_container_width=True)

else:
    st.subheader("Exploded Commodity Data")
    # Use the filtered data for display, but show the cleaned commodity name
    df_display = exploded_df_filtered.sort_values(["date", "customer_name"], ascending=False).drop(columns=['amount_pkr'], errors='ignore')

    styled_df = df_display.style.format({
        "amount_per_commodity": f"{CURRENCY_CODE} {{:,.0f}}",
    })
    st.dataframe(styled_df, use_container_width=True)
