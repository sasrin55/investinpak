import pandas as pd
import streamlit as st
from datetime import date, timedelta
import altair as alt
import numpy as np

# ==============================================================================
# 1. CONFIGURATION AND INITIAL SETUP
# ==============================================================================

# --- SETTINGS ---
SHEET_ID = "1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q"
TAB_NAME = "Master"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={TAB_NAME}"
CURRENCY_CODE = "PKR" # For professional labeling
CURRENCY_FORMAT = "$,.0f" # Example format for PKR

st.set_page_config(
    page_title="Sales Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Title and Header ---
st.title("Enterprise Sales Performance Dashboard")
st.markdown("A unified view of transaction and commodity-level sales metrics.")
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

    # Map columns by position (A, B, C, D, E, F...)
    cols = df.columns
    if len(cols) < 6:
        st.error(f"Data structure error: Expected at least 6 columns, found {len(cols)}. Check the '{TAB_NAME}' tab.")
        st.stop()

    rename_map = {
        cols[0]: "date",
        cols[1]: "customer_name",
        cols[2]: "phone",
        cols[3]: "txn_type",
        cols[4]: "commodities",
        cols[5]: "amount_pkr",
    }
    df = df.rename(columns=rename_map)

    # Robust Cleaning and Conversion
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["amount_pkr"] = pd.to_numeric(df["amount_pkr"], errors="coerce").fillna(0)
    df["txn_type"] = df["txn_type"].astype(str).str.strip().str.title()
    df["customer_name"] = df["customer_name"].astype(str).str.strip()

    # Drop rows without a valid date
    df = df.dropna(subset=["date"])

    return df


def explode_commodities(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits transaction rows into one row per commodity, fairly allocating the total amount.
    """
    if base_df.empty or "commodities" not in base_df.columns:
        return base_df.copy()

    temp = base_df.copy()
    temp["commodities"] = temp["commodities"].fillna("").astype(str)

    # Normalize separators to commas using regex for robustness
    temp["commodities_clean"] = (
        temp["commodities"]
        .str.replace(r"[\s]*[&\/]| and ", ",", regex=True)
        .str.strip()
    )

    # Build list of commodities and calculate count
    temp["commodity_list"] = temp["commodities_clean"].apply(
        lambda s: [x.strip().title() for x in s.split(",") if x.strip() != ""]
    )
    temp["n_commodities"] = temp["commodity_list"].apply(
        lambda lst: len(lst) if len(lst) > 0 else np.nan # Use NaN for missing commodities
    )

    # Explode and filter out rows where no commodity was listed
    temp = temp.explode("commodity_list")
    temp = temp[temp["commodity_list"].notna() & (temp["n_commodities"].notna())]

    # Allocate amount evenly
    temp["amount_per_commodity"] = temp["amount_pkr"] / temp["n_commodities"]

    # Final cleanup
    temp = temp.rename(columns={"commodity_list": "commodity"})
    
    return temp[["date", "customer_name", "txn_type", "commodity", "amount_per_commodity", "amount_pkr"]]


# ==============================================================================
# 3. METRICS AND DATE LOGIC
# ==============================================================================

def sum_between(df, start, end, amount_col="amount_pkr"):
    """Calculates the sum of amount between two dates (inclusive)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask, amount_col].sum()

def count_transactions(df, start, end):
    """Counts the number of unique transactions (rows in the raw data)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].shape[0]

# ==============================================================================
# 4. DATA LOADING, FILTERING, AND PRE-CALCULATIONS
# ==============================================================================

raw_df = load_data()
exploded_df = explode_commodities(raw_df)

if raw_df.empty:
    st.stop()

# Date Calculations
today = date.today()
min_data_date = raw_df["date"].min()
max_data_date = raw_df["date"].max()

start_of_week = today - timedelta(days=today.weekday())
start_of_month = today.replace(day=1)
start_of_year = date(today.year, 1, 1)
last_7_days_start = today - timedelta(days=6)


# --- SIDEBAR FILTERS (Key for Enterprise Dashboards) ---
st.sidebar.header("Data Filter Selection")

# Date range selector
date_range = st.sidebar.date_input(
    "Reporting Period",
    value=(min_data_date, today),
    min_value=min_data_date,
    max_value=max_data_date
)

filter_start_date = min(date_range)
filter_end_date = max(date_range) if len(date_range) == 2 else date_range[0]

# Apply date filter to both dataframes
raw_df_filtered = raw_df[(raw_df["date"] >= filter_start_date) & (raw_df["date"] <= filter_end_date)]
exploded_df_filtered = exploded_df[(exploded_df["date"] >= filter_start_date) & (exploded_df["date"] <= filter_end_date)]

# Commodity filter (based on currently available data)
available_commodities = sorted(exploded_df_filtered["commodity"].unique().tolist())
selected_commodities = st.sidebar.multiselect(
    "Filter by Commodity Type",
    options=available_commodities,
    default=available_commodities # Default to All
)

if selected_commodities:
    exploded_df_filtered = exploded_df_filtered[exploded_df_filtered["commodity"].isin(selected_commodities)]

if raw_df_filtered.empty or exploded_df_filtered.empty:
    st.warning("No data matches the current filter criteria. Please adjust your selections.")
    st.stop()

# ==============================================================================
# 5. KEY PERFORMANCE INDICATORS (KPIs)
# ==============================================================================

st.subheader(f"Current Period KPIs (Through {today.strftime('%b %d, %Y')})")

col1, col2, col3, col4, col5 = st.columns(5)

# Helper function for metric formatting
def metric_format(value):
    return f"{CURRENCY_CODE} {value:,.0f}"

with col1:
    st.metric("Transactions Today", count_transactions(raw_df, today, today))
    st.metric("Amount Today", metric_format(sum_between(raw_df, today, today)))

with col2:
    st.metric("Last 7-Day Count", count_transactions(raw_df, last_7_days_start, today))
    st.metric("Last 7-Day Amount", metric_format(sum_between(raw_df, last_7_days_start, today)))

with col3:
    st.metric("Current Week Count", count_transactions(raw_df, start_of_week, today))
    st.metric("Current Week Amount", metric_format(sum_between(raw_df, start_of_week, today)))

with col4:
    st.metric("Current Month Count", count_transactions(raw_df, start_of_month, today))
    st.metric("Current Month Amount", metric_format(sum_between(raw_df, start_of_month, today)))

with col5:
    st.metric("YTD Count", count_transactions(raw_df, start_of_year, today))
    st.metric("YTD Amount", metric_format(sum_between(raw_df, start_of_year, today)))

st.markdown("---")

# ==============================================================================
# 6. VISUALIZATION: SALES TREND
# ==============================================================================

st.header("Sales Trend Analysis")

daily_summary = (
    raw_df_filtered.groupby("date")["amount_pkr"]
    .sum()
    .reset_index()
    .rename(columns={"amount_pkr": "Total_Sales"})
    .sort_values("date")
)

# Altair Line Chart for professional look and interactivity
chart_trend = alt.Chart(daily_summary).mark_line(point=True).encode(
    x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d, %Y")),
    y=alt.Y("Total_Sales:Q", title=f"Total Sales ({CURRENCY_CODE})", axis=alt.Axis(format=CURRENCY_FORMAT)),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("Total_Sales:Q", title="Amount", format=CURRENCY_FORMAT)
    ]
).properties(
    title=f"Daily Sales Volume: {filter_start_date.strftime('%b %d, %Y')} to {filter_end_date.strftime('%b %d, %Y')}"
).interactive() # Allows Zoom/Pan

st.altair_chart(chart_trend, use_container_width=True)

st.markdown("---")

# ==============================================================================
# 7. VISUALIZATION: COMMODITY BREAKDOWN
# ==============================================================================

st.header("Commodity Performance & Mix")

# Commodity summary – uses the split amount
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
    
    top_n = st.slider("Select Top N Commodities to Display", 5, len(commodity_summary), 10, key="top_n_slider")
    
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
    # Use pandas style to apply currency formatting
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
# 8. DATA EXPLORER
# ==============================================================================

st.header("Data Explorer: Transaction and Commodity Detail")
st.caption(f"Showing data for period: {filter_start_date.strftime('%b %d, %Y')} to {filter_end_date.strftime('%b %d, %Y')}")

# Select box for data view
data_choice = st.selectbox(
    "Select Data View:",
    options=["Raw Transactions (Total Amount)", "Exploded Data (Commodity Level Amount)"]
)

if data_choice == "Raw Transactions (Total Amount)":
    st.subheader("Raw Transaction Data")
    # Prepare data for display
    df_display = raw_df_filtered.sort_values("date", ascending=False).drop(columns=['phone'], errors='ignore')
    
    # Apply styling
    styled_df = df_display.style.format({
        "amount_pkr": f"{CURRENCY_CODE} {{:,.0f}}",
    })
    st.dataframe(styled_df, use_container_width=True)

else:
    st.subheader("Exploded Commodity Data")
    # Prepare data for display
    df_display = exploded_df_filtered.sort_values(["date", "customer_name"], ascending=False).drop(columns=['amount_pkr'], errors='ignore')

    # Apply styling
    styled_df = df_display.style.format({
        "amount_per_commodity": f"{CURRENCY_CODE} {{:,.0f}}",
    })
    st.dataframe(styled_df, use_container_width=True)
