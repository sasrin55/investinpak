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
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = "$,.0f"

# --- CUSTOM THEME COLORS ---
PAR_PRIMARY = "#006B3F"      # Dark Teal/Green (Buttons, Sliders, Highlights)
PAR_BACKGROUND = "#F0F5F2"   # Light Gray/Green background
PAR_TEXT = "#101010" # Dark text color

# Configure the page layout and theme
st.set_page_config(
    page_title="Zaraimandi Sales Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    
    primaryColor=PAR_PRIMARY,
    backgroundColor=PAR_BACKGROUND,
    secondaryBackgroundColor="#FFFFFF", 
    textColor=PAR_TEXT,
    font="sans serif"
)

# --- CSS Injection for Robust Theming ---
st.markdown(f"""
<style>
    /* Set Primary Color for buttons, selections, etc. */
    .stButton>button, .stSlider .st-bd, .stMultiSelect [data-baseweb="tag"] > span {{
        background-color: {PAR_PRIMARY};
        color: white;
    }}
    .stSelectbox div[data-baseweb="select"] {{
        border-color: {PAR_PRIMARY};
    }}
    /* Set overall background color */
    [data-testid="stAppViewContainer"] {{
        background-color: {PAR_BACKGROUND};
    }}
    /* Ensure Streamlit containers (like metric boxes) are white for contrast */
    [data-testid="stVerticalBlock"] {{
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
    }}
    /* Style headers */
    h1, h2, h3, h4 {{
        color: {PAR_PRIMARY};
    }}
</style>
""", unsafe_allow_html=True)
# ----------------------------------------

# --- Title and Header ---
st.title("Zaraimandi Sales Dashboard")
st.markdown("A unified view of transaction and commodity-level sales metrics.")
st.markdown("---")


# ==============================================================================
# 2. DATA LOADING AND CLEANUP (Functions remain the same)
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
    """Splits transaction rows into one row per commodity, fairly allocating the total amount."""
    if base_df.empty or "commodities" not in base_df.columns:
        return base_df.copy()

    temp = base_df.copy()
    temp["commodities"] = temp["commodities"].fillna("").astype(str)
    temp["commodities_clean"] = (
        temp["commodities"]
        .str.replace(r"[\s]*[&\/]| and ", ",", regex=True)
        .str.strip()
    )

    temp["commodity_list"] = temp["commodities_clean"].apply(
        lambda s: [x.strip().title() for x in s.split(",") if x.strip() != ""]
    )
    temp["n_commodities"] = temp["commodity_list"].apply(
        lambda lst: len(lst) if len(lst) > 0 else np.nan
    )

    temp = temp.explode("commodity_list")
    temp = temp[temp["commodity_list"].notna() & (temp["n_commodities"].notna())]
    temp["amount_per_commodity"] = temp["amount_pkr"] / temp["n_commodities"]
    temp = temp.rename(columns={"commodity_list": "commodity"})
    
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

start_of_week = today - timedelta(days=today.weekday())
start_of_month = today.replace(day=1)
start_of_year = date(today.year, 1, 1)
last_7_days_start = today - timedelta(days=6)


## 📊 Top-Level Filters
st.subheader("Reporting Filters")
filter_cols = st.columns([1, 4])

# 1. Date range selector
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
# 4. KEY PERFORMANCE INDICATORS (KPIs) - Structured Logically
# ==============================================================================

st.header("Key Performance Indicators (KPIs)")

def metric_format(value):
    return f"{CURRENCY_CODE} {value:,.0f}"

## KPI Section 1: Top Transaction Count Summary Table
st.subheader("Transaction Count Summary")

txn_data = {
    "Period": ["Transactions Today", "Transactions This Week", "Transactions This Month"],
    "Count": [
        count_transactions(raw_df, today, today),
        count_transactions(raw_df, start_of_week, today),
        count_transactions(raw_df, start_of_month, today)
    ]
}
txn_df = pd.DataFrame(txn_data)

# Transpose the DataFrame to get periods as columns
txn_df_transposed = txn_df.set_index('Period').T 

# Display using st.dataframe for a professional table look
st.dataframe(
    txn_df_transposed,
    hide_index=True,
    use_container_width=True
)

st.markdown("---")


## KPI Section 2: Amount Metrics (Original format preserved)
st.subheader("Current Activity Snapshot")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric("**Transactions Today**", count_transactions(raw_df, today, today))
    st.metric("**Amount Today**", metric_format(sum_between(raw_df, today, today)))

with kpi_col2:
    st.metric("**Last 7-Day Count**", count_transactions(raw_df, last_7_days_start, today))
    st.metric("**Last 7-Day Amount**", metric_format(sum_between(raw_df, last_7_days_start, today)))

with kpi_col3:
    st.metric("**Current Week Count**", count_transactions(raw_df, start_of_week, today))
    st.metric("**Current Week Amount**", metric_format(sum_between(raw_df, start_of_week, today)))

with kpi_col4:
    st.metric("**Current Month Count**", count_transactions(raw_df, start_of_month, today))
    st.metric("**Current Month Amount**", metric_format(sum_between(raw_df, start_of_month, today)))

st.markdown("---")

## KPI Section 3: Cumulative Performance
st.subheader("Cumulative Performance")
kpi_col_ytd, _, _, _ = st.columns(4)

with kpi_col_ytd:
    st.metric("**YTD Transactions**", count_transactions(raw_df, start_of_year, today))
    st.metric("**YTD Amount**", metric_format(sum_between(raw_df, start_of_year, today)))

st.markdown("---")


# ==============================================================================
# 5. VISUALIZATION: SALES TREND
# ==============================================================================

st.header("Sales Trend Analysis")

daily_summary = (
    raw_df_filtered.groupby("date")["amount_pkr"]
    .sum()
    .reset_index()
    .rename(columns={"amount_pkr": "Total_Sales"})
    .sort_values("date")
)

chart_trend = alt.Chart(daily_summary).mark_line(point=True).encode(
    x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %d, %Y")),
    y=alt.Y("Total_Sales:Q", title=f"Total Sales ({CURRENCY_CODE})", axis=alt.Axis(format=CURRENCY_FORMAT)),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("Total_Sales:Q", title="Amount", format=CURRENCY_FORMAT)
    ]
).properties(
    title=f"Daily Sales Volume: {filter_start_date.strftime('%b %d, %Y')} to {filter_end_date.strftime('%b %d, %Y')}"
).interactive()

st.altair_chart(chart_trend, use_container_width=True)

st.markdown("---")

# ==============================================================================
# 6. VISUALIZATION: COMMODITY BREAKDOWN
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
# 7. DATA EXPLORER
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
    df_display = exploded_df_filtered.sort_values(["date", "customer_name"], ascending=False).drop(columns=['amount_pkr'], errors='ignore')

    styled_df = df_display.style.format({
        "amount_per_commodity": f"{CURRENCY_CODE} {{:,.0f}}",
    })
    st.dataframe(styled_df, use_container_width=True)
