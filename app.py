import pandas as pd
import streamlit as st
from datetime import date, timedelta
import altair as alt # For more advanced charting

# ==============================================================================
# 1. SETTINGS AND CONFIGURATION
# ==============================================================================

# Use Streamlit secrets for sensitive information (Recommended in production)
# st.secrets['sheet_id'] or similar for production
SHEET_ID = "1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q"
TAB_NAME = "Master"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={TAB_NAME}"
CURRENCY_SYMBOL = "₹" # Added a setting for currency

st.set_page_config(
    page_title="💰 Pro Sales Tracker Dashboard",
    layout="wide",
    initial_sidebar_state="expanded" # Start with sidebar open
)
st.title("💰 Pro Sales Tracker Dashboard")

# ==============================================================================
# 2. DATA LOADING AND CLEANUP
# ==============================================================================

@st.cache_data(show_spinner="Connecting to Google Sheet and loading data...")
def load_data():
    """Reads data, renames columns, and cleans/converts data types."""
    try:
        # Read Google Sheet as CSV
        df = pd.read_csv(CSV_URL)
    except Exception as e:
        st.error(f"❌ **Error loading data.** Check if the Google Sheet ID is correct and publicly shared. Details: {e}")
        st.stop()
        return pd.DataFrame() # Return empty DataFrame on failure

    cols = df.columns

    # Check for expected number of columns (Improved error message)
    if len(cols) < 6:
        st.error(f"⚠️ Expected at least 6 columns, but found only {len(cols)}. Check the sheet tab name and header row.")
        st.stop()

    # Define robust column renaming by position
    rename_map = {
        cols[0]: "date",
        cols[1]: "name",
        cols[2]: "phone",
        cols[3]: "txn_type",
        cols[4]: "commodities",
        cols[5]: "amount",
    }
    df = df.rename(columns=rename_map)

    # Clean and convert data types (Improved conversion with robust error handling)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["name"] = df["name"].astype(str).str.strip()
    df["txn_type"] = df["txn_type"].astype(str).str.strip().str.title() # Normalize case

    # Drop rows without a valid date
    initial_rows = len(df)
    df = df.dropna(subset=["date"])
    st.sidebar.info(f"Loaded {len(df)} records. Dropped {initial_rows - len(df)} rows with invalid dates.")

    return df

def explode_commodities(df):
    """Explodes grouped commodities into one row per commodity."""
    if df.empty or "commodities" not in df.columns:
        return df

    temp = df.copy()
    temp["commodities"] = temp["commodities"].fillna("").astype(str)

    # Robust normalization: standard separators and then split
    temp["commodities"] = (
        temp["commodities"]
        .str.replace(r"[\s]*[&\/]| and ", ",", regex=True) # Replaced fixed strings with regex for better coverage
        .str.strip()
    )

    # Split into lists, explode, and strip whitespace again
    temp["commodities"] = temp["commodities"].str.split(",")
    temp = temp.explode("commodities")
    temp["commodities"] = temp["commodities"].str.strip().str.title() # Normalize commodity names

    # Remove empty rows after stripping
    temp = temp[temp["commodities"] != ""]

    return temp

# ==============================================================================
# 3. HELPER FUNCTIONS FOR METRICS
# ==============================================================================

def sum_between(df, start, end):
    """Calculates the sum of 'amount' between two dates (inclusive)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask, "amount"].sum()

def count_between(df, start, end):
    """Counts the number of unique transactions between two dates (inclusive)."""
    # Use the raw dataframe for unique transaction count to avoid counting exploded rows
    mask = (raw_df["date"] >= start) & (raw_df["date"] <= end)
    # Assuming each row in raw_df is one unique transaction
    return raw_df.loc[mask].shape[0]

# ==============================================================================
# 4. DATA PROCESSING AND DATE CALCULATIONS
# ==============================================================================

# Load and process data
raw_df = load_data() # Load the data first
df = explode_commodities(raw_df) # Then explode for commodity analysis

# Stop if data loading failed or DataFrame is empty
if raw_df.empty:
    st.stop()

# Date calculations
today = date.today()
start_of_week = today - timedelta(days=today.weekday())
start_of_month = today.replace(day=1)
start_of_year = date(today.year, 1, 1)
last_7_days = today - timedelta(days=6)

# Find the earliest date in the data for filtering
min_date = raw_df["date"].min()
max_date = raw_df["date"].max()

# ==============================================================================
# 5. SIDEBAR FILTERS (USER INTERACTIVITY)
# ==============================================================================

st.sidebar.header("📊 Data Filters")

# Date range selector
date_range = st.sidebar.date_input(
    "Select a Date Range",
    value=(min_date, today),
    min_value=min_date,
    max_value=max_date
)

# Ensure the tuple has two dates
if len(date_range) == 2:
    filter_start_date = min(date_range)
    filter_end_date = max(date_range)
else:
    # Handle single date selection (use it as start and end)
    filter_start_date = date_range[0]
    filter_end_date = date_range[0]

# Apply date range filter to both raw and exploded data
raw_df_filtered = raw_df[(raw_df["date"] >= filter_start_date) & (raw_df["date"] <= filter_end_date)]
df_filtered = df[(df["date"] >= filter_start_date) & (df["date"] <= filter_end_date)]

# Commodity multiselect filter (based on currently filtered data)
available_commodities = sorted(df_filtered["commodities"].unique().tolist())
selected_commodities = st.sidebar.multiselect(
    "Filter by Commodity",
    options=available_commodities,
    default=available_commodities # Default to all
)

# Apply commodity filter
if selected_commodities:
    df_filtered = df_filtered[df_filtered["commodities"].isin(selected_commodities)]

# Final check after filtering
if df_filtered.empty:
    st.warning("No data matches the selected filters. Please adjust the filters.")
    st.stop()

# ==============================================================================
# 6. TOP METRICS
# ==============================================================================

st.header(f"Key Metrics (As of {today.strftime('%b %d, %Y')})")
st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)

# Function to format amounts
def format_amount(amount):
    return f"{CURRENCY_SYMBOL}{amount:,.0f}"

with col1:
    st.metric("Transactions Today", count_between(raw_df, today, today))
    st.metric("Amount Today", format_amount(sum_between(df, today, today)))

with col2:
    st.metric("Last 7 Days (Txn)", count_between(raw_df, last_7_days, today))
    st.metric("Last 7 Days (Amount)", format_amount(sum_between(df, last_7_days, today)))

with col3:
    st.metric("This Week (Txn)", count_between(raw_df, start_of_week, today))
    st.metric("This Week (Amount)", format_amount(sum_between(df, start_of_week, today)))

with col4:
    st.metric("This Month (Txn)", count_between(raw_df, start_of_month, today))
    st.metric("This Month (Amount)", format_amount(sum_between(df, start_of_month, today)))

with col5:
    st.metric("This Year (Txn)", count_between(raw_df, start_of_year, today))
    st.metric("This Year (Amount)", format_amount(sum_between(df, start_of_year, today)))

st.markdown("---")

# ==============================================================================
# 7. DAILY SALES TREND (VISUALIZATION IMPROVEMENT)
# ==============================================================================

st.header("📈 Daily Sales Trend")

daily_summary = (
    df_filtered.groupby("date")["amount"]
      .sum()
      .reset_index()
      .sort_values("date")
)

# Use Altair for a more professional and interactive chart
chart = alt.Chart(daily_summary).mark_line(point=True).encode(
    x=alt.X("date", title="Date"),
    y=alt.Y("amount", title=f"Total Amount ({CURRENCY_SYMBOL})"),
    tooltip=[alt.Tooltip("date"), alt.Tooltip("amount", format=".2f")]
).properties(
    title=f"Sales Trend: {filter_start_date.strftime('%b %d, %Y')} to {filter_end_date.strftime('%b %d, %Y')}"
).interactive() # Allows zooming and panning

st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# ==============================================================================
# 8. SALES BY COMMODITY (VISUALIZATION IMPROVEMENT)
# ==============================================================================

st.header("🌾 Sales by Commodity")

commodity_summary = (
    df_filtered.groupby("commodities")["amount"]
      .sum()
      .reset_index()
      .sort_values("amount", ascending=False)
)

# Display top N commodities with a slider filter
top_n = st.slider("Show Top N Commodities", 5, len(commodity_summary), 10)
top_commodity_summary = commodity_summary.head(top_n)

col_chart, col_data = st.columns([2, 1])

with col_chart:
    # Use Altair Bar Chart
    bar_chart = alt.Chart(top_commodity_summary).mark_bar().encode(
        x=alt.X("amount", title=f"Total Amount ({CURRENCY_SYMBOL})"),
        y=alt.Y("commodities", sort="-x", title="Commodity"),
        tooltip=["commodities", alt.Tooltip("amount", format=".0f")]
    ).properties(
        title=f"Top {top_n} Commodities by Sales Amount"
    )
    st.altair_chart(bar_chart, use_container_width=True)

with col_data:
    # Use st.dataframe with formatting
    st.subheader("Summary Table")
    st.dataframe(
        top_commodity_summary.style.format({"amount": f"{CURRENCY_SYMBOL} {{:,.0f}}"})
            .set_caption(f"Top {top_n} Commodities"),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==============================================================================
# 9. RAW DATA VIEWER
# ==============================================================================

st.header("🔍 Filtered Data Explorer")

# Allow users to choose which dataset to view
data_choice = st.radio(
    "Select Data View:",
    options=["Exploded Data (Commodity Level)", "Raw Transaction Data (Initial Rows)"]
)

if data_choice == "Exploded Data (Commodity Level)":
    df_display = df_filtered.sort_values("date", ascending=False)
    # Re-apply formatting for display
    st.dataframe(
        df_display.style.format({"amount": f"{CURRENCY_SYMBOL} {{:,.0f}}"}),
        use_container_width=True
    )
else:
    # Filter the raw data just by date for display
    raw_df_display = raw_df_filtered.sort_values("date", ascending=False)
    st.dataframe(
        raw_df_display.style.format({"amount": f"{CURRENCY_SYMBOL} {{:,.0f}}"}),
        use_container_width=True
    )
