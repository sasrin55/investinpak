import pandas as pd
import streamlit as st
from datetime import date, timedelta
import altair as alt
import numpy as np
import re

# ==============================================================================
# 1. CONFIGURATION AND INITIAL SETUP
# ==============================================================================

# --- SETTINGS ---
SHEET_ID = "1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q"
TAB_NAME = "Master"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={TAB_NAME}"
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f" 

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

    # Convert date and amount, dropping rows with invalid dates afterward
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["amount_pkr"] = pd.to_numeric(df["amount_pkr"], errors="coerce").fillna(0)
    df["txn_type"] = df["txn_type"].astype(str).str.strip().str.title()
    df["customer_name"] = df["customer_name"].astype(str).str.strip()
    df = df.dropna(subset=["date"])

    return df


def explode_commodities(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits transaction rows into one row per commodity, fairly allocating the total amount,
    with ENHANCED cleaning for de-duplication and non-commodity removal.
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
        if not name:
            return None
        
        # 1. Convert to lowercase and strip punctuation/extra spaces
        name = name.lower().strip()
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        if not name:
            return None
            
        # Hardcoded list of items to REMOVE (non-commodities or internal tracking)
        NON_COMMODITY_KEYWORDS = [
            'unknown', 'not confirm yet', 'discussion', 'contact sale', 
            'live market', 'data', 'group', 'g', 'um', 'l m', 'lm', 
            'fruits', 'fruit', 'vegetables', 'vegetable' 
        ]
        
        if any(keyword in name for keyword in NON_COMMODITY_KEYWORDS):
            return None

        # Consolidation mapping for common misspellings/variants
        mapping = {
            'cotton': 'Cotton', 'coton': 'Cotton', 'cottonmillet': 'Cotton Millet',
            'paddy': 'Paddy', 'padd': 'Paddy',
            'wheat': 'Wheat', 'wheatandpaddy': 'Wheat & Paddy',
            'edibleoil': 'Edible Oil',
            'fertilizer': 'Fertilizer', 
            'pulses': 'Pulses', 'daal': 'Pulses',
            'bajra': 'Bajra',
            'livestock': 'Livestock', 
            'sesame': 'Sesame', 
            'sugar': 'Sugar', 'sugarwheat': 'Sugar + Wheat',
            'mustard': 'Mustard', 'mustrad': 'Mustard',
            'kiryana': 'Kiryana',
            'dryfruits': 'Dry Fruits',
            'spices': 'Spices',
            'rice': 'Rice',
            'maize': 'Maize',
            'dates': 'Dates'
        }
        
        # Apply mapping or title case if no map found
        for key, value in mapping.items():
            if name == key:
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
start_of_year = date(today.year, 1, 1)
last_30_days_start = today - timedelta(days=29) # 30 days including today


# --- FIX: Define universally safe dates for widget initialization ---
safe_min_date = date(2020, 1, 1) # A fixed, safe minimum date
safe_max_date = today 

# We still need the true min/max data dates for filtering default
raw_min = raw_df["date"].min()
raw_max = raw_df["date"].max()

min_data_date = raw_min if pd.notna(raw_min) else start_of_year
max_data_date = raw_max if pd.notna(raw_max) else today

# Convert to python date objects for use in widget value argument
if isinstance(min_data_date, pd.Timestamp): min_data_date = min_data_date.date()
if isinstance(max_data_date, pd.Timestamp): max_data_date = max_data_date.date()

if min_data_date > max_data_date:
    min_data_date = max_data_date


## 📊 Top-Level Filters
st.subheader("Reporting Filters")
filter_cols = st.columns([1, 4])

# 1. Date range selector 
with filter_cols[0]:
    # Use safe hardcoded dates for min/max boundary, but use the sensible data min for default value
    date_range = st.date_input(
        "Reporting Period",
        value=(min_data_date, today), 
        min_value=safe_min_date,      
        max_value=safe_max_date,      
        key="top_date_filter"
    )

filter_start_date = min(date_range)
filter_end_date = max(date_range) if len(date_range) == 2 else date_range[0]

# Apply filter: Use the filtered dates from the widget to filter the actual data
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

def metric_format(value):
    return f"{CURRENCY_CODE} {value:,.0f}"

## KPI Section 1: Detailed Transaction Summaries

def create_summary_table(df, period_start, period_end, title):
    """Generates a detailed table showing Customer, Commodity, and Amount for a period."""
    
    AMOUNT_COL_NAME = f"Amount ({CURRENCY_CODE})"

    # Filter the exploded data for the specific period
    period_mask = (df["date"] >= period_start) & (df["date"] <= period_end)
    summary_df = exploded_df.loc[period_mask].copy()
    
    # Group by customer and commodity to summarize the transaction amount
    summary_df = summary_df.groupby(["customer_name", "commodity"])["amount_per_commodity"].sum().reset_index()
    
    summary_df = summary_df.rename(columns={
        "customer_name": "Customer",
        "commodity": "Commodity",
        "amount_per_commodity": AMOUNT_COL_NAME # Use the calculated name
    })
    
    # Sort and format the output
    summary_df = summary_df.sort_values(AMOUNT_COL_NAME, ascending=False)
    
    # Apply styling for better presentation
    styled_df = summary_df.style.format({
        AMOUNT_COL_NAME: f"{CURRENCY_CODE} {{:,.0f}}",
    })

    st.subheader(f"{title} (Total Transactions: {count_transactions(raw_df, period_start, period_end)})")
    
    # Use st.container to ensure the table sits nicely in the column
    with st.container(border=True):
        st.dataframe(styled_df, use_container_width=True, hide_index=True)


col_today, col_30days, col_ytd_tables = st.columns(3)

# 1. SALES TODAY
with col_today:
    # Today Amount Metric
    st.subheader("Today's Total Sales")
    today_amount = sum_between(raw_df, today, today)
    st.metric("**Total Today's Sales**", metric_format(today_amount))
    st.markdown("---")
    
    create_summary_table(exploded_df, today, today, "Today's Sales Breakdown")
    
# 2. LAST 30 DAYS
with col_30days:
    # Last 30 Days Amount Metric
    st.subheader("Last 30 Days Total Sales")
    last_30_amount = sum_between(raw_df, last_30_days_start, today)
    st.metric("**Total Last 30 Days Sales**", metric_format(last_30_amount))
    st.markdown("---")
    
    create_summary_table(exploded_df, last_30_days_start, today, "Last 30 Days Sales Breakdown")

# 3. YTD SUMMARY & TABLE 
with col_ytd_tables:
    # YTD Amount Metric
    st.subheader("Year-To-Date Total Sales")
    ytd_amount = sum_between(raw_df, start_of_year, today)
    st.metric("**Total YTD Sales**", metric_format(ytd_amount))
    st.markdown("---")
    
    # YTD Sales Breakdown Table
    create_summary_table(exploded_df, start_of_year, today, "YTD Sales Breakdown")
    

st.markdown("---")

# ==============================================================================
# 5. COMMODITY LOYALTY (New vs. Repeat Count)
# ==============================================================================

st.header("Commodity Loyalty Analysis")
st.markdown("Analyzes commodity performance based on the count of unique **New** vs. **Repeat** buyers across the entire dataset.")

# 1. Calculate transaction count per customer per commodity
txn_count_by_customer_commodity = (
    exploded_df.groupby(["customer_name", "commodity"])["date"].nunique().reset_index()
)
txn_count_by_customer_commodity.rename(columns={"date": "Total Transactions"}, inplace=True)

# 2. Determine Buyer Type (New vs. Repeat) for each customer-commodity pair
# A buyer is 'Repeat' if their Transaction Count for that commodity > 1.
# A buyer is 'New' if their Transaction Count for that commodity == 1.

txn_count_by_customer_commodity["Buyer Type"] = np.where(
    txn_count_by_customer_commodity["Transaction Count"] > 1, 
    "Repeat", 
    "New"
)

# 3. Group by Commodity and Buyer Type, then count unique customers
loyalty_summary = (
    txn_count_by_customer_commodity.groupby(["commodity", "Buyer Type"])
    .size()
    .unstack(fill_value=0) # Pivot 'New' and 'Repeat' into columns
)

# 4. Ensure both columns exist for consistency
if 'New' not in loyalty_summary.columns:
    loyalty_summary['New'] = 0
if 'Repeat' not in loyalty_summary.columns:
    loyalty_summary['Repeat'] = 0

loyalty_summary = loyalty_summary[['Repeat', 'New']] # Order columns

# 5. Calculate Total Buyers (unique customers)
loyalty_summary['Total Buyers'] = loyalty_summary['Repeat'] + loyalty_summary['New']

# 6. Final cleanup and sorting by Repeat count
loyalty_summary = loyalty_summary.reset_index()
loyalty_summary = loyalty_summary.sort_values("Repeat", ascending=False)
loyalty_summary.rename(columns={"Repeat": "Repeat Buyers", "New": "New Buyers"}, inplace=True)

st.subheader("Commodity Buyer Loyalty (Ranked by Repeat Buyers)")

if not loyalty_summary.empty:
    st.dataframe(loyalty_summary, use_container_width=True, hide_index=True)
else:
    st.info("No sales data available to analyze buyer loyalty.")


st.markdown("---")


# ==============================================================================
# 6. COMMODITY SEASONALITY (Functions remain the same)
# ==============================================================================

st.header("Commodity Seasonality Analysis")
st.markdown("Sales trend by month to identify peak selling seasons for each commodity.")

# Prepare data for Altair chart
seasonality_df = exploded_df.copy()
seasonality_df["Month"] = seasonality_df["date"].apply(lambda x: x.replace(day=1)) # Normalize date to month start
seasonality_df["Month_Name"] = seasonality_df["date"].apply(lambda x: x.strftime("%Y-%m"))

seasonality_summary = seasonality_df.groupby(["Month_Name", "commodity"])["amount_per_commodity"].sum().reset_index()
seasonality_summary.rename(columns={"amount_per_commodity": "Total Sales"}, inplace=True)

# Select box to pick the commodity for visualization
commodity_list = sorted(seasonality_summary["commodity"].unique().tolist())
selected_season_commodity = st.selectbox(
    "Select Commodity for Seasonality Chart",
    options=commodity_list,
    index=0 # Default to the first commodity
)

seasonality_chart_data = seasonality_summary[seasonality_summary["commodity"] == selected_season_commodity]

if not seasonality_chart_data.empty:
    # Altair chart definition
    season_chart = alt.Chart(seasonality_chart_data).mark_line(point=True).encode(
        x=alt.X("Month_Name:T", title="Month", axis=alt.Axis(format="%Y-%m")),
        y=alt.Y("Total Sales:Q", title=f"Total Sales ({CURRENCY_CODE})", axis=alt.Axis(format=CURRENCY_FORMAT)),
        tooltip=[
            alt.Tooltip("Month_Name:T", title="Month", format="%Y-%m"),
            alt.Tooltip("Total Sales:Q", title="Sales Amount", format=CURRENCY_FORMAT)
        ]
    ).properties(
        title=f"Monthly Sales Trend for {selected_season_commodity}"
    ).interactive()

    st.altair_chart(season_chart, use_container_width=True)
else:
    st.info(f"No seasonality data found for {selected_season_commodity} in the dataset.")

st.markdown("---")

# ==============================================================================
# 7. COMMODITY BREAKDOWN (Bar Chart) (Functions remain the same)
# ==============================================================================

st.header("Commodity Performance & Mix (All Data)")

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
    ).interactive() # Allow zoom/pan

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
# 8. DATA EXPLORER (Functions remain the same)
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
