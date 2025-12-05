import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from report_utils import load_data, explode_commodities, CURRENCY_CODE, CURRENCY_FORMAT, metric_format

# -----------------------------------------------------------------------------
# DEFINE THE SHEET URL FOR FINANCE HISTORICALS (GID=156572199)
# -----------------------------------------------------------------------------
# Spreadsheet ID: 1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q
FINANCE_DATA_URL = "https://docs.google.com/spreadsheets/d/1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q/gviz/tq?tqx=out:csv&gid=156572199"

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.title("Finance Historicals: Full Trend Analysis")
st.markdown("Overview of all recorded sales, grouped by month and broken down by individual commodity.")
st.markdown("---")

# -----------------------------------------------------------------------------
# DATA LOADING 
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Loading Finance Historical Data...")
def cached_load_finance_data():
    """
    Load data using the direct CSV URL method.
    """
    df = load_data(sheet_url=FINANCE_DATA_URL)
    
    if df.empty:
        st.error("No data found or data loading failed for the 'Finance Historical' tab.")
        return df
    
    # We now expect 'amount_pkr', 'commodities_list', and 'month_str' after load_data runs
    if 'month_str' not in df.columns or 'amount_pkr' not in df.columns or 'commodities_list' not in df.columns:
        st.error("Error: Standardized columns (Month, Amount, Type) not found. Check the sheet headers.")
        return pd.DataFrame() 
        
    # Calculate Month_DT here, using the 'month_str' column cleaned in report_utils
    df['Month_DT'] = pd.to_datetime(df['month_str'], format='%m-%Y', errors='coerce')
    df = df.dropna(subset=['Month_DT']).sort_values('Month_DT')
    
    return df.dropna(how='all')


df_raw_finance = cached_load_finance_data()

if df_raw_finance.empty:
    st.stop()

# --- EXPLODE THE COMMODITIES (Separates grouped items) ---
exploded_df_finance = explode_commodities(df_raw_finance)

# Ensure the exploded DataFrame has the Month_DT column for grouping
if 'Month_DT' not in exploded_df_finance.columns:
     exploded_df_finance = exploded_df_finance.merge(
         df_raw_finance[['month_str', 'Month_DT']], 
         on='month_str', 
         how='left'
     ).drop_duplicates()

# -----------------------------------------------------------------------------
# GLOBAL METRICS AND KPIS (All Time)
# -----------------------------------------------------------------------------

st.header("Financial Overview (All Time)")

total_sales = df_raw_finance['amount_pkr'].sum() 
transaction_count = len(df_raw_finance)
unique_commodities = exploded_df_finance['commodity'].nunique() 

# Calculate Highest Selling Month
monthly_sales_summary = df_raw_finance.groupby('Month_DT')['amount_pkr'].sum().reset_index()
if not monthly_sales_summary.empty:
    highest_month_row = monthly_sales_summary.loc[monthly_sales_summary['amount_pkr'].idxmax()]
    highest_selling_month = highest_month_row['Month_DT'].strftime('%b %Y')
    highest_selling_amount = metric_format(highest_month_row['amount_pkr'])
else:
    highest_selling_month = "N/A"
    highest_selling_amount = "N/A"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("**Total Sales (All Time)**", 
              metric_format(total_sales))
with col2:
    st.metric("**Total Transactions**", transaction_count)
with col3:
    st.metric("**Unique Commodities Sold**", unique_commodities)
with col4:
    st.metric(
        "**Highest Selling Month**", 
        highest_selling_month,
        delta=highest_selling_amount,
        delta_color="normal"
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# TIME SERIES CHART (All Months)
# -----------------------------------------------------------------------------

st.header("Total Sales Trend Over Time")

chart = (
    alt.Chart(monthly_sales_summary)
    .mark_line(point=True, color='#006B3F')
    .encode(
        x=alt.X('Month_DT:T', title='Month', axis=alt.Axis(format='%Y-%m')),
        y=alt.Y('amount_pkr:Q', title=f"Total Sales ({CURRENCY_CODE})", 
                axis=alt.Axis(format=CURRENCY_FORMAT)),
        tooltip=[
            alt.Tooltip('Month_DT:T', title='Month', format='%Y-%m'),
            alt.Tooltip('amount_pkr:Q', title='Sales Amount', format=CURRENCY_FORMAT),
        ],
    )
    .properties(title="Total Monthly Sales Trend (All Available Data)")
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# COMMODITY MIX & SEASONALITY 
# -----------------------------------------------------------------------------

st.header("Commodity Performance & Seasonality")

# Group by the separated 'commodity' column for overall ranking
commodity_summary = (
    exploded_df_finance.groupby('commodity')['gross_amount_per_commodity']
    .sum()
    .reset_index()
    .rename(columns={'gross_amount_per_commodity': 'Total_Amount'})
    .sort_values('Total_Amount', ascending=False)
)

col_chart, col_table = st.columns([3, 2])

with col_table:
    st.subheader("Commodity Ranking (All Time)")
    # Format the table for display
    styled_df = commodity_summary.style.format(
        {"Total_Amount": f"{CURRENCY_CODE} {{:,.0f}}"}
    ).hide(axis="index")
    
    st.dataframe(styled_df, use_container_width=True, height=500)

with col_chart:
    st.subheader("Top 5 Commodity Monthly Trends")
    
    # 1. Identify Top 5 Commodities for charting
    top_5_commodities = commodity_summary['commodity'].head(5).tolist()

    # 2. Filter exploded data to include only Top 5
    seasonality_df = exploded_df_finance[
        exploded_df_finance['commodity'].isin(top_5_commodities)
    ].copy()
    
    # 3. Group by Month and Commodity
    monthly_commodity_sales = (
        seasonality_df.groupby(['Month_DT', 'commodity'])['gross_amount_per_commodity']
        .sum()
        .reset_index()
        .rename(columns={'gross_amount_per_commodity': 'Monthly_Sales'})
    )

    if not monthly_commodity_sales.empty:
        seasonality_chart = (
            alt.Chart(monthly_commodity_sales)
            .mark_line(point=True)
            .encode(
                x=alt.X('Month_DT:T', title='Month', axis=alt.Axis(format='%Y-%m')),
                y=alt.Y('Monthly_Sales:Q', title=f"Sales ({CURRENCY_CODE})", 
                        axis=alt.Axis(format=CURRENCY_FORMAT)),
                color='commodity:N',
                tooltip=[
                    alt.Tooltip('Month_DT:T', title='Month', format='%Y-%m'),
                    alt.Tooltip('commodity:N', title='Commodity'),
                    alt.Tooltip('Monthly_Sales:Q', title='Sales', format=CURRENCY_FORMAT),
                ],
            )
            .properties(title="Monthly Sales Trend for Top 5 Commodities")
            .interactive()
        )
        st.altair_chart(seasonality_chart, use_container_width=True)
    else:
        st.info("Not enough data to chart monthly commodity trends.")

st.markdown("---")

# -----------------------------------------------------------------------------
# RAW DATA EXPLORER 
# -----------------------------------------------------------------------------

st.header("Raw Transaction Data Explorer")
st.caption("Showing all transactions from the 'Finance Historical' spreadsheet row.")

# Prepare the RAW data for display
df_display = df_raw_finance[['month_str', 'amount_pkr', 'commodities_list']].copy()
df_display.rename(
    columns={
        'month_str': 'Month',
        'amount_pkr': f"Gross Amount ({CURRENCY_CODE})", 
        'commodities_list': 'Commodities (Grouped)'
    }, 
    inplace=True
)

styled_df = df_display.style.format(
    {f"Gross Amount ({CURRENCY_CODE})": f"{CURRENCY_CODE} {{:,.0f}}"}
).hide(axis="index")

st.dataframe(styled_df, use_container_width=True, height=500)
