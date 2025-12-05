import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
# Import load_data and explode_commodities from the report_utils.py file
from report_utils import load_data, explode_commodities, CURRENCY_CODE, CURRENCY_FORMAT 

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.title("Finance Historicals")
st.markdown("Detailed breakdown of monthly sales transactions from the 'Finance Historical' sheet, **separated by commodity**.")
st.markdown("---")

# -----------------------------------------------------------------------------
# DATA LOADING (Pulls from the 'Finance Historical' tab and RENAMES columns)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Loading Finance Historical Data...")
def cached_load_finance_data():
    """
    Load and prepare data specifically from the 'Finance Historical' Google Sheets tab.
    This function handles the column name mapping necessary for the explode_commodities utility.
    """
    # Load data from the specific sheet name. Columns will be lowercased by report_utils.py.
    df = load_data(sheet_name="Finance Historical")
    
    if df.empty:
        return df
    
    # 1. Map columns from the Finance Historical sheet to the names expected by the pipeline:
    # 'month' is the original Month column
    # 'amount' is the original Amount column
    # 'type' is the original Type column
    df.rename(columns={
        'amount': 'amount_pkr_raw', # Store original amount before explosion
        'type': 'commodities_list', # This column contains the grouped items
        'month': 'month_str' # Store original month string
    }, inplace=True)
    
    # Ensure the new columns exist after renaming (checking common errors)
    if 'month_str' not in df.columns or 'amount_pkr_raw' not in df.columns or 'commodities_list' not in df.columns:
        st.error("Error: Could not find expected columns (Month, Amount, Type) in the 'Finance Historical' tab after standardization.")
        return pd.DataFrame()

    # 2. Re-clean and ensure types are correct for the explosion process
    df['amount_pkr'] = pd.to_numeric(df['amount_pkr_raw'], errors='coerce').fillna(0)
    df['Month_DT'] = pd.to_datetime(df['month_str'], format='%m-%Y', errors='coerce')
    
    # Drop rows that failed conversion or are completely empty
    df = df.dropna(subset=['Month_DT', 'amount_pkr']).sort_values('Month_DT')
        
    return df.dropna(how='all')


df_raw_finance = cached_load_finance_data()

if df_raw_finance.empty:
    st.error("No data found or data loading failed for the 'Finance Historical' tab.")
    st.stop()

# --- EXPLODE THE COMMODITIES (Separates grouped items) ---
# Now explode_commodities receives the required 'commodities_list' and 'amount_pkr'
exploded_df_finance = explode_commodities(df_raw_finance)

st.caption(f"Data period: {exploded_df_finance['Month_DT'].min().strftime('%B %Y')} to {exploded_df_finance['Month_DT'].max().strftime('%B %Y')}")

# -----------------------------------------------------------------------------
# FILTERS
# -----------------------------------------------------------------------------

unique_months = sorted(df_raw_finance['month_str'].unique(), key=lambda x: datetime.strptime(x, '%m-%Y'), reverse=True)
selected_month = st.selectbox("Select Month for Detail View:", options=unique_months, index=0)

# Filter the EXPLODED dataframe based on the selected month
df_filtered_exploded = exploded_df_finance[exploded_df_finance['month_str'] == selected_month].copy()
# Filter the RAW dataframe for KPI calculations that need total transaction counts
df_filtered_raw = df_raw_finance[df_raw_finance['month_str'] == selected_month].copy() 

# -----------------------------------------------------------------------------
# METRICS AND KPIS 
# -----------------------------------------------------------------------------

st.header("Monthly Summary")
total_sales = df_filtered_raw['amount_pkr'].sum() 
transaction_count = len(df_filtered_raw)
unique_commodities = df_filtered_exploded['commodity'].nunique() 

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(f"**Total Sales ({selected_month})**", 
              f"{CURRENCY_CODE} {total_sales:,.0f}")
with col2:
    st.metric("**Total Transactions**", transaction_count)
with col3:
    st.metric("**Unique Commodities Sold**", unique_commodities)

st.markdown("---")

# -----------------------------------------------------------------------------
# TIME SERIES CHART (All Months)
# -----------------------------------------------------------------------------

st.header("Sales Trend Over Time")

monthly_sales = df_raw_finance.groupby('Month_DT')['amount_pkr'].sum().reset_index()
monthly_sales['Month_Label'] = monthly_sales['Month_DT'].dt.strftime('%Y-%m')

chart = (
    alt.Chart(monthly_sales)
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
    .properties(title="Total Monthly Sales Trend")
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# COMMODITY MIX (Monthly) 
# -----------------------------------------------------------------------------

st.header(f"Commodity Breakdown ({selected_month})")

# Group by the separated 'commodity' column
commodity_summary = (
    df_filtered_exploded.groupby('commodity')['gross_amount_per_commodity']
    .sum()
    .reset_index()
    .rename(columns={'gross_amount_per_commodity': 'Total_Amount'})
    .sort_values('Total_Amount', ascending=False)
)

col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.subheader("Sales by Individual Commodity")
    
    if not commodity_summary.empty:
        pie_chart = (
            alt.Chart(commodity_summary)
            .mark_arc(outerRadius=120)
            .encode(
                theta=alt.Theta("Total_Amount", stack=True),
                color=alt.Color("commodity", title="Commodity"), 
                order=alt.Order("Total_Amount", sort="descending"),
                tooltip=["commodity", 
                         alt.Tooltip("Total_Amount", title="Sales Amount", format=CURRENCY_FORMAT)],
            )
            .properties(title=f"Sales Distribution for {selected_month}")
        )
        st.altair_chart(pie_chart, use_container_width=True)
    else:
        st.info(f"No commodity data found for {selected_month}.")

with col_table:
    st.subheader("Detail Table")
    styled_df = commodity_summary.style.format(
        {"Total_Amount": f"{CURRENCY_CODE} {{:,.0f}}"}
    ).hide(axis="index")
    
    st.dataframe(styled_df, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# RAW DATA EXPLORER (Monthly)
# -----------------------------------------------------------------------------

st.header(f"Raw Transaction Data Explorer ({selected_month})")
st.caption("This table shows the raw, un-exploded transactions from the spreadsheet row.")

# Prepare the RAW data for display
df_display = df_filtered_raw[['month_str', 'amount_pkr', 'commodities_list']].copy()
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

st.dataframe(styled_df, use_container_width=True)
