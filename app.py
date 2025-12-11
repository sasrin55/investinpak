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

# --- CONSTANTS ---
CUSTOMER_TYPE_COL = "customer_type"
GROSS_AMOUNT_PER_COMMODITY_COL = "gross_amount_per_commodity"
COMMODITIES_COL = "commodities_list"
AMOUNT_COL = "amount_pkr"
CUSTOMER_COL = "customer_name"
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f" # For formatting PKR values
AVG_FORMAT = ",.2f" # For formatting average spend

# ---------- CONFIG ----------

# EXPLICIT CSV EXPORT LINK for 'Master for SalesOps' (GID=1105756916)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q/gviz/tq?tqx=out:csv&gid=1105756916"


# ---------- DATA TRANSFORMATIONS (Explode Logic) ----------

def explode_commodities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe and ensures each row represents one commodity, 
    calculating the proportional amount for it.
    """
    df_temp = df.copy()
    
    # 1. Rename columns to the names expected by this function
    df_temp.rename(
        columns={
            "commodity": COMMODITIES_COL,
            "amount": AMOUNT_COL,
            "customer": CUSTOMER_COL,
        },
        inplace=True,
    )
    
    original_cols = list(df_temp.columns)

    required_cols = [COMMODITIES_COL, AMOUNT_COL, CUSTOMER_TYPE_COL]
    if not all(col in df_temp.columns for col in required_cols):
        return pd.DataFrame(columns=original_cols + ["commodity", GROSS_AMOUNT_PER_COMMODITY_COL])

    df_temp[COMMODITIES_COL] = df_temp[COMMODITIES_COL].fillna("")

    # --- FIX: ASSUME SIMPLE LISTS AND SPLIT AMOUNT EQUALLY ---
    
    # 2. Split the list by comma and explode
    df_temp["commodity_items"] = df_temp[COMMODITIES_COL].str.split(",").apply(
        lambda x: [item.strip() for item in x if str(item).strip()]
    )
    df_exploded = df_temp.explode("commodity_items")
    
    # 3. Assign commodity name (The item itself)
    df_exploded["commodity"] = df_exploded["commodity_items"].astype(str).str.strip()
    
    # 4. Calculate proportional amount (Split total transaction amount equally)
    
    # Count how many commodities were in the original transaction
    count_items_per_txn = df_exploded.groupby(df_exploded.index)["commodity_items"].transform("count")
    
    df_exploded[GROSS_AMOUNT_PER_COMMODITY_COL] = np.where(
         count_items_per_txn > 0,
         df_exploded[AMOUNT_COL] / count_items_per_txn, # Split total amount equally
         0,
    )

    # 5. Filter out empty commodities and keep necessary columns
    df_final = df_exploded[df_exploded["commodity"] != ""].copy()

    # Define final columns to keep
    final_cols = [
        "date", CUSTOMER_COL, CUSTOMER_TYPE_COL, "duration", 
        "commodity", GROSS_AMOUNT_PER_COMMODITY_COL, AMOUNT_COL
    ]

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
    # Expecting clean headers: date, customer, customer_type, commodity, amount, duration

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
    
    # --- FIX: UNIFY CUSTOMER TYPE ---
    if CUSTOMER_TYPE_COL in df.columns:
        df[CUSTOMER_TYPE_COL] = df[CUSTOMER_TYPE_COL].str.strip().str.title()

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


# --- Date range calculation (Type Error Fix) ---
valid_dates = raw_df["date"].dropna()

if valid_dates.empty:
    min_data_date = date(2020, 1, 1)
    max_data_date = date.today()
else:
    min_data_date = valid_dates.min()
    max_data_date = valid_dates.max()


st.write("---")

# ---------- KPI BLOCKS ----------

total_amount = raw_df["amount"].sum() if "amount" in raw_df.columns else 0
total_txns = len(raw_df)
unique_customers = raw_df["customer"].nunique() if "customer" in raw_df.columns else 0

st.header("Sales Performance Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales (PKR)", f"{total_amount:{CURRENCY_FORMAT}}")
col2.metric("Total Transactions", f"{total_txns:,}")
col3.metric("Unique Customers (by Phone)", f"{unique_customers:,}")


# ---------- CUSTOMER RETENTION METRIC ----------

if CUSTOMER_TYPE_COL in raw_df.columns and AMOUNT_COL in raw_df.columns:
    st.header("Customer Lifetime Value & Retention")
    
    # 1. Total unique customers (based on phone number)
    total_unique_customers = raw_df[CUSTOMER_COL].nunique()
    
    # 2. Customers classified as 'Return'
    returned_customers = raw_df[raw_df[CUSTOMER_TYPE_COL] == "Return"][CUSTOMER_COL].nunique()
    
    # --- Calculate Average Transaction Values (using raw_df for transaction value) ---
    
    # Group raw transactions by customer type and calculate average amount
    avg_txn_by_type = (
        raw_df.groupby(CUSTOMER_TYPE_COL)[AMOUNT_COL]
        .mean()
        .rename("Avg Txn Value")
    )
    
    avg_new = avg_txn_by_type.get("New", 0)
    avg_return = avg_txn_by_type.get("Return", 0)
    
    # Calculate difference
    avg_spend_increase = avg_return - avg_new
    
    # Determine the indicator color
    if avg_spend_increase > 0:
        delta_color = "inverse" # Green delta for increase
    elif avg_spend_increase < 0:
        delta_color = "off" # Red delta for decrease
    else:
        delta_color = "normal"
    
    ret_col1, ret_col2, ret_col3 = st.columns(3)
    
    # Metric 1: Total Unique Customers
    ret_col1.metric("Total Unique Customers", f"{total_unique_customers:,}")
    
    # Metric 2: Customers That Returned
    ret_col2.metric("Customers That Returned", f"{returned_customers:,}")
    
    # Metric 3: Average Spend Increase (Retention Value)
    ret_col3.metric(
        "Avg Spend Increase (Return vs New)", 
        f"{avg_return:{AVG_FORMAT}}", 
        delta=f"{avg_spend_increase:{AVG_FORMAT}}",
        delta_color=delta_color,
        help="The average transaction value for a 'Return' customer vs. their initial 'New' average."
    )
    
    st.markdown("---")
    
    # Display detailed retention breakdown table (Original logic)
    retention_sales = (
        exploded_df.groupby(CUSTOMER_TYPE_COL)[GROSS_AMOUNT_PER_COMMODITY_COL]
        .sum()
        .rename("Sales Amount")
    )

    retention_customers = (
        exploded_df.groupby(CUSTOMER_TYPE_COL)[CUSTOMER_COL]
        .nunique()
        .rename("Unique Customers")
    )
    
    retention_df = pd.concat([retention_sales, retention_customers], axis=1).fillna(0)
    
    if not retention_df.empty:
        total_customers_in_table = retention_df["Unique Customers"].sum()
        retention_df["Customer Share (%)"] = (retention_df["Unique Customers"] / total_customers_in_table * 100).round(1)

        st.subheader("Sales and Customer Breakdown by Type")
        st.dataframe(retention_df.style.format({
            "Sales Amount": f"{CURRENCY_CODE} {{:{CURRENCY_FORMAT}}}",
            "Unique Customers": "{:,.0f}",
            "Customer Share (%)": "{:,.1f}%"
        }), use_container_width=True)

        st.subheader("Customer Type Sales Share")
        sales_share_chart = alt.Chart(retention_df.reset_index()).mark_arc().encode(
            theta=alt.Theta(field="Sales Amount", type="quantitative"),
            color=alt.Color(field=CUSTOMER_TYPE_COL, type="nominal", title="Customer Type"),
            order=alt.Order(field="Sales Amount", sort="descending"),
            tooltip=[CUSTOMER_TYPE_COL, "Sales Amount", "Unique Customers"]
        ).properties(title="Sales Amount Split by Customer Type")

        st.altair_chart(sales_share_chart, use_container_width=True)
    
else:
    st.info("Customer retention analysis requires the 'customer_type' and 'amount' columns.")

st.markdown("---")

# ---------- SALES BY COMMODITY (Chart Format) ----------

if "commodity" in exploded_df.columns:
    st.header("Sales by Commodity")

    commodity_sales_df = (
        exploded_df.groupby("commodity")[GROSS_AMOUNT_PER_COMMODITY_COL] 
        .sum()
        .rename("Total Sales")
        .reset_index()
    )

    # Filter out the 'nan' commodity group
    commodity_sales_df = commodity_sales_df[commodity_sales_df['commodity'].notna()].head(15)

    if not commodity_sales_df.empty:
        st.subheader("Top Selling Commodities by Sales")

        # Create Altair Bar Chart
        commodity_chart = alt.Chart(commodity_sales_df).mark_bar().encode(
            x=alt.X('Total Sales', title=f'Total Sales ({CURRENCY_CODE})', axis=alt.Axis(format=CURRENCY_FORMAT)),
            y=alt.Y('commodity', sort='-x', title='Commodity'),
            tooltip=['commodity', alt.Tooltip('Total Sales', format=f"{CURRENCY_CODE} {CURRENCY_FORMAT}", title='Total Sales')]
        ).properties(title="Commodity Sales Volume").interactive()
        
        st.altair_chart(commodity_chart, use_container_width=True)
    else:
        st.info("No valid commodity data found to plot.")


# ---------- RETURN CUSTOMERS BY COMMODITY ----------

if CUSTOMER_TYPE_COL in exploded_df.columns:
    st.header("Return Customer Loyalty by Commodity")
    st.caption("Count of unique 'Return' customers who purchased each commodity.")

    # 1. Filter for Return Customers (using the unified 'Return' type)
    return_customers_df = exploded_df[
        exploded_df[CUSTOMER_TYPE_COL] == "Return"
    ]
    
    if not return_customers_df.empty:
        # 2. Group by commodity and count unique phone numbers (customers)
        return_customer_loyalty_df = (
            return_customers_df
            .groupby("commodity")[CUSTOMER_COL]
            .nunique()
            .rename("Unique Return Customers")
            .reset_index()
            .sort_values(by="Unique Return Customers", ascending=False)
            .head(15)
        )
        
        st.subheader("Commodities with Most Unique Return Buyers")
        st.dataframe(return_customer_loyalty_df, use_container_width=True, hide_index=True)
        
        # Create Altair Bar Chart for Return Customers
        return_chart = alt.Chart(return_customer_loyalty_df).mark_bar().encode(
            x=alt.X('Unique Return Customers', title='Unique Return Customers'),
            y=alt.Y('commodity', sort='-x', title='Commodity'),
            tooltip=['commodity', 'Unique Return Customers']
        ).properties(title="Return Customer Count by Commodity").interactive()
        
        st.altair_chart(return_chart, use_container_width=True)
    else:
        st.info("No 'Return' customer data found to analyze loyalty.")

st.markdown("---")

# ---------- COMMODITY SEASONALITY ANALYSIS (Line Chart) ----------

if "date" in exploded_df.columns and "commodity" in exploded_df.columns:
    st.header("Commodity Seasonality Analysis")
    st.caption("Monthly sales trends to identify peak selling seasons.")
    
    # Ensure 'date' is a datetime object for resampling (Fix for AttributeError)
    seasonality_df = exploded_df.copy()
    seasonality_df['date'] = pd.to_datetime(seasonality_df['date'])

    # Aggregate sales by month and commodity
    seasonality_summary = (
        seasonality_df.set_index('date')
        .groupby(['commodity'])
        .resample('ME')[GROSS_AMOUNT_PER_COMMODITY_COL] # 'ME' means Month End
        .sum()
        .rename("Total Sales")
        .reset_index()
    )
    
    # Filter out 'nan' commodity type
    seasonality_summary = seasonality_summary[seasonality_summary['commodity'].notna()]
    
    if not seasonality_summary.empty:
        
        commodity_list = sorted(seasonality_summary["commodity"].unique().tolist())
        selected_commodity = st.selectbox(
            "Select Commodity for Trend Analysis", options=commodity_list, index=0
        )

        seasonality_chart_data = seasonality_summary[
            seasonality_summary["commodity"] == selected_commodity
        ]
        
        if not seasonality_chart_data.empty:
            # Create Altair Line Chart for Seasonality
            seasonality_chart = alt.Chart(seasonality_chart_data).mark_line(point=True).encode(
                x=alt.X('date', title='Month', axis=alt.Axis(format="%Y-%m")),
                y=alt.Y('Total Sales', title=f'Total Sales ({CURRENCY_CODE})', axis=alt.Axis(format=CURRENCY_FORMAT)),
                tooltip=[
                    alt.Tooltip('date', title='Month', format="%Y-%m"),
                    alt.Tooltip('Total Sales', title='Sales Amount', format=f"{CURRENCY_CODE} {CURRENCY_FORMAT}")
                ]
            ).properties(
                title=f"Monthly Sales Trend for {selected_commodity}"
            ).interactive()

            st.altair_chart(seasonality_chart, use_container_width=True)
        else:
            st.info(f"No seasonality data found for {selected_commodity}.")

    else:
        st.info("Not enough data to analyze seasonality.")


st.markdown("---")
st.subheader("Reporting Period:")
st.caption(f"Data available from {min_data_date} to {max_data_date}")
