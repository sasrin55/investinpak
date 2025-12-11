import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Zarai Mandi Sales Dashboard",
    layout="wide",
)

st.title("Zarai Mandi Sales Dashboard")
st.caption("Transaction and Commodity-level Sales Intelligence.")

# ---------- CONFIG ----------

SHEET_URL = "https://docs.google.com/spreadsheets/d/1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q/export?format=csv&gid=1105756916"


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
    # Expected: date, customer, customer_type, commodity, amount, duration, ...

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Clean numeric amount
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    df = df.dropna(how="all")

    return df


df = load_data()

if df.empty:
    st.error("No data loaded from Google Sheets. Check sharing permissions and the URL.")
    st.stop()

# ---------- SIDEBAR FILTERS ----------

with st.sidebar:
    st.header("Filters")

    if "date" in df.columns:
        min_date = df["date"].min()
        max_date = df["date"].max()
        start_date, end_date = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        start_date, end_date = None, None

    customer_type_filter = None
    if "customer_type" in df.columns:
        all_types = sorted(df["customer_type"].dropna().unique().tolist())
        customer_type_filter = st.multiselect(
            "Customer type",
            options=all_types,
            default=all_types,
        )

# Apply filters
df_filtered = df.copy()

if start_date and end_date and "date" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["date"] >= start_date) & (df_filtered["date"] <= end_date)
    ]

if customer_type_filter and "customer_type" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["customer_type"].isin(customer_type_filter)]

# If still empty after filters, bail early
if df_filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------- DATA PREVIEW ----------

with st.expander("Preview data (filtered)", expanded=False):
    st.dataframe(df_filtered.head(20), use_container_width=True)

# ---------- CORE KPIs ----------

total_amount = df_filtered["amount"].sum() if "amount" in df_filtered.columns else 0
total_txns = len(df_filtered)
unique_customers = (
    df_filtered["customer"].nunique() if "customer" in df_filtered.columns else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales (PKR)", f"{total_amount:,.0f}")
col2.metric("Total Transactions", f"{total_txns:,}")
col3.metric("Unique Customers", f"{unique_customers:,}")

# ---------- NEW vs RETURNING CUSTOMERS ----------

# Definition:
# - For each customer, compute their first-ever transaction date in the FULL dataset
# - In the filtered range:
#     New = first_txn_date within [start_date, end_date]
#     Returning = first_txn_date before start_date
new_customers_count = 0
returning_customers_count = 0
retention_rate = 0.0

if "customer" in df.columns and "date" in df.columns and start_date and end_date:
    # First transaction date per customer on the full dataset
    first_txn = df.groupby("customer")["date"].min()

    # Customers present in the current filtered dataset
    customers_in_range = df_filtered["customer"].unique()

    new_customers = []
    returning_customers = []

    for cust in customers_in_range:
        first_date = first_txn.get(cust)
        if first_date is None:
            continue
        if start_date <= first_date <= end_date:
            new_customers.append(cust)
        elif first_date < start_date:
            returning_customers.append(cust)

    new_customers_count = len(new_customers)
    returning_customers_count = len(returning_customers)

    total_cust_in_range = len(customers_in_range)
    retention_rate = (
        returning_customers_count / total_cust_in_range * 100
        if total_cust_in_range > 0
        else 0.0
    )

st.markdown("### Customer Dynamics")

c1, c2, c3 = st.columns(3)
c1.metric("New Customers (in range)", new_customers_count)
c2.metric("Returning Customers (in range)", returning_customers_count)
c3.metric("Retention Rate", f"{retention_rate:.1f}%")

# You can also show the lists if you want:
with st.expander("New vs Returning customer lists"):
    if new_customers_count > 0:
        st.subheader("New customers")
        st.write(pd.DataFrame({"customer": new_customers}))
    else:
        st.write("No new customers in this range.")

    if returning_customers_count > 0:
        st.subheader("Returning customers")
        st.write(pd.DataFrame({"customer": returning_customers}))
    else:
        st.write("No returning customers in this range.")

# ---------- TOP 10 CUSTOMERS ----------

st.markdown("### Top 10 Customers (by sales in selected range)")

if "customer" in df_filtered.columns and "amount" in df_filtered.columns:
    top_cust = (
        df_filtered.groupby("customer")
        .agg(
            total_amount=("amount", "sum"),
            transactions=("amount", "count"),
        )
        .sort_values("total_amount", ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(top_cust, use_container_width=True)

    # Quick bar chart of total_amount by customer
    st.bar_chart(
        data=top_cust.set_index("customer")["total_amount"]
    )
else:
    st.info("Missing 'customer' or 'amount' column, cannot compute top customers.")

# ---------- SALES BY COMMODITY ----------

st.markdown("### Sales by Commodity (filtered)")

if "commodity" in df_filtered.columns and "amount" in df_filtered.columns:
    commodity_sales = (
        df_filtered.groupby("commodity")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(commodity_sales)
else:
    st.info("No 'commodity' column found to plot.")
