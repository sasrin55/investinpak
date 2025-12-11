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
    # Expecting:
    # date, customer, customer_type, commodity, amount, duration, ...

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

    # Drop fully empty rows
    df = df.dropna(how="all")

    return df


df = load_data()

# If nothing loaded, show a clear message
if df.empty:
    st.error("No data loaded from Google Sheets. Check sharing permissions and the URL.")
    st.stop()

st.write("Preview of data (first 10 rows):")
st.dataframe(df.head(10), use_container_width=True)

# ---------- SIMPLE KPIs ----------

total_amount = df["amount"].sum() if "amount" in df.columns else 0
total_txns = len(df)
unique_customers = df["customer"].nunique() if "customer" in df.columns else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales (PKR)", f"{total_amount:,.0f}")
col2.metric("Total Transactions", f"{total_txns:,}")
col3.metric("Unique Customers", f"{unique_customers:,}")

# ---------- SALES BY COMMODITY ----------

if "commodity" in df.columns:
    st.subheader("Sales by Commodity")

    commodity_sales = (
        df.groupby("commodity")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(commodity_sales)
else:
    st.info("No 'commodity' column found to plot.")
