import pandas as pd
import streamlit as st
from datetime import date, timedelta
import altair as alt
import numpy as np
import re

# ==============================================================================
# 1. CONFIGURATION AND INITIAL SETUP
# ==============================================================================

SHEET_ID = "1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q"
TAB_NAME = "Master"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={TAB_NAME}"
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f"

st.set_page_config(
    page_title="Zaraimandi Sales Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Zaraimandi Sales Dashboard")
st.markdown("Transaction and Commodity level Sales Intelligence.")
st.markdown("---")

# ==============================================================================
# 2. DATA LOADING AND CLEANUP
# ==============================================================================

@st.cache_data(show_spinner="Connecting to Google Sheet and loading...")
def load_data():
    """Reads data, renames columns, and cleans data types."""
    try:
        df = pd.read_csv(CSV_URL)
    except Exception as e:
        st.error(
            f"Error connecting to Google Sheet. Check ID, tab name, and permissions. Details: {e}"
        )
        st.stop()
        return pd.DataFrame()

    cols = df.columns
    if len(cols) < 6:
        st.error(
            f"Data structure error: Expected at least 6 columns, found {len(cols)}. Check the '{TAB_NAME}' tab."
        )
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

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["amount_pkr"] = pd.to_numeric(df["amount_pkr"], errors="coerce").fillna(0)
    df["txn_type"] = df["txn_type"].astype(str).str.strip().str.title()
    df["customer_name"] = df["customer_name"].astype(str).str.strip()
    df = df.dropna(subset=["date"])

    return df


def explode_commodities(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits transaction rows into one row per commodity, fairly allocating the total amount,
    with cleaning for de duplication and removal of non commodities.
    """
    if base_df.empty or "commodities" not in base_df.columns:
        return base_df.copy()

    temp = base_df.copy()
    temp["commodities"] = temp["commodities"].fillna("").astype(str)

    # Normalize separators to commas, including plus sign
    temp["commodities_clean"] = (
        temp["commodities"]
        .str.replace(r"[&+/]", ",", regex=True)
        .str.replace(r"\band\b", ",", regex=True)
        .str.strip()
    )

    # Build list of commodities
    temp["commodity_list"] = temp["commodities_clean"].apply(
        lambda s: [x.strip() for x in s.split(",") if x.strip() != ""]
    )

    def normalize_commodity_name(name):
        if not name:
            return None

        name = name.lower().strip()
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", " ", name).strip()

        if not name:
            return None

        NON_COMMODITY_KEYWORDS = [
            "unknown",
            "not confirm yet",
            "discussion",
            "contact sale",
            "live market",
            "data",
            "group",
            "g",
            "um",
            "l m",
            "lm",
        ]

        if any(keyword in name for keyword in NON_COMMODITY_KEYWORDS):
            return None

        mapping = {
            "cotton": "Cotton",
            "coton": "Cotton",
            "paddy": "Paddy",
            "wheat": "Wheat",
            "edibleoil": "Edible Oil",
            "edible oil": "Edible Oil",
            "fertilizer": "Fertilizer",
            "pulses": "Pulses",
            "daal": "Pulses",
            "bajra": "Bajra",
            "livestock": "Livestock",
            "sesame": "Sesame",
            "sugar": "Sugar",
            "mustard": "Mustard",
            "mustrad": "Mustard",
            "kiryana": "Kiryana",
            "dryfruits": "Dry Fruits",
            "dry fruits": "Dry Fruits",
            "spices": "Spices",
            "rice": "Rice",
            "maize": "Maize",
            "dates": "Dates",
        }

        if name in mapping:
            return mapping[name]

        return name.title()

    temp["commodity_list"] = temp["commodity_list"].apply(
        lambda lst: [
            normalize_commodity_name(item)
            for item in lst
            if normalize_commodity_name(item) is not None
        ]
    )

    temp["n_commodities"] = temp["commodity_list"].apply(
        lambda lst: len(lst) if len(lst) > 0 else np.nan
    )

    temp = temp.explode("commodity_list")
    temp = temp[temp["commodity_list"].notna() & (temp["n_commodities"].notna())]
    temp = temp.rename(columns={"commodity_list": "commodity"})

    temp["amount_per_commodity"] = temp["amount_pkr"] / temp["n_commodities"]

    return temp[
        [
            "date",
            "customer_name",
            "txn_type",
            "commodity",
            "amount_per_commodity",
            "amount_pkr",
        ]
    ]


def sum_between(df, start, end, amount_col="amount_pkr"):
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask, amount_col].sum()


def count_transactions(df, start, end):
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].shape[0]


# ==============================================================================
# 3. DATA LOADING, FILTERING, AND PRE CALCULATIONS
# ==============================================================================

raw_df = load_data()
exploded_df = explode_commodities(raw_df)

if raw_df.empty:
    st.stop()

today = date.today()
start_of_year = date(today.year, 1, 1)
last_30_days_start = today - timedelta(days=29)
last_7_days_start = today - timedelta(days=6)

safe_min_date = date(2020, 1, 1)
safe_max_date = today

raw_min = raw_df["date"].min()
raw_max = raw_df["date"].max()

min_data_date = raw_min if pd.notna(raw_min) else start_of_year
max_data_date = raw_max if pd.notna(raw_max) else today

if isinstance(min_data_date, pd.Timestamp):
    min_data_date = min_data_date.date()
if isinstance(max_data_date, pd.Timestamp):
    max_data_date = max_data_date.date()

if min_data_date > max_data_date:
    min_data_date = max_data_date

st.subheader("Reporting Filters")
filter_cols = st.columns([1, 4])

with filter_cols[0]:
    date_range = st.date_input(
        "Reporting Period",
        value=(min_data_date, today),
        min_value=safe_min_date,
        max_value=safe_max_date,
        key="top_date_filter",
    )

filter_start_date = min(date_range)
filter_end_date = max(date_range) if len(date_range) == 2 else date_range[0]

raw_df_filtered = raw_df[
    (raw_df["date"] >= filter_start_date) & (raw_df["date"] <= filter_end_date)
]
exploded_df_filtered = exploded_df[
    (exploded_df["date"] >= filter_start_date)
    & (exploded_df["date"] <= filter_end_date)
]

if raw_df_filtered.empty or exploded_df_filtered.empty:
    st.warning(
        "No data matches the current date filter criteria. Please adjust your selections."
    )
    st.stop()

st.markdown("---")

# ==============================================================================
# 4. KEY PERFORMANCE INDICATORS
# ==============================================================================

st.header("Key Performance Indicators (KPIs)")


def metric_format(value):
    return f"{CURRENCY_CODE} {value:,.0f}"


def get_kpi_metrics(start_date, end_date):
    period_mask = (exploded_df["date"] >= start_date) & (
        exploded_df["date"] <= end_date
    )
    period_df = exploded_df.loc[period_mask].copy()

    total_amount = period_df["amount_per_commodity"].sum()
    total_transactions = count_transactions(raw_df, start_date, end_date)
    unique_customers = period_df["customer_name"].nunique()

    top_commodity_series = (
        period_df.groupby("commodity")["amount_per_commodity"]
        .sum()
        .nlargest(1)
    )
    top_commodity_name = (
        top_commodity_series.index[0] if not top_commodity_series.empty else "N/A"
    )
    top_commodity_amount = (
        metric_format(top_commodity_series.iloc[0])
        if not top_commodity_series.empty
        else "N/A"
    )

    return {
        "df": period_df,
        "total_amount": total_amount,
        "total_transactions": total_transactions,
        "unique_customers": unique_customers,
        "top_commodity_name": top_commodity_name,
        "top_commodity_amount": top_commodity_amount,
    }


def create_summary_table_vertical(df, period_title, transactions_count):
    amount_col_name = f"Amount ({CURRENCY_CODE})"

    summary_df = (
        df.groupby(["customer_name", "commodity"])["amount_per_commodity"]
        .sum()
        .reset_index()
    )

    summary_df = summary_df.rename(
        columns={
            "customer_name": "Customer",
            "commodity": "Commodity",
            "amount_per_commodity": amount_col_name,
        }
    )

    summary_df = summary_df.sort_values(amount_col_name, ascending=False)

    styled_df = summary_df.style.format(
        {
            amount_col_name: f"{CURRENCY_CODE} {{:,.0f}}",
        }
    )

    st.subheader(f"Detailed Breakdown (Total Transactions: {transactions_count})")

    with st.container(border=True):
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)


st.markdown("## Today Sales Performance")
today_sales_date = today
today_metrics = get_kpi_metrics(today_sales_date, today_sales_date)

col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    st.metric("Total Sales", metric_format(today_metrics["total_amount"]))
with col_t2:
    st.metric("Total Transactions", today_metrics["total_transactions"])
with col_t3:
    st.metric("Unique Customers", today_metrics["unique_customers"])
with col_t4:
    st.metric(
        "Top Commodity",
        f"{today_metrics['top_commodity_name']} ({today_metrics['top_commodity_amount']})",
    )

create_summary_table_vertical(
    today_metrics["df"], "Today Sales Breakdown", today_metrics["total_transactions"]
)
st.markdown("---")

st.markdown("## Last 7 Days Performance")
last_7_metrics = get_kpi_metrics(last_7_days_start, today)

col_7_1, col_7_2, col_7_3, col_7_4 = st.columns(4)
with col_7_1:
    st.metric("Total Sales", metric_format(last_7_metrics["total_amount"]))
with col_7_2:
    st.metric("Total Transactions", last_7_metrics["total_transactions"])
with col_7_3:
    st.metric("Unique Customers", last_7_metrics["unique_customers"])
with col_7_4:
    st.metric(
        "Top Commodity",
        f"{last_7_metrics['top_commodity_name']} ({last_7_metrics['top_commodity_amount']})",
    )

create_summary_table_vertical(
    last_7_metrics["df"], "Last 7 Days Breakdown", last_7_metrics["total_transactions"]
)
st.markdown("---")

st.markdown("## Last 30 Days Performance")
last_30_metrics = get_kpi_metrics(last_30_days_start, today)

col_30_1, col_30_2, col_30_3, col_30_4 = st.columns(4)
with col_30_1:
    st.metric("Total Sales", metric_format(last_30_metrics["total_amount"]))
with col_30_2:
    st.metric("Total Transactions", last_30_metrics["total_transactions"])
with col_30_3:
    st.metric("Unique Customers", last_30_metrics["unique_customers"])
with col_30_4:
    st.metric(
        "Top Commodity",
        f"{last_30_metrics['top_commodity_name']} ({last_30_metrics['top_commodity_amount']})",
    )

create_summary_table_vertical(
    last_30_metrics["df"],
    "Last 30 Days Breakdown",
    last_30_metrics["total_transactions"],
)
st.markdown("---")

st.markdown("## Year to Date Performance")
ytd_metrics = get_kpi_metrics(start_of_year, today)

col_ytd_1, col_ytd_2, col_ytd_3, col_ytd_4 = st.columns(4)
with col_ytd_1:
    st.metric("Total Sales", metric_format(ytd_metrics["total_amount"]))
with col_ytd_2:
    st.metric("Total Transactions", ytd_metrics["total_transactions"])
with col_ytd_3:
    st.metric("Unique Customers", ytd_metrics["unique_customers"])
with col_ytd_4:
    st.metric(
        "Top Commodity",
        f"{ytd_metrics['top_commodity_name']} ({ytd_metrics['top_commodity_amount']})",
    )

create_summary_table_vertical(
    ytd_metrics["df"], "YTD Sales Breakdown", ytd_metrics["total_transactions"]
)
st.markdown("---")

# ==============================================================================
# 5. COMMODITY LOYALTY
# ==============================================================================

st.header("Commodity Loyalty Analysis")
st.markdown(
    "Analyzes commodity performance based on the count of unique New vs Repeat buyers across the entire dataset."
)

txn_count_by_customer_commodity = (
    exploded_df.groupby(["customer_name", "commodity"])["date"]
    .nunique()
    .reset_index()
)
txn_count_by_customer_commodity.rename(columns={"date": "Total Transactions"}, inplace=True)

txn_count_by_customer_commodity["Buyer Type"] = np.where(
    txn_count_by_customer_commodity["Total Transactions"] > 1, "Repeat", "New"
)

loyalty_summary = (
    txn_count_by_customer_commodity.groupby(["commodity", "Buyer Type"])
    .size()
    .unstack(fill_value=0)
)

if "New" not in loyalty_summary.columns:
    loyalty_summary["New"] = 0
if "Repeat" not in loyalty_summary.columns:
    loyalty_summary["Repeat"] = 0

loyalty_summary = loyalty_summary[["Repeat", "New"]]
loyalty_summary["Total Buyers"] = loyalty_summary["Repeat"] + loyalty_summary["New"]

loyalty_summary = loyalty_summary.reset_index()
loyalty_summary = loyalty_summary.sort_values("Repeat", ascending=False)
loyalty_summary.rename(
    columns={"Repeat": "Repeat Buyers", "New": "New Buyers"}, inplace=True
)

st.subheader("Commodity Buyer Loyalty ranked by Repeat Buyers")

if not loyalty_summary.empty:
    st.dataframe(loyalty_summary, use_container_width=True, hide_index=True)
else:
    st.info("No sales data available to analyze buyer loyalty.")

st.markdown("---")

# ==============================================================================
# 6. COMMODITY SEASONALITY
# ==============================================================================

st.header("Commodity Seasonality Analysis")
st.markdown("Sales trend by month to identify peak selling seasons for each commodity.")

seasonality_df = exploded_df.copy()
seasonality_df["Month_Name"] = seasonality_df["date"].apply(lambda x: x.strftime("%Y-%m"))

seasonality_summary = (
    seasonality_df.groupby(["Month_Name", "commodity"])["amount_per_commodity"]
    .sum()
    .reset_index()
)
seasonality_summary.rename(columns={"amount_per_commodity": "Total Sales"}, inplace=True)

commodity_list = sorted(seasonality_summary["commodity"].unique().tolist())
selected_season_commodity = st.selectbox(
    "Select Commodity for Seasonality Chart",
    options=commodity_list,
    index=0,
)

seasonality_chart_data = seasonality_summary[
    seasonality_summary["commodity"] == selected_season_commodity
]

if not seasonality_chart_data.empty:
    season_chart = (
        alt.Chart(seasonality_chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month_Name:N", title="Month"),
            y=alt.Y(
                "Total Sales:Q",
                title=f"Total Sales ({CURRENCY_CODE})",
                axis=alt.Axis(format=CURRENCY_FORMAT),
            ),
            tooltip=[
                alt.Tooltip("Month_Name:N", title="Month"),
                alt.Tooltip("Total Sales:Q", title="Sales Amount", format=CURRENCY_FORMAT),
            ],
        )
        .properties(title=f"Monthly Sales Trend for {selected_season_commodity}")
        .interactive()
    )

    st.altair_chart(season_chart, use_container_width=True)
else:
    st.info(f"No seasonality data found for {selected_season_commodity} in the dataset.")

st.markdown("---")

# ==============================================================================
# 7. COMMODITY BREAKDOWN
# ==============================================================================

st.header("Commodity Performance and Mix for Selected Period")

commodity_summary = (
    exploded_df_filtered.groupby("commodity")["amount_per_commodity"]
    .sum()
    .reset_index()
    .rename(columns={"amount_per_commodity": "Amount"})
    .sort_values("Amount", ascending=False)
)

col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.subheader("Top Selling Commodities by Amount")

    max_commodities = len(commodity_summary)
    top_n_default = min(10, max_commodities)

    top_n = st.slider(
        "Select Top N Commodities to Display",
        1,
        max_commodities,
        top_n_default,
        key="top_n_slider",
    )

    top_commodity_summary = commodity_summary.head(top_n)

    chart_bar = (
        alt.Chart(top_commodity_summary)
        .mark_bar()
        .encode(
            x=alt.X(
                "Amount:Q",
                title=f"Total Sales ({CURRENCY_CODE})",
                axis=alt.Axis(format=CURRENCY_FORMAT),
            ),
            y=alt.Y("commodity:N", sort="-x", title="Commodity"),
            tooltip=[
                alt.Tooltip("commodity:N", title="Commodity"),
                alt.Tooltip("Amount:Q", title="Sales Amount", format=CURRENCY_FORMAT),
            ],
        )
        .properties(title=f"Top {top_n} Commodities by Sales Amount")
        .interactive()
    )

    st.altair_chart(chart_bar, use_container_width=True)

with col_table:
    st.subheader("Summary Table")
    styled_df = commodity_summary.style.format(
        {
            "Amount": f"{CURRENCY_CODE} {{:,.0f}}",
        }
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ==============================================================================
# 8. DATA EXPLORER
# ==============================================================================

st.header("Data Explorer: Transaction and Commodity Detail")
st.caption(
    f"Showing data for period: {filter_start_date.strftime('%b %d, %Y')} to {filter_end_date.strftime('%b %d, %Y')}"
)

data_choice = st.selectbox(
    "Select Data View:",
    options=["Raw Transactions (Total Amount)", "Exploded Data (Commodity Level Amount)"],
)

if data_choice == "Raw Transactions (Total Amount)":
    st.subheader("Raw Transaction Data")
    df_display = raw_df_filtered.sort_values("date", ascending=False).drop(
        columns=["phone"], errors="ignore"
    )

    styled_df = df_display.style.format(
        {
            "amount_pkr": f"{CURRENCY_CODE} {{:,.0f}}",
        }
    )
    st.dataframe(styled_df, use_container_width=True)

else:
    st.subheader("Exploded Commodity Data")
    df_display = exploded_df_filtered.sort_values(
        ["date", "customer_name"], ascending=False
    ).drop(columns=["amount_pkr"], errors="ignore")

    styled_df = df_display.style.format(
        {
            "amount_per_commodity": f"{CURRENCY_CODE} {{:,.0f}}",
        }
    )
    st.dataframe(styled_df, use_container_width=True)
