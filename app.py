import pandas as pd
import streamlit as st
from datetime import date, timedelta, datetime
import altair as alt
import numpy as np
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo
from report_utils import load_data, explode_commodities, get_kpi_metrics, metric_format, count_transactions, sum_between, CURRENCY_CODE, CURRENCY_FORMAT

# --- TITLE AND CONFIGURATION (remains at the very top) ---
st.set_page_config(
    page_title="Zaraimandi Sales Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("Zaraimandi Sales Dashboard")
st.markdown("Transaction and Commodity-level Sales Intelligence.")
st.markdown("---")

# ==============================================================================
# 1. EMAIL HELPER FUNCTION DEFINITIONS (FIXED AND MOVED TO TOP)
# ==============================================================================

def build_daily_email_html(report_date: date, metrics: dict):
    """Builds a simple HTML email for a given date using KPI data."""
    
    rows_html = ""
    # Get the top 5 commodities table data from the calculated metrics
    # Using the DataFrame returned by get_kpi_metrics
    df = metrics["df"].groupby("commodity")["gross_amount_per_commodity"].sum().reset_index().sort_values("gross_amount_per_commodity", ascending=False).head(5)
    
    for _, row in df.iterrows():
        rows_html += f"""
        <tr>
            <td>{row['commodity']}</td>
            <td style="text-align:right">{CURRENCY_CODE} {row['gross_amount_per_commodity']:,.0f}</td>
        </tr>
        """
    
    html = f"""
    <html>
    <body>
        <h2>Zaraimandi Daily Report for {report_date}</h2>
        <p>This report summarizes the day's gross sales activity.</p>

        <p><b>Total sales:</b> {metric_format(metrics["total_amount"])}</p>
        <p><b>Total transactions:</b> {metrics["total_transactions"]}</p>
        <p><b>Unique customers:</b> {metrics["unique_customers"]}</p>
        <p><b>Top commodity:</b> {metrics["top_commodity_name"]} ({metrics["top_commodity_amount"]})</p>

        <h3>Top 5 Commodities</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
            <tr>
                <th>Commodity</th>
                <th>Amount ({CURRENCY_CODE})</th>
            </tr>
            {rows_html}
        </table>

        <p style="margin-top:18px;">
            <a href="YOUR_STREAMLIT_APP_URL_HERE" target="_blank">Open Full Interactive Dashboard</a>
        </p>
    </body>
    </html>
    """
    return html

# @st.cache_data wrapper around load_data is needed inside the app context
@st.cache_data(show_spinner="Connecting to Google Sheet and loading...")
def cached_load_data():
    return load_data(use_cache=False) # Delegate loading to report_utils

def send_email_report(recipient_emails: list, report_date: date):
    """Send the HTML report email to multiple recipients using st.secrets."""
    try:
        smtp_user = st.secrets["SMTP_USER"]
        smtp_pass = st.secrets["SMTP_PASS"]
        smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(st.secrets.get("SMTP_PORT", 465))
    except Exception as e:
        st.error(f"SMTP credentials not found in st.secrets. Please configure SMTP_USER / SMTP_PASS. Details: {e}")
        return False

    # FIX: Ensure raw_df_local and exploded_df_local are available for get_kpi_metrics
    try:
        raw_df_local = cached_load_data()
        exploded_df_local = explode_commodities(raw_df_local)
        metrics = get_kpi_metrics(raw_df_local, exploded_df_local, report_date, report_date)
    except Exception as e:
        st.error(f"Error calculating metrics for email report: {e}")
        return False

    html_body = build_daily_email_html(report_date, metrics)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Zaraimandi Daily Sales Report – {report_date}"
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipient_emails)

    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient_emails, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False


# ==============================================================================
# 2. DATA LOADING AND PRE-CALCULATIONS (Standard App Loading)
# ==============================================================================

raw_df = cached_load_data()
exploded_df = explode_commodities(raw_df)

if raw_df.empty:
    st.stop()

# Date Calculations
today = date.today()
start_of_year = date(today.year, 1, 1)
last_30_days_start = today - timedelta(days=29)
last_7_days_start = today - timedelta(days=6)


# --- Date Handling and Filters (CRASH FIX SECTION) ---
safe_min_date = date(2020, 1, 1) # CRASH FIX: Safe boundary
safe_max_date = today 
raw_min = raw_df["date"].min()
raw_max = raw_df["date"].max()
min_data_date = raw_min if pd.notna(raw_min) else start_of_year
max_data_date = raw_max if pd.notna(raw_max) else today

if isinstance(min_data_date, pd.Timestamp): min_data_date = min_data_date.date()
if isinstance(max_data_date, pd.Timestamp): max_data_date = max_data_date.date()

if min_data_date > max_data_date: min_data_date = max_data_date

st.subheader("Reporting Filters")
filter_cols = st.columns([1, 4])
with filter_cols[0]:
    date_range = st.date_input(
        "Reporting Period", value=(min_data_date, today), 
        min_value=safe_min_date, max_value=safe_max_date, key="top_date_filter"
    )

filter_start_date = min(date_range)
filter_end_date = max(date_range) if len(date_range) == 2 else date_range[0]

raw_df_filtered = raw_df[(raw_df["date"] >= filter_start_date) & (raw_df["date"] <= filter_end_date)]
exploded_df_filtered = exploded_df[(exploded_df["date"] >= filter_start_date) & (exploded_df["date"] <= filter_end_date)]

if raw_df_filtered.empty or exploded_df_filtered.empty:
    st.warning("No data matches the current date filter criteria. Please adjust your selections.")
    st.stop()

st.markdown("---")

# ==============================================================================
# 9. EMAIL REPORT FROM DASHBOARD UI (ON-DEMAND BUTTON) - MOVED TO TOP
# ==============================================================================

st.header("Email Report (On-Demand Sender)")

st.caption("Enter a single email address to securely send today's summary report.")

col_email_input, col_email_button = st.columns([3, 1])

with col_email_input:
    recipient_email = st.text_input("Recipient email", placeholder="someone@example.com")

with col_email_button:
    st.write("")
    st.write("")
    send_now = st.button("Send Today's Report Now")

if send_now:
    if not recipient_email:
        st.warning("Please enter an email address first.")
    else:
        # Use Pakistan time for choosing 'today' if you care about date rollover
        try:
            today_pk = datetime.now(ZoneInfo("Asia/Karachi")).date()
        except Exception:
            # Fallback if zoneinfo is not perfectly configured
            today_pk = datetime.now().date() 

        success = send_email_report([recipient_email], today_pk) 
        if success:
            st.success(f"Report sent to {recipient_email}")

st.markdown("---")


# ==============================================================================
# 4. KEY PERFORMANCE INDICATORS (KPIs) - VERTICAL SECTIONS 
# ==============================================================================

st.header("Key Performance Indicators (KPIs) - Gross Sales")

def create_summary_table_vertical(df, period_title, transactions_count):
    """Generates the detailed table for the vertical sections."""
    AMOUNT_COL_NAME = f"Amount ({CURRENCY_CODE})"
    summary_df = df.groupby(["customer_name", "commodity"])["gross_amount_per_commodity"].sum().reset_index()
    summary_df = summary_df.rename(columns={"customer_name": "Customer", "commodity": "Commodity", "gross_amount_per_commodity": AMOUNT_COL_NAME})
    summary_df = summary_df.sort_values(AMOUNT_COL_NAME, ascending=False)
    styled_df = summary_df.style.format({AMOUNT_COL_NAME: f"{CURRENCY_CODE} {{:,.0f}}",})
    
    st.subheader(f"Detailed Breakdown (Total Transactions: {transactions_count})")
    with st.container(border=True):
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)


# --- RENDER ALL KPI SECTIONS VERTICALLY ---

def render_kpi_block(title, start_date, end_date):
    st.markdown(f"## {title}")
    # Pass raw_df and exploded_df to the utility function
    metrics = get_kpi_metrics(raw_df, exploded_df, start_date, end_date) 

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("**Total Sales**", metric_format(metrics["total_amount"]))
    with col2:
        st.metric("**Total Transactions**", metrics["total_transactions"])
    with col3:
        st.metric("**Unique Customers**", metrics["unique_customers"])
    with col4:
        st.metric("**Top Commodity**", f"{metrics['top_commodity_name']} ({metrics['top_commodity_amount']})")

    create_summary_table_vertical(
        metrics["df"], title, metrics["total_transactions"]
    )
    st.markdown("---")
    return metrics 

render_kpi_block("Today's Sales Performance", today, today)
render_kpi_block("Last 7 Days Performance", last_7_days_start, today)
render_kpi_block("Last 30 Days Performance", last_30_days_start, today)
render_kpi_block("Year-to-Date (YTD) Performance", start_of_year, today)

# ==============================================================================
# 10. LOYALTY, SEASONALITY, DATA EXPLORER (Original Sections 5-8)
# ==============================================================================

# 5. COMMODITY LOYALTY (New vs. Repeat Count)
st.header("Commodity Loyalty Analysis")
st.markdown("Analyzes commodity performance based on the count of unique **New** vs. **Repeat** buyers across the entire dataset.")

txn_count_by_customer_commodity = (
    raw_df.groupby(["customer_name"])["date"].nunique().reset_index()
)
txn_count_by_customer_commodity.rename(columns={"date": "Total Transactions"}, inplace=True)

txn_count_by_customer_commodity["Buyer Type"] = np.where(
    txn_count_by_customer_commodity["Total Transactions"] > 1, 
    "Repeat", 
    "New"
)

loyalty_summary = (
    exploded_df.merge(txn_count_by_customer_commodity, on='customer_name', how='left')
    .groupby(["commodity", "Buyer Type"])
    ["customer_name"].nunique()
    .unstack(fill_value=0)
)

if 'New' not in loyalty_summary.columns: loyalty_summary['New'] = 0
if 'Repeat' not in loyalty_summary.columns: loyalty_summary['Repeat'] = 0

loyalty_summary = loyalty_summary[['Repeat', 'New']] # Order columns
loyalty_summary['Total Buyers'] = loyalty_summary['Repeat'] + loyalty_summary['New']

loyalty_summary = loyalty_summary.reset_index()
loyalty_summary = loyalty_summary.sort_values("Repeat", ascending=False)
loyalty_summary.rename(columns={"Repeat": "Repeat Buyers", "New": "New Buyers"}, inplace=True)

st.subheader("Commodity Buyer Loyalty (Ranked by Repeat Buyers)")
if not loyalty_summary.empty:
    st.dataframe(loyalty_summary, use_container_width=True, hide_index=True)
else:
    st.info("No sales data available to analyze buyer loyalty.")
st.markdown("---")


# 6. COMMODITY SEASONALITY
st.header("Commodity Seasonality Analysis")
st.markdown("Sales trend by month to identify peak selling seasons for each commodity.")

seasonality_df = exploded_df.copy()
seasonality_df["Month"] = seasonality_df["date"].apply(lambda x: x.replace(day=1))
seasonality_df["Month_Name"] = seasonality_df["date"].apply(lambda x: x.strftime("%Y-%m"))

seasonality_summary = seasonality_df.groupby(["Month_Name", "commodity"])["gross_amount_per_commodity"].sum().reset_index()
seasonality_summary.rename(columns={"gross_amount_per_commodity": "Total Sales"}, inplace=True)

commodity_list = sorted(seasonality_summary["commodity"].unique().tolist())
selected_season_commodity = st.selectbox(
    "Select Commodity for Seasonality Chart",
    options=commodity_list,
    index=0 
)

seasonality_chart_data = seasonality_summary[seasonality_summary["commodity"] == selected_season_commodity]

if not seasonality_chart_data.empty:
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


# 7. COMMODITY BREAKDOWN (Bar Chart)
st.header("Commodity Performance & Mix (All Data)")

commodity_summary = (
    exploded_df_filtered.groupby("commodity")["gross_amount_per_commodity"]
    .sum()
    .reset_index()
    .rename(columns={"gross_amount_per_commodity": "Amount"})
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
    ).interactive() 

    st.altair_chart(chart_bar, use_container_width=True)

with col_table:
    st.subheader("Summary Table")
    styled_df = commodity_summary.style.format({
        "Amount": f"{CURRENCY_CODE} {{:,.0f}}",
    })

    st.dataframe(styled_df, use_container_width=True, hide_index=True)
st.markdown("---")


# 8. DATA EXPLORER
st.header("Data Explorer: Transaction and Commodity Detail")
st.caption(f"Showing data for period: {filter_start_date.strftime('%b %d, %Y')} to {filter_end_date.strftime('%b %d, %Y')}")

data_choice = st.selectbox(
    "Select Data View:",
    options=["Raw Transactions (Total Amount)", "Exploded Data (Commodity Level Amount)"]
)

if data_choice == "Raw Transactions (Total Amount)":
    st.subheader("Raw Transaction Data")
    df_display = raw_df_filtered.sort_values("date", ascending=False).drop(columns=['phone'], errors='ignore')
    
    df_display.rename(columns={'amount_pkr': f'Gross Amount ({CURRENCY_CODE})'}, inplace=True)

    styled_df = df_display.style.format({
        f'Gross Amount ({CURRENCY_CODE})': f"{CURRENCY_CODE} {{:,.0f}}",
    })
    st.dataframe(styled_df, use_container_width=True)

else:
    st.subheader("Exploded Commodity Data")
    df_display = exploded_df_filtered.sort_values(["date", "customer_name"], ascending=False).drop(columns=['amount_pkr'], errors='ignore')
    
    df_display.rename(columns={'gross_amount_per_commodity': f'Gross Amount per Commodity ({CURRENCY_CODE})'}, inplace=True)


    styled_df = df_display.style.format({
        f'Gross Amount per Commodity ({CURRENCY_CODE})': f"{CURRENCY_CODE} {{:,.0f}}",
    })
    st.dataframe(styled_df, use_container_width=True)
