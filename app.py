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
from report_utils import load_data, explode_commodities, get_kpi_metrics, metric_format, count_transactions, sum_between, CURRENCY_CODE

# --- SETTINGS ---
# Sheet ID and URL are now handled in report_utils.py

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
# 2. DATA LOADING AND PRE-CALCULATIONS (Using functions from report_utils)
# ==============================================================================

# Custom load_data wrapper to use st.cache_data
@st.cache_data(show_spinner="Connecting to Google Sheet and loading...")
def cached_load_data():
    return load_data(use_cache=False) # Delegate loading to report_utils

raw_df = cached_load_data()
exploded_df = explode_commodities(raw_df)

if raw_df.empty:
    st.stop()

# Date Calculations
today = date.today()
start_of_year = date(today.year, 1, 1)
last_30_days_start = today - timedelta(days=29)
last_7_days_start = today - timedelta(days=6)


# --- Date Handling and Filters (Copied from previous steps for consistency) ---
safe_min_date = date(2020, 1, 1)
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
    return metrics # Return metrics for potential email use

render_kpi_block("Today's Sales Performance", today, today)
render_kpi_block("Last 7 Days Performance", last_7_days_start, today)
render_kpi_block("Last 30 Days Performance", last_30_days_start, today)
render_kpi_block("Year-to-Date (YTD) Performance", start_of_year, today)

# ==============================================================================
# 5. EMAIL HELPERS (For On-Demand Button)
# ==============================================================================

# Note: This logic is duplicated in daily_email_runner.py, but is kept here
# to avoid circular imports and allow the scheduled runner to run outside Streamlit.

def build_daily_email_html(report_date: date, metrics: dict):
    """Builds a simple HTML email for a given date using cached KPI data."""
    
    rows_html = ""
    top_table = metrics["df"].groupby("commodity")["gross_amount_per_commodity"].sum().reset_index().sort_values("gross_amount_per_commodity", ascending=False).head(5)
    
    for _, row in top_table.iterrows():
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

def send_email_report(recipient_emails: list, report_date: date):
    """Send the HTML report email to multiple recipients using st.secrets."""
    try:
        smtp_user = st.secrets["SMTP_USER"]
        smtp_pass = st.secrets["SMTP_PASS"]
        smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(st.secrets.get("SMTP_PORT", 465))
    except Exception as e:
        st.error(f"SMTP credentials not found in st.secrets. Details: {e}")
        return False

    # Calculate TODAY's metrics (using cached data)
    metrics = get_kpi_metrics(raw_df, exploded_df, report_date, report_date)
    
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

# ------------------------------------------------------------------
# 9. EMAIL REPORT FROM DASHBOARD UI (ON-DEMAND BUTTON)
# ------------------------------------------------------------------

st.markdown("---")
st.header("Email this report (On-Demand)")

st.caption("Enter a single email address and send today's summary report directly.")

col_email_input, col_email_button = st.columns([3, 1])

with col_email_input:
    recipient_email = st.text_input("Recipient email", placeholder="someone@example.com")

with col_email_button:
    # Use a dummy variable for button positioning
    st.write("")
    st.write("")
    send_now = st.button("Send Today's Report Now")

if send_now:
    if not recipient_email:
        st.warning("Please enter an email address first.")
    else:
        today_pk = datetime.now(ZoneInfo("Asia/Karachi")).date()
        success = send_email_report([recipient_email], today_pk)
        if success:
            st.success(f"Report sent to {recipient_email}")

st.markdown("---")

# ==============================================================================
# 10. LOYALTY, SEASONALITY, DATA EXPLORER (Rest of the app)
# ==============================================================================

# ... (Sections 5, 6, 7, 8 go here, using exploded_df and raw_df as loaded) ...
# I will omit the rest of the code here as it remains unchanged from the last block
# you received, but it must be included in your final app.py file.
