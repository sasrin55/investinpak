import pandas as pd
import streamlit as st
from datetime import date, timedelta, datetime
import altair as alt
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from report_utils import (
    load_data,
    explode_commodities,
    get_kpi_metrics,
    metric_format,
    count_transactions,
    sum_between,
    CURRENCY_CODE,
    CURRENCY_FORMAT,
)

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Zaraimandi Sales Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("Zaraimandi Sales Dashboard")
st.markdown("Transaction and Commodity-level Sales Intelligence.")
st.markdown("---")

# -----------------------------------------------------------------------------
# EMAIL HELPERS (ADVANCED VERSION ONLY)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Connecting to Google Sheet and loading...")
def cached_load_data():
    """Load data from Google Sheets and remember when it was refreshed.

    ttl=300 means Streamlit will reload from the Sheet at most every 5 minutes.
    """
    df = load_data(use_cache=False)
    refreshed_at = datetime.now(ZoneInfo("Asia/Karachi"))
    return df, refreshed_at

def get_commodity_comparisons(exploded_df: pd.DataFrame, report_date: date) -> pd.DataFrame:
    """Calculates WoW and MoM change percentages for the top commodities."""
    current_7_start = report_date - timedelta(days=6)
    previous_7_start = report_date - timedelta(days=13)
    current_30_start = report_date - timedelta(days=29)
    previous_30_start = report_date - timedelta(days=59)

    mask_current_30 = (exploded_df["date"] >= previous_30_start) & (exploded_df["date"] <= report_date)
    df_base = exploded_df.loc[mask_current_30]

    sales_current_week = df_base[
        (df_base["date"] >= current_7_start) & (df_base["date"] <= report_date)
    ]
    sales_current_month = df_base[
        (df_base["date"] >= current_30_start) & (df_base["date"] <= report_date)
    ]
    sales_previous_week = df_base[
        (df_base["date"] >= previous_7_start) & (df_base["date"] < current_7_start)
    ]
    sales_previous_month = df_base[
        (df_base["date"] >= previous_30_start) & (df_base["date"] < current_30_start)
    ]

    agg_sales = lambda df: df.groupby("commodity")["gross_amount_per_commodity"].sum()

    current_week = agg_sales(sales_current_week).rename("Current_Week")
    previous_week = agg_sales(sales_previous_week).rename("Previous_Week")
    current_month = agg_sales(sales_current_month).rename("Current_Month")
    previous_month = agg_sales(sales_previous_month).rename("Previous_Month")

    comparison_df = pd.concat(
        [current_week, previous_week, current_month, previous_month], axis=1
    ).fillna(0)

    comparison_df["WoW Change %"] = np.where(
        comparison_df["Previous_Week"] > 0,
        (comparison_df["Current_Week"] - comparison_df["Previous_Week"])
        / comparison_df["Previous_Week"]
        * 100,
        np.where(comparison_df["Current_Week"] > 0, 100, 0),
    ).round(1)

    comparison_df["MoM Change %"] = np.where(
        comparison_df["Previous_Month"] > 0,
        (comparison_df["Current_Month"] - comparison_df["Previous_Month"])
        / comparison_df["Previous_Month"]
        * 100,
        np.where(comparison_df["Current_Month"] > 0, 100, 0),
    ).round(1)

    comparison_df = comparison_df.reset_index()
    comparison_df = comparison_df.sort_values("Current_Month", ascending=False).head(10)

    return comparison_df[["commodity", "Current_Month", "WoW Change %", "MoM Change %"]]


def calculate_rank_movement(exploded_df: pd.DataFrame, report_date: date) -> pd.DataFrame:
    """
    Calculates current YTD ranking and compares it to YTD ranking 30 days prior.
    This version is NaN safe and will not fail when converting to int.
    """
    last_month_end_date = report_date - timedelta(days=30)
    start_of_year = date(report_date.year, 1, 1)

    # YTD data up to report_date
    current_ytd = exploded_df[
        (exploded_df["date"] >= start_of_year) & (exploded_df["date"] <= report_date)
    ]
    if current_ytd.empty:
        return pd.DataFrame(columns=["commodity", "Amount", "Current Rank", "Rank Change"])

    # Current ranks
    current_totals = current_ytd.groupby("commodity")["gross_amount_per_commodity"].sum()
    rank_current = current_totals.rank(method="min", ascending=False)

    # Baseline YTD up to 30 days ago
    baseline_ytd = exploded_df[
        (exploded_df["date"] >= start_of_year) & (exploded_df["date"] <= last_month_end_date)
    ]
    baseline_totals = baseline_ytd.groupby("commodity")["gross_amount_per_commodity"].sum()
    rank_baseline = baseline_totals.rank(method="min", ascending=False)

    # Align on commodity
    rank_comparison = pd.concat(
        [rank_current.rename("Current Rank"), rank_baseline.rename("Baseline Rank")],
        axis=1,
    )

    # If there was no baseline rank, treat baseline as current so movement starts at zero
    rank_comparison["Baseline Rank"] = rank_comparison["Baseline Rank"].fillna(
        rank_comparison["Current Rank"]
    )

    # Now both columns are finite, safe to convert
    rank_comparison["Current Rank"] = rank_comparison["Current Rank"].astype(int)
    rank_comparison["Baseline Rank"] = rank_comparison["Baseline Rank"].astype(int)

    # Movement: positive means it moved up the table
    rank_comparison["Movement"] = (
        rank_comparison["Baseline Rank"] - rank_comparison["Current Rank"]
    )

    # Sales for context
    current_sales = current_totals.rename("Amount")

    final_df = pd.concat(
        [current_sales, rank_comparison["Current Rank"], rank_comparison["Movement"]],
        axis=1,
    ).reset_index()

    # Clean movement so we never try to cast NaN
    movement_int = final_df["Movement"].fillna(0).astype(int)

    final_df["Movement Symbol"] = np.where(
        movement_int > 0, "▲",
        np.where(movement_int < 0, "▼", "—"),
    )

    final_df["Rank Change"] = np.where(
        movement_int == 0,
        "—",
        final_df["Movement Symbol"] + " " + movement_int.abs().astype(str),
    )

    final_df = final_df.sort_values("Amount", ascending=False).head(10)

    return final_df[["commodity", "Amount", "Current Rank", "Rank Change"]]


def build_daily_email_html(
    report_date: date,
    today_metrics: dict,
    last_7_metrics: dict,
    ytd_metrics: dict,
    trend_df: pd.DataFrame,
    rank_df: pd.DataFrame,
) -> str:
    """Builds the HTML body for the daily email."""
    today_sales = today_metrics["total_amount"]
    last_7_total = last_7_metrics["total_amount"]
    last_7_avg = last_7_total / 7 if last_7_total else 0

    if last_7_avg > 0:
        change_percent = ((today_sales - last_7_avg) / last_7_avg) * 100
        direction = "higher" if change_percent >= 0 else "lower"
        change_text = (
            f"Today’s sales are <b>{abs(change_percent):.1f}% {direction}</b> "
            f"than the average of the last 7 days."
        )
    else:
        change_text = (
            "Not enough sales data in the last week for a meaningful comparison."
        )

    top_commodity = today_metrics["top_commodity_name"]
    top_commodity_amount = today_metrics["top_commodity_amount"]

    # Trend table
    trend_rows_html = ""
    for _, row in trend_df.iterrows():
        wow_style = "color:#006B3F;" if row["WoW Change %"] >= 0 else "color:#CC0000;"
        mom_style = "color:#006B3F;" if row["MoM Change %"] >= 0 else "color:#CC0000;"
        trend_rows_html += f"""
        <tr>
            <td>{row['commodity']}</td>
            <td style="text-align:right">{metric_format(row['Current_Month'])}</td>
            <td style="text-align:right;{wow_style}">{row['WoW Change %']:.1f}%</td>
            <td style="text-align:right;{mom_style}">{row['MoM Change %']:.1f}%</td>
        </tr>
        """

    # Rank movement table
    rank_rows_html = ""
    for _, row in rank_df.iterrows():
        rank_rows_html += f"""
        <tr>
            <td>{row['commodity']}</td>
            <td style="text-align:right">{metric_format(row['Amount'])}</td>
            <td style="text-align:center">{row['Current Rank']}</td>
            <td style="text-align:center">{row['Rank Change']}</td>
        </tr>
        """

    html = f"""
    <html>
    <body style="font-family:Arial, sans-serif; font-size:14px;">
        <h2>Zaraimandi Daily Report – {report_date}</h2>
        <p>This report summarizes the day's gross sales activity with context vs recent performance.</p>

        <ul>
            <li><b>Total sales:</b> {metric_format(today_metrics["total_amount"])}</li>
            <li><b>Total transactions:</b> {today_metrics["total_transactions"]}</li>
            <li><b>Unique customers:</b> {today_metrics["unique_customers"]}</li>
            <li><b>Top commodity:</b> {top_commodity} ({top_commodity_amount})</li>
        </ul>

        <p>{change_text}</p>

        <h3>Top commodities – last 30 days trend</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
            <tr>
                <th>Commodity</th>
                <th>Sales this month</th>
                <th>WoW change</th>
                <th>MoM change</th>
            </tr>
            {trend_rows_html}
        </table>

        <h3 style="margin-top:20px;">YTD rank movement (vs 30 days ago)</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
            <tr>
                <th>Commodity</th>
                <th>YTD Sales</th>
                <th>Current Rank</th>
                <th>Rank Change</th>
            </tr>
            {rank_rows_html}
        </table>

    <p style="margin-top:18px;">
    <a href="https://zmsales.streamlit.app/"
       target="_blank"
       style="font-size:14px; text-decoration:none; color:#1a73e8;">
       Open Full Interactive Dashboard
    </a>
</p>
      
    </body>
    </html>
    """
    return html


def send_email_report(recipient_emails: list, report_date: date) -> bool:
    """Send the HTML report email to multiple recipients using st.secrets."""
    try:
        smtp_user = st.secrets["SMTP_USER"]
        smtp_pass = st.secrets["SMTP_PASS"]
        smtp_server = st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(st.secrets.get("SMTP_PORT", 465))
    except Exception as e:
        st.error(
            f"SMTP credentials not found in st.secrets. Please configure SMTP_USER/SMTP_PASS. Details: {e}"
        )
        return False

    try:
        raw_df_local = cached_load_data()
        exploded_df_local = explode_commodities(raw_df_local)

        today_metrics = get_kpi_metrics(
            raw_df_local, exploded_df_local, report_date, report_date
        )
        last_7_metrics = get_kpi_metrics(
            raw_df_local, exploded_df_local, report_date - timedelta(days=6), report_date
        )
        ytd_metrics = get_kpi_metrics(
            raw_df_local,
            exploded_df_local,
            date(report_date.year, 1, 1),
            report_date,
        )

        trend_df = get_commodity_comparisons(exploded_df_local, report_date)
        rank_df = calculate_rank_movement(exploded_df_local, report_date)

    except Exception as e:
        st.error(f"Error calculating metrics for email report: {e}")
        return False

    html_body = build_daily_email_html(
        report_date, today_metrics, last_7_metrics, ytd_metrics, trend_df, rank_df
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[V2] Zaraimandi Daily Sales Report – {report_date}"
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


# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------

raw_df = cached_load_data()
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

# -----------------------------------------------------------------------------
# EMAIL SECTION (ON-DEMAND)
# -----------------------------------------------------------------------------

st.header("Email Report (On-Demand Sender)")
st.caption("Enter a single email address and send today's summary report.")

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
        try:
            today_pk = datetime.now(ZoneInfo("Asia/Karachi")).date()
        except Exception:
            today_pk = datetime.now().date()

        success = send_email_report([recipient_email], today_pk)
        if success:
            st.success(f"Report sent to {recipient_email}")

st.markdown("---")

# -----------------------------------------------------------------------------
# KPI BLOCKS
# -----------------------------------------------------------------------------

st.header("Key Performance Indicators (KPIs) - Gross Sales")


def create_summary_table_vertical(df, period_title, transactions_count):
    AMOUNT_COL_NAME = f"Amount ({CURRENCY_CODE})"
    summary_df = (
        df.groupby(["customer_name", "commodity"])["gross_amount_per_commodity"]
        .sum()
        .reset_index()
    )
    summary_df = summary_df.rename(
        columns={
            "customer_name": "Customer",
            "commodity": "Commodity",
            "gross_amount_per_commodity": AMOUNT_COL_NAME,
        }
    )
    summary_df = summary_df.sort_values(AMOUNT_COL_NAME, ascending=False)
    styled_df = summary_df.style.format(
        {AMOUNT_COL_NAME: f"{CURRENCY_CODE} {{:,.0f}}"}
    )

    st.subheader(f"Detailed Breakdown (Total Transactions: {transactions_count})")
    with st.container(border=True):
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)


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
        st.metric(
            "**Top Commodity**",
            f"{metrics['top_commodity_name']} ({metrics['top_commodity_amount']})",
        )

    create_summary_table_vertical(
        metrics["df"], title, metrics["total_transactions"]
    )
    st.markdown("---")
    return metrics


render_kpi_block("Today's Sales Performance", today, today)
render_kpi_block("Last 7 Days Performance", last_7_days_start, today)
render_kpi_block("Last 30 Days Performance", last_30_days_start, today)
render_kpi_block("Year-to-Date (YTD) Performance", start_of_year, today)

# -----------------------------------------------------------------------------
# LOYALTY, SEASONALITY, COMMODITY MIX, DATA EXPLORER
# -----------------------------------------------------------------------------

st.header("Commodity Loyalty Analysis")
st.markdown(
    "Analyzes commodity performance based on the count of unique **New** vs. **Repeat** buyers across the entire dataset."
)

txn_count_by_customer_commodity = (
    raw_df.groupby(["customer_name"])["date"].nunique().reset_index()
)
txn_count_by_customer_commodity.rename(
    columns={"date": "Total Transactions"}, inplace=True
)

txn_count_by_customer_commodity["Buyer Type"] = np.where(
    txn_count_by_customer_commodity["Total Transactions"] > 1, "Repeat", "New"
)

loyalty_summary = (
    exploded_df.merge(txn_count_by_customer_commodity, on="customer_name", how="left")
    .groupby(["commodity", "Buyer Type"])["customer_name"]
    .nunique()
    .unstack(fill_value=0)
)

if "New" not in loyalty_summary.columns:
    loyalty_summary["New"] = 0
if "Repeat" not in loyalty_summary.columns:
    loyalty_summary["Repeat"] = 0

loyalty_summary = loyalty_summary[["Repeat", "New"]]
loyalty_summary["Total Buyers"] = (
    loyalty_summary["Repeat"] + loyalty_summary["New"]
)

loyalty_summary = loyalty_summary.reset_index()
loyalty_summary = loyalty_summary.sort_values("Repeat", ascending=False)
loyalty_summary.rename(
    columns={"Repeat": "Repeat Buyers", "New": "New Buyers"}, inplace=True
)

st.subheader("Commodity Buyer Loyalty (Ranked by Repeat Buyers)")
if not loyalty_summary.empty:
    st.dataframe(loyalty_summary, use_container_width=True, hide_index=True)
else:
    st.info("No sales data available to analyze buyer loyalty.")
st.markdown("---")

st.header("Commodity Seasonality Analysis")
st.markdown("Sales trend by month to identify peak selling seasons for each commodity.")

seasonality_df = exploded_df.copy()
seasonality_df["Month"] = seasonality_df["date"].apply(lambda x: x.replace(day=1))
seasonality_df["Month_Name"] = seasonality_df["date"].apply(
    lambda x: x.strftime("%Y-%m")
)

seasonality_summary = (
    seasonality_df.groupby(["Month_Name", "commodity"])[
        "gross_amount_per_commodity"
    ]
    .sum()
    .reset_index()
)
seasonality_summary.rename(
    columns={"gross_amount_per_commodity": "Total Sales"}, inplace=True
)

commodity_list = sorted(seasonality_summary["commodity"].unique().tolist())
selected_season_commodity = st.selectbox(
    "Select Commodity for Seasonality Chart", options=commodity_list, index=0
)

seasonality_chart_data = seasonality_summary[
    seasonality_summary["commodity"] == selected_season_commodity
]

if not seasonality_chart_data.empty:
    season_chart = (
        alt.Chart(seasonality_chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Month_Name:T",
                title="Month",
                axis=alt.Axis(format="%Y-%m"),
            ),
            y=alt.Y(
                "Total Sales:Q",
                title=f"Total Sales ({CURRENCY_CODE})",
                axis=alt.Axis(format=CURRENCY_FORMAT),
            ),
            tooltip=[
                alt.Tooltip("Month_Name:T", title="Month", format="%Y-%m"),
                alt.Tooltip(
                    "Total Sales:Q", title="Sales Amount", format=CURRENCY_FORMAT
                ),
            ],
        )
        .properties(title=f"Monthly Sales Trend for {selected_season_commodity}")
        .interactive()
    )

    st.altair_chart(season_chart, use_container_width=True)
else:
    st.info(
        f"No seasonality data found for {selected_season_commodity} in the dataset."
    )
st.markdown("---")

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
        {"Amount": f"{CURRENCY_CODE} {{:,.0f}}"}
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
st.markdown("---")

st.header("Data Explorer: Transaction and Commodity Detail")
st.caption(
    f"Showing data for period: {filter_start_date.strftime('%b %d, %Y')} to {filter_end_date.strftime('%b %d, %Y')}"
)

data_choice = st.selectbox(
    "Select Data View:",
    options=[
        "Raw Transactions (Total Amount)",
        "Exploded Data (Commodity Level Amount)",
    ],
)

if data_choice == "Raw Transactions (Total Amount)":
    st.subheader("Raw Transaction Data")
    df_display = raw_df_filtered.sort_values("date", ascending=False).drop(
        columns=["phone"], errors="ignore"
    )
    df_display.rename(
        columns={"amount_pkr": f"Gross Amount ({CURRENCY_CODE})"}, inplace=True
    )
    styled_df = df_display.style.format(
        {f"Gross Amount ({CURRENCY_CODE})": f"{CURRENCY_CODE} {{:,.0f}}"}
    )
    st.dataframe(styled_df, use_container_width=True)
else:
    st.subheader("Exploded Commodity Data")
    df_display = exploded_df_filtered.sort_values(
        ["date", "customer_name"], ascending=False
    ).drop(columns=["amount_pkr"], errors="ignore")
    df_display.rename(
        columns={
            "gross_amount_per_commodity": f"Gross Amount per Commodity ({CURRENCY_CODE})"
        },
        inplace=True,
    )
    styled_df = df_display.style.format(
        {
            f"Gross Amount per Commodity ({CURRENCY_CODE})": f"{CURRENCY_CODE} {{:,.0f}}"
        }
    )
    st.dataframe(styled_df, use_container_width=True)
