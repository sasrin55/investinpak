import pandas as pd
import streamlit as st
from datetime import date, timedelta, datetime
import altair as alt
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

# NOTE: This imports the updated load_data function which uses the direct CSV URL.
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

# --- GOOGLE SHEET URL CONFIGURATION ---
# Updated GID to point to the 'Master for SalesOps' tab (GID=1105756916)
MAIN_DATA_URL = "https://docs.google.com/spreadsheets/d/1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q/gviz/tq?tqx=out:csv&gid=1105756916"

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Zarai Mandi Sales Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("Zarai Mandi Sales Dashboard")
st.markdown("Transaction and Commodity-level Sales Intelligence.")
st.markdown("---")

# -----------------------------------------------------------------------------
# EMAIL HELPERS AND DATA CACHE
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Connecting to Google Sheet and loading...")
def cached_load_data():
    """
    Load data from Google Sheets using the direct URL.
    
    ttl=300 means Streamlit will reload from the Sheet at most every 5 minutes.
    """
    # Use the simplified load_data function with the URL
    df = load_data(sheet_url=MAIN_DATA_URL)
    
    # Check if necessary columns exist after loading and cleaning
    if df.empty or 'date' not in df.columns or 'customer_name' not in df.columns:
        st.error("Error: Main dashboard data is missing essential columns (date or customer_name). Check sheet headers or access.")
        # Return empty data structure if load fails
        return pd.DataFrame(), datetime.now(ZoneInfo("Asia/Karachi"))

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

    sales_current_week = df_base[(df_base["date"] >= current_7_start) & (df_base["date"] <= report_date)]
    sales_current_month = df_base[(df_base["date"] >= current_30_start) & (df_base["date"] <= report_date)]
    sales_previous_week = df_base[(df_base["date"] >= previous_7_start) & (df_base["date"] < current_7_start)]
    sales_previous_month = df_base[(df_base["date"] >= previous_30_start) & (df_base["date"] < current_30_start)]

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
    NaN safe.
    """
    last_month_end_date = report_date - timedelta(days=30)
    start_of_year = date(report_date.year, 1, 1)

    current_ytd = exploded_df[
        (exploded_df["date"] >= start_of_year) & (exploded_df["date"] <= report_date)
    ]
    if current_ytd.empty:
        return pd.DataFrame(columns=["commodity", "Amount", "Current Rank", "Rank Change"])

    current_totals = current_ytd.groupby("commodity")["gross_amount_per_commodity"].sum()
    rank_current = current_totals.rank(method="min", ascending=False)

    baseline_ytd = exploded_df[
        (exploded_df["date"] >= start_of_year) & (exploded_df["date"] <= last_month_end_date)
    ]
    baseline_totals = baseline_ytd.groupby("commodity")["gross_amount_per_commodity"].sum()
    rank_baseline = baseline_totals.rank(method="min", ascending=False)

    rank_comparison = pd.concat(
        [rank_current.rename("Current Rank"), rank_baseline.rename("Baseline Rank")],
        axis=1,
    )

    # If there was no baseline rank, start baseline at current so movement = 0
    rank_comparison["Baseline Rank"] = rank_comparison["Baseline Rank"].fillna(
        rank_comparison["Current Rank"]
    )

    rank_comparison["Current Rank"] = rank_comparison["Current Rank"].astype(int)
    rank_comparison["Baseline Rank"] = rank_comparison["Baseline Rank"].astype(int)

    rank_comparison["Movement"] = (
        rank_comparison["Baseline Rank"] - rank_comparison["Current Rank"]
    )

    current_sales = current_totals.rename("Amount")

    final_df = pd.concat(
        [current_sales, rank_comparison["Current Rank"], rank_comparison["Movement"]],
        axis=1,
    ).reset_index()

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
    """Build the HTML body for the daily email."""
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
        change_text = "Not enough sales data in the last week for a meaningful comparison."

    top_commodity = today_metrics["top_commodity_name"]
    top_commodity_amount = today_metrics["top_commodity_amount"]

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
        <h2>Zarai Mandi Daily Report – {report_date}</h2>
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
    """
    Send the HTML report email to multiple recipients.
    NOTE: Requires SMTP_USER/SMTP_PASS in st.secrets to work for email.
    """
    try:
        # Note: This still relies on st.secrets for SMTP credentials.
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
        # Load the raw data using the cached function
        raw_df_local, _ = cached_load_data()
        if raw_df_local.empty:
            st.error("Cannot send email: Data load failed.")
            return False
            
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
    msg["Subject"] = f"[V2] Zarai Mandi Daily Sales Report – {report_date}"
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
