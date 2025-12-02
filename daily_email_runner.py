# daily_email_runner.py

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime_text import MIMEText
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from report_utils import (
    load_data,
    explode_commodities,
    get_kpi_metrics,
    metric_format,
    CURRENCY_CODE,
)

# -------------------------------------------------------------------
# HELPERS FOR COMPARISONS & RANK MOVEMENT
# -------------------------------------------------------------------

def get_commodity_comparisons(exploded_df: pd.DataFrame, report_date: date) -> pd.DataFrame:
    """WoW and MoM % for top commodities (same logic as dashboard)."""
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
        [current_week, previous_week, current_month, previous_month],
        axis=1
    ).fillna(0)

    comparison_df["WoW Change %"] = np.where(
        comparison_df["Previous_Week"] > 0,
        (comparison_df["Current_Week"] - comparison_df["Previous_Week"])
        / comparison_df["Previous_Week"] * 100,
        np.where(comparison_df["Current_Week"] > 0, 100, 0),
    ).round(1)

    comparison_df["MoM Change %"] = np.where(
        comparison_df["Previous_Month"] > 0,
        (comparison_df["Current_Month"] - comparison_df["Previous_Month"])
        / comparison_df["Previous_Month"] * 100,
        np.where(comparison_df["Current_Month"] > 0, 100, 0),
    ).round(1)

    comparison_df = comparison_df.reset_index()
    comparison_df = comparison_df.sort_values("Current_Month", ascending=False).head(10)

    return comparison_df[["commodity", "Current_Month", "WoW Change %", "MoM Change %"]]


def calculate_rank_movement(exploded_df: pd.DataFrame, report_date: date) -> pd.DataFrame:
    """NaN-safe YTD rank movement vs 30 days ago."""
    last_month_end_date = report_date - timedelta(days=30)
    start_of_year = date(report_date.year, 1, 1)

    current_ytd = exploded_df[
        (exploded_df["date"] >= start_of_year)
        & (exploded_df["date"] <= report_date)
    ]
    if current_ytd.empty:
        return pd.DataFrame(columns=["commodity", "Amount", "Current Rank", "Rank Change"])

    current_totals = current_ytd.groupby("commodity")["gross_amount_per_commodity"].sum()
    rank_current = current_totals.rank(method="min", ascending=False)

    baseline_ytd = exploded_df[
        (exploded_df["date"] >= start_of_year)
        & (exploded_df["date"] <= last_month_end_date)
    ]
    baseline_totals = baseline_ytd.groupby("commodity")["gross_amount_per_commodity"].sum()
    rank_baseline = baseline_totals.rank(method="min", ascending=False)

    rank_comparison = pd.concat(
        [rank_current.rename("Current Rank"), rank_baseline.rename("Baseline Rank")],
        axis=1,
    )

    # If no baseline, use current rank so movement starts at 0
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

# -------------------------------------------------------------------
# EMAIL HTML
# -------------------------------------------------------------------

def build_daily_email_html(
    report_date: date,
    today_metrics: dict,
    last_7_metrics: dict,
    ytd_metrics: dict,
    trend_df: pd.DataFrame,
    rank_df: pd.DataFrame,
) -> str:
    """Build the HTML email body."""
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

# -------------------------------------------------------------------
# SENDING LOGIC
# -------------------------------------------------------------------

def get_recipients_from_env() -> list:
    """
    Read REPORT_RECIPIENT_EMAIL from env and turn into a clean list.
    Falls back to your three default emails if the env var is missing.
    """
    raw = os.environ.get("REPORT_RECIPIENT_EMAIL", "")
    if not raw.strip():
        return [
            "abdul.raafey@zaraimandi.com",
            "ghasharib.shoukat@gmail.com",
            "raahimshoukat99@gmail.com",
        ]
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return parts


def send_email_report(report_date: date) -> None:
    """Load data, build metrics, and send the email using GitHub Actions env vars."""
    # These names match your workflow: SMTP_USER / SMTP_PASS / SMTP_SERVER / SMTP_PORT
    smtp_user = os.environ["SMTP_USER"]
    smtp_pass = os.environ["SMTP_PASS"]
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "465"))

    recipients = get_recipients_from_env()
    if not recipients:
        raise RuntimeError("No recipients found in REPORT_RECIPIENT_EMAIL or fallback list.")

    # Load data
    raw_df = load_data(use_cache=False)
    exploded_df = explode_commodities(raw_df)

    # Metrics
    today_metrics = get_kpi_metrics(raw_df, exploded_df, report_date, report_date)
    last_7_metrics = get_kpi_metrics(
        raw_df, exploded_df, report_date - timedelta(days=6), report_date
    )
    ytd_metrics = get_kpi_metrics(
        raw_df, exploded_df, date(report_date.year, 1, 1), report_date
    )

    trend_df = get_commodity_comparisons(exploded_df, report_date)
    rank_df = calculate_rank_movement(exploded_df, report_date)

    html_body = build_daily_email_html(
        report_date, today_metrics, last_7_metrics, ytd_metrics, trend_df, rank_df
    )

    # Compose and send
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Zaraimandi Daily Sales Report – {report_date}"
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipients, msg.as_string())


def main():
    # Use Pakistan time for the report date
    report_date = datetime.now(ZoneInfo("Asia/Karachi")).date()
    send_email_report(report_date)


if __name__ == "__main__":
    main()
