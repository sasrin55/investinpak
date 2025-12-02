import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from report_utils import load_data, explode_commodities, get_kpi_metrics, metric_format, CURRENCY_CODE, CURRENCY_FORMAT
import pandas as pd
import numpy as np

# --- HARDCODED RECIPIENTS ---
RECIPIENT_EMAILS = [
    "abdul.raafey@zaraimandi.com", 
    "ghasharib.shoukat@gmail.com"
]

# --- NEW HELPER FOR COMPARATIVE ANALYSIS ---

def get_commodity_comparisons(exploded_df: pd.DataFrame, report_date: date) -> pd.DataFrame:
    """Calculates WoW and MoM change percentages for the top commodities."""
    
    # Define comparison periods
    current_7_start = report_date - timedelta(days=6)
    previous_7_start = report_date - timedelta(days=13)
    
    current_30_start = report_date - timedelta(days=29)
    previous_30_start = report_date - timedelta(days=59)

    # 1. Base DataFrame for all relevant periods
    mask_current_30 = (exploded_df["date"] >= previous_30_start) & (exploded_df["date"] <= report_date)
    df_base = exploded_df.loc[mask_current_30]

    # 2. Calculate sales for each period
    
    # Current Period Sales (This Week/Month)
    sales_current_week = df_base[(df_base["date"] >= current_7_start) & (df_base["date"] <= report_date)]
    sales_current_month = df_base[(df_base["date"] >= current_30_start) & (df_base["date"] <= report_date)]
    
    # Baseline Sales (Previous Week/Month)
    sales_previous_week = df_base[(df_base["date"] >= previous_7_start) & (df_base["date"] < current_7_start)]
    sales_previous_month = df_base[(df_base["date"] >= previous_30_start) & (df_base["date"] < current_30_start)]

    # 3. Aggregate data by commodity
    
    # Aggregations
    agg_sales = lambda df: df.groupby("commodity")["gross_amount_per_commodity"].sum()
    
    current_week = agg_sales(sales_current_week).rename("Current_Week")
    previous_week = agg_sales(sales_previous_week).rename("Previous_Week")
    current_month = agg_sales(sales_current_month).rename("Current_Month")
    previous_month = agg_sales(sales_previous_month).rename("Previous_Month")

    # Combine all series
    comparison_df = pd.concat([current_week, previous_week, current_month, previous_month], axis=1).fillna(0)

    # 4. Calculate WoW and MoM change percentages
    
    # WoW Change %: (Current Week - Previous Week) / Previous Week
    comparison_df["WoW Change %"] = np.where(
        comparison_df["Previous_Week"] > 0,
        ((comparison_df["Current_Week"] - comparison_df["Previous_Week"]) / comparison_df["Previous_Week"]) * 100,
        np.where(comparison_df["Current_Week"] > 0, 100, 0) 
    ).round(1)

    # MoM Change %: (Current Month - Previous Month) / Previous Month
    comparison_df["MoM Change %"] = np.where(
        comparison_df["Previous_Month"] > 0,
        ((comparison_df["Current_Month"] - comparison_df["Previous_Month"]) / comparison_df["Previous_Month"]) * 100,
        np.where(comparison_df["Current_Month"] > 0, 100, 0)
    ).round(1)
    
    comparison_df = comparison_df.reset_index()

    # Get the Top 10 by current month sales
    comparison_df = comparison_df.sort_values("Current_Month", ascending=False).head(10)

    return comparison_df[["commodity", "Current_Month", "WoW Change %", "MoM Change %"]]


def calculate_rank_movement(exploded_df: pd.DataFrame, report_date: date) -> pd.DataFrame:
    """
    Calculates current YTD ranking and compares it to YTD ranking 30 days prior.
    """
    last_month_end_date = report_date - timedelta(days=30)
    start_of_year = date(report_date.year, 1, 1)

    # 1. Calculate Current YTD Ranking
    current_ytd = exploded_df[(exploded_df["date"] >= start_of_year) & (exploded_df["date"] <= report_date)]
    rank_current = (
        current_ytd.groupby('commodity')['gross_amount_per_commodity'].sum()
        .rank(method='min', ascending=False).astype(int).rename('Current Rank')
    )

    # 2. Calculate Baseline (Previous YTD) Ranking
    baseline_ytd = exploded_df[(exploded_df["date"] >= start_of_year) & (exploded_df["date"] <= last_month_end_date)]
    rank_baseline = (
        baseline_ytd.groupby('commodity')['gross_amount_per_commodity'].sum()
        .rank(method='min', ascending=False).astype(int).rename('Baseline Rank')
    )

    # 3. Combine and calculate movement
    rank_comparison = pd.concat([rank_current, rank_baseline], axis=1).fillna(999) # 999 for items not ranked before
    
    # Movement: Baseline Rank - Current Rank (Positive = moved up, Negative = moved down)
    rank_comparison['Movement'] = rank_comparison['Baseline Rank'] - rank_comparison['Current Rank']
    
    # Get Current Total for display
    current_sales = current_ytd.groupby('commodity')['gross_amount_per_commodity'].sum().rename('Amount')
    
    final_df = pd.concat([current_sales, rank_current, rank_comparison['Movement']], axis=1).dropna(subset=['Current Rank'])
    
    # Format Movement Column
    final_df['Movement Symbol'] = np.where(
        final_df['Movement'] > 0, '▲',
        np.where(final_df['Movement'] < 0, '▼', '—')
    )
    final_df['Rank Change'] = final_df['Movement Symbol'] + ' ' + final_df['Movement'].abs().astype(int).astype(str)
    final_df.loc[final_df['Movement'] == 0, 'Rank Change'] = '—'
    final_df.loc[final_df['Movement'] == 999, 'Rank Change'] = 'New' # Item didn't exist in baseline

    final_df = final_df.sort_values('Amount', ascending=False).head(10).reset_index()
    
    return final_df[['commodity', 'Amount', 'Current Rank', 'Rank Change']]


def build_daily_email_html(report_date: date, today_metrics: dict, last_7_metrics: dict, ytd_metrics: dict, trend_df: pd.DataFrame, rank_df: pd.DataFrame):
    """Builds a robust HTML email with comparative metrics, narrative, and trend tables."""
    
    # --- 1. Comparative Narrative Logic ---
    today_sales = today_metrics["total_amount"]
    last_7_total = last_7_metrics["total_amount"]
    last_7_avg = last_7_total / 7
    
    if last_7_avg > 0:
        change_percent = ((today_sales - last_7_avg) / last_7_avg) * 100
        change_text = f"Today's sales are **{abs(change_percent):.1f}% {'higher' if change_percent >= 0 else 'lower'}** than the average sales of the last 7 days."
    else:
        change_text = "Not enough sales data in the last week for a meaningful comparison."
        
    top_commodity = today_metrics["top_commodity_name"]
    top_commodity_amount = today_metrics["top_commodity_amount"]
    
    # --- 2. WoW/MoM Trend Table HTML ---
    trend_rows_html = ""
    for _, row in trend_df.iterrows():
        wow_style = 'color: #006B3F;' if row['WoW Change %'] >= 0 else 'color: #CC0000;'
        mom_style = 'color: #006B3F;' if row['MoM Change %'] >= 0 else 'color: #CC0000;'
        trend_rows_html += f"""
        <tr>
            <td>{row['commodity']}</td>
            <td style="text-align:right">{metric_format(row['Current_Month'])}</td>
            <td style="text-align:right; {wow_style}">{row['WoW Change %']:,.1f}%</td>
            <td style="text-align:right; {mom_style}">{row['MoM Change %']:,.1f}%</td>
        </tr>
        """
        
    # --- 3. YTD Rank Movement Table HTML ---
    rank_rows_html = ""
    for _, row in rank_df.iterrows():
        movement_symbol = row['Rank Change'].split(' ')[0]
        movement_style = 'color: #006B3F;' if movement_symbol == '▲' else ('color: #CC0000;' if movement_symbol == '▼' else 'color: #666;')
        
        rank_rows_html += f"""
        <tr>
            <td style="text-align:left;">{row['Current Rank']}</td>
            <td style="text-align:left;">{row['commodity']}</td>
            <td style="text-align:right;">{metric_format(row['Amount'])}</td>
            <td style="text-align:right; font-weight:bold; {movement_style}">{row['Rank Change']}</td>
        </tr>
        """

    # --- 4. FINAL HTML OUTPUT ---
    html = f"""
    <html>
    <body>
        <h2>Daily Sales Report: {report_date.strftime('%A, %B %d, %Y')}</h2>
        <p style="font-weight: bold; color: {'#006B3F' if change_percent >= 0 else '#CC0000'}; margin-bottom: 20px;">
            {change_text}
        </p>

        <h3>Key Performance Indicators</h3>
        <table border="0" cellpadding="8" cellspacing="0" style="width:100%; border-collapse:collapse; font-size: 14px;">
            <tr>
                <td style="background-color:#f0f0f0; padding:10px; width: 33%;">
                    <p style="font-weight:bold; margin:0; color:#006B3F;">YTD Sales Total:</p>
                    <h2 style="margin:0; color:#006B3F;">{metric_format(ytd_metrics["total_amount"])}</h2>
                </td>
                <td style="background-color:#f0f0f0; padding:10px; width: 33%;">
                    <p style="font-weight:bold; margin:0;">Today's Gross Sales:</p>
                    <h2 style="margin:0;">{metric_format(today_metrics["total_amount"])}</h2>
                </td>
                <td style="background-color:#f0f0f0; padding:10px; width: 33%;">
                    <p style="font-weight:bold; margin:0;">Last 7 Days Sales:</p>
                    <h2 style="margin:0;">{metric_format(last_7_metrics["total_amount"])}</h2>
                </td>
            </tr>
        </table>
        
        <br>

        <h3>Today's Activity Highlights</h3>
        <p style="font-size: 15px;">
            The top-selling commodity today was <b>{top_commodity}</b>, generating <b>{top_commodity_amount}</b> in sales.
        </p>
        
        <br>
        
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="width: 50%; vertical-align: top; padding-right: 20px;">
                    <h3>YTD Top 10 Rank Movement</h3>
                    <table border="1" cellpadding="6" cellspacing="0" style="width: 100%; border-collapse:collapse; font-size: 14px;">
                        <tr>
                            <th style="background-color:#e0e0e0; text-align:left; width: 15%;">Rank</th>
                            <th style="background-color:#e0e0e0; text-align:left; width: 40%;">Commodity</th>
                            <th style="background-color:#e0e0e0; text-align:right; width: 30%;">YTD Amount</th>
                            <th style="background-color:#e0e0e0; text-align:right; width: 15%;">MoM Change</th>
                        </tr>
                        {rank_rows_html}
                    </table>
                </td>
                <td style="width: 50%; vertical-align: top;">
                    <h3>Top 10 MoM/WoW Trend</h3>
                    <table border="1" cellpadding="6" cellspacing="0" style="width: 100%; border-collapse:collapse; font-size: 14px;">
                        <tr>
                            <th style="background-color:#e0e0e0; text-align:left;">Commodity</th>
                            <th style="background-color:#e0e0e0; text-align:right;">Current Month</th>
                            <th style="background-color:#e0e0e0; text-align:right;">WoW %</th>
                            <th style="background-color:#e0e0e0; text-align:right;">MoM %</th>
                        </tr>
                        {trend_rows_html}
                    </table>
                </td>
            </tr>
        </table>


        <p style="margin-top:20px; font-size: 12px; color:#666;">
            <a href="YOUR_STREAMLIT_APP_URL_HERE" target="_blank">Open Full Interactive Dashboard</a>
        </p>
    </body>
    </html>
    """
    return html

def run_and_send_report(report_date: date):
    """Loads data, computes metrics, and sends the email."""
    try:
        # 1. Load data
        raw_df = load_data()
        exploded_df = explode_commodities(raw_df)
        
        # 2. Calculate comparison dates
        last_7_days_start = report_date - timedelta(days=6)
        start_of_year = date(report_date.year, 1, 1)

        # 3. Calculate all necessary KPI metrics
        today_metrics = get_kpi_metrics(raw_df, exploded_df, report_date, report_date)
        last_7_metrics = get_kpi_metrics(raw_df, exploded_df, last_7_days_start, report_date)
        ytd_metrics = get_kpi_metrics(raw_df, exploded_df, start_of_year, report_date)
        
        # 4. Calculate WoW/MoM comparison table (Trend)
        trend_df = get_commodity_comparisons(exploded_df, report_date)
        
        # 5. Calculate YTD Rank Movement table
        rank_df = calculate_rank_movement(exploded_df, report_date)
        
        # 6. Setup SMTP
        smtp_user = os.environ["REPORT_SENDER_EMAIL"]
        smtp_pass = os.environ["REPORT_SENDER_PASS"]
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 465))
        
        # 7. Build HTML using all computed metrics
        html_body = build_daily_email_html(report_date, today_metrics, last_7_metrics, ytd_metrics, trend_df, rank_df)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"DAILY REPORT: Zaraimandi Sales for {report_date}"
        msg["From"] = smtp_user
        msg["To"] = ", ".join(RECIPIENT_EMAILS)

        msg.attach(MIMEText(html_body, "html"))

        # 8. Send Email
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, RECIPIENT_EMAILS, msg.as_string())
        
        print(f"Successfully sent report for {report_date} to {len(RECIPIENT_EMAILS)} recipients.")

    except Exception as e:
        print(f"FATAL ERROR during report run for {report_date}: {e}")

if __name__ == "__main__":
    # Get the current date in Pakistan time (Asia/Karachi)
    try:
        report_date_pk = datetime.now(ZoneInfo("Asia/Karachi")).date()
    except Exception:
        # Fallback if zoneinfo is not perfectly configured
        report_date_pk = datetime.now().date() 

    run_and_send_report(report_date_pk)
