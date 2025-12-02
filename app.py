import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, date
from zoneinfo import ZoneInfo
# Import functions directly from report_utils.py
from report_utils import load_data, explode_commodities, get_kpi_metrics, metric_format, CURRENCY_CODE

# --- HARDCODED RECIPIENTS ---
# The fixed list of recipients you provided
RECIPIENT_EMAILS = [
    "abdul.raafey@zaraimandi.com", 
    "ghasharib.shoukat@gmail.com"
]

def build_daily_email_html(report_date: date, metrics: dict):
    """Builds a simple HTML email for a given date using computed metrics."""
    
    rows_html = ""
    # Get the top 5 commodities table data from the calculated metrics
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
        <h2>Zaraimandi Daily Scheduled Report for {report_date}</h2>
        <p>This report summarizes the day's gross sales activity, calculated at 5 PM PKT.</p>

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

def run_and_send_report(report_date: date):
    """Loads data, computes metrics, and sends the email."""
    try:
        # Load fresh data (bypasses Streamlit cache)
        raw_df = load_data()
        exploded_df = explode_commodities(raw_df)
        
        # Calculate TODAY's metrics 
        metrics = get_kpi_metrics(raw_df, exploded_df, report_date, report_date)
        
        # SMTP credentials pulled from GitHub Actions environment secrets
        smtp_user = os.environ["REPORT_SENDER_EMAIL"]
        smtp_pass = os.environ["REPORT_SENDER_PASS"]
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 465))
        
        html_body = build_daily_email_html(report_date, metrics)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"DAILY REPORT: Zaraimandi Sales for {report_date}"
        msg["From"] = smtp_user
        msg["To"] = ", ".join(RECIPIENT_EMAILS)

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, RECIPIENT_EMAILS, msg.as_string())
        
        print(f"Successfully sent report for {report_date} to {len(RECIPIENT_EMAILS)} recipients.")

    except Exception as e:
        print(f"FATAL ERROR during report run for {report_date}: {e}")
        # In a real system, you'd send an error notification here.

if __name__ == "__main__":
    # Get the current date in Pakistan time (Asia/Karachi)
    try:
        report_date_pk = datetime.now(ZoneInfo("Asia/Karachi")).date()
    except Exception:
        # Fallback if zoneinfo is not perfectly configured
        report_date_pk = datetime.now().date() 

    run_and_send_report(report_date_pk)
