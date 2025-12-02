# analytics_service.py

from datetime import date, timedelta
from report_utils import (
    load_data,
    explode_commodities,
    get_kpi_metrics,
)

def get_dashboard_data(report_date: date):
    """Returns all KPI blocks (today, last7, last30, ytd) as pure data."""
    
    # load and prepare data
    raw_df = load_data(use_cache=False)
    exploded_df = explode_commodities(raw_df)

    today = report_date
    last_7 = report_date - timedelta(days=6)
    last_30 = report_date - timedelta(days=29)
    start_of_year = date(report_date.year, 1, 1)

    return {
        "today": get_kpi_metrics(raw_df, exploded_df, today, today),
        "last7": get_kpi_metrics(raw_df, exploded_df, last_7, today),
        "last30": get_kpi_metrics(raw_df, exploded_df, last_30, today),
        "ytd": get_kpi_metrics(raw_df, exploded_df, start_of_year, today),
    }
