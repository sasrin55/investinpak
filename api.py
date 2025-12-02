# api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
from analytics_service import get_dashboard_data

app = FastAPI()

# Allow any frontend (Blazor/React/Streamlit) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/dashboard")
def dashboard(date_str: str | None = None):
    """
    Returns KPI blocks: today, last7, last30, ytd
    """
    if date_str:
        report_date = date.fromisoformat(date_str)
    else:
        report_date = date.today()

    return get_dashboard_data(report_date)
