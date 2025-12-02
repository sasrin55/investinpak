import pandas as pd
import streamlit as st
from datetime import date, timedelta
import altair as alt
import numpy as np
import re

# ==============================================================================
# 1. CONFIGURATION AND INITIAL SETUP
# ==============================================================================

# --- SETTINGS ---
SHEET_ID = "1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q"
TAB_NAME = "Master"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={TAB_NAME}"
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f" 

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
# 2. DATA LOADING AND CLEANUP
# ==============================================================================

@st.cache_data(show_spinner="Connecting to Data Source and Loading...")
def load_data():
    """Reads data, renames columns, and cleans data types."""
    try:
        df = pd.read_csv(CSV_URL)
    except Exception as e:
        st.error(f"Error connecting to Google Sheet. Check ID, tab name, and permissions. Details: {e}")
        st.stop()
        return pd.DataFrame()

    cols = df.columns
    if len(cols) < 6:
        st.error(f"Data structure error: Expected at least 6 columns, found {len(cols)}. Check the '{TAB_NAME}' tab.")
        st.stop()

    rename_map = {
        cols[0]: "date", cols[1]: "customer_name", cols[2]: "phone",
        cols[3]: "txn_type", cols[4]: "commodities", cols[5]: "amount_pkr",
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
    with ENHANCED cleaning for de-duplication and non-commodity removal.
    """
    if base_df.empty or "commodities" not in base_df.columns:
        return base_df.copy()

    temp = base_df.copy()
    temp["commodities"] = temp["commodities"].fillna("").astype(str)

    # Normalize separators to commas
    temp["commodities_clean"] = (
        temp["commodities"]
        .str.replace(r"[\s]*[&\/]| and ", ",", regex=True)
        .str.strip()
    )

    # Build list of commodities
    temp["commodity_list"] = temp["commodities_clean"].apply(
        lambda s: [x.strip() for x in s.split(",") if x.strip() != ""]
    )
    
    # --- AGGRESSIVE COMMODITY CLEANING/NORMALIZATION ---
    def normalize_commodity_name(name):
        if not name:
            return None
        
        # 1. Convert to lowercase and strip punctuation/extra spaces
        name = name.lower().strip()
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        if not name:
            return None
            
        # Hardcoded list of items to REMOVE (non-commodities or internal tracking)
        NON_COMMODITY_KEYWORDS = [
            'unknown', 'not confirm yet', 'discussion', 'contact sale', 
            'live market', 'data', 'group', 'g', 'um', 'l m', 'lm', 
            'fruits', 'fruit', 'vegetables', 'vegetable' 
        ]
        
        if any(keyword in name for keyword in NON_COMMODITY_KEYWORDS):
            return None

        # Consolidation mapping for common misspellings/variants
        mapping = {
            'cotton': 'Cotton', 'coton': 'Cotton', 'cottonmillet': 'Cotton Millet',
            'paddy': 'Paddy', 'padd': 'Paddy',
            'wheat': 'Wheat', 'wheatandpaddy': 'Wheat & Paddy',
            'edibleoil': 'Edible Oil',
            'fertilizer': 'Fertilizer', 
            'pulses': 'Pulses', 'daal': 'Pulses',
            'bajra': 'Bajra',
            'livestock': 'Livestock',
