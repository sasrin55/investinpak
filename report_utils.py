import pandas as pd
import numpy as np
import re
from datetime import date, timedelta

# --- SETTINGS ---
SHEET_ID = "1kTy_-jB_cPfvXN-Lqe9WMSD-moeI-OF5kE4PbMN7M1Q"
TAB_NAME = "Master"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={TAB_NAME}"
CURRENCY_CODE = "PKR"

# --- HELPER FUNCTIONS ---

def sum_between(df, start, end, amount_col="amount_pkr"):
    """Calculates the GROSS sum of amount between two dates (inclusive)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask, amount_col].sum()

def count_transactions(df, start, end):
    """Counts the number of unique transactions (rows in the raw data)."""
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].shape[0]

def metric_format(value):
    return f"{CURRENCY_CODE} {value:,.0f}"

# --- DATA FUNCTIONS ---

def load_data(use_cache=True):
    """Reads data, renames columns, and cleans data types."""
    # Note: st.cache_data is removed here because this file runs outside Streamlit's context
    try:
        df = pd.read_csv(CSV_URL)
    except Exception as e:
        # In a cron job, this will just log the error
        raise Exception(f"Error loading data from Google Sheet: {e}")

    cols = df.columns
    if len(cols) < 6:
        raise ValueError("Data structure error: Expected at least 6 columns.")

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
    """Explodes commodities and calculates gross amount per commodity."""
    if base_df.empty or "commodities" not in base_df.columns:
        return base_df.copy()

    temp = base_df.copy()
    temp["commodities"] = temp["commodities"].fillna("").astype(str)
    temp["commodities_clean"] = (
        temp["commodities"]
        .str.replace(r"[\s]*[&\/]| and ", ",", regex=True)
        .str.strip()
    )
    temp["commodity_list"] = temp["commodities_clean"].apply(
        lambda s: [x.strip() for x in s.split(",") if x.strip() != ""]
    )
    
    def normalize_commodity_name(name):
        if not name: return None
        name = name.lower().strip()
        name_clean = re.sub(r'[^\w\s]', '', name)
        name_clean = re.sub(r'\s+', ' ', name_clean).strip()
        if not name_clean: return None
            
        NON_COMMODITY_KEYWORDS = [
            'unknown', 'not confirm yet', 'discussion', 'contact sale', 
            'live market', 'data', 'group', 'fruits', 'fruit', 'vegetables', 'vegetable' 
        ]
        if any(keyword in name_clean for keyword in NON_COMMODITY_KEYWORDS):
            return None

        mapping = {
            'cotton': 'Cotton', 'coton': 'Cotton', 'cottonmillet': 'Cotton Millet',
            'paddy': 'Paddy', 'padd': 'Paddy', 'wheat': 'Wheat', 'wheatandpaddy': 'Wheat & Paddy',
            'edibleoil': 'Edible Oil', 'fertilizer': 'Fertilizer', 
            'pulses': 'Pulses', 'daal': 'Pulses', 'bajra': 'Bajra', 'livestock': 'Livestock', 
            'sesame': 'Sesame', 'sugar': 'Sugar', 'sugarsugar': 'Sugar', 'sugarwheat': 'Sugar + Wheat',
            'mustard': 'Mustard', 'mustrad': 'Mustard', 'kiryana': 'Kiryana',
            'dryfruits': 'Dry Fruits', 'spices': 'Spices', 'rice': 'Rice', 'maize': 'Maize', 'dates': 'Dates'
        }
        
        if name_clean in mapping:
            return mapping[name_clean]
        return name_clean.title()

    temp["commodity_list"] = temp["commodity_list"].apply(
        lambda lst: [normalize_commodity_name(item) for item in lst if normalize_commodity_name(item) is not None]
    )
    temp["n_commodities"] = temp["commodity_list"].apply(
        lambda lst: len(lst) if len(lst) > 0 else np.nan
    )
    temp = temp.explode("commodity_list")
    temp = temp[temp["commodity_list"].notna() & (temp["n_commodities"].notna())]
    temp = temp.rename(columns={"commodity_list": "commodity"})
    
    temp["gross_amount_per_commodity"] = temp["amount_pkr"] / temp["n_commodities"]
    
    return temp[["date", "customer_name", "txn_type", "commodity", "gross_amount_per_commodity", "amount_pkr"]]


def get_kpi_metrics(raw_df, exploded_df, start_date, end_date):
    """Calculates all metrics for a given period."""
    
    # Total Amount: Calculated directly from RAW data for Gross Sum accuracy
    raw_period_mask = (raw_df["date"] >= start_date) & (raw_df["date"] <= end_date)
    total_amount = raw_df.loc[raw_period_mask, "amount_pkr"].sum()

    # Metrics calculated from exploded data
    period_mask = (exploded_df["date"] >= start_date) & (exploded_df["date"] <= end_date)
    period_df = exploded_df.loc[period_mask].copy()

    total_transactions = count_transactions(raw_df, start_date, end_date)
    unique_customers = period_df["customer_name"].nunique()

    # Calculate Top Commodity
    top_commodity_series = period_df.groupby("commodity")["gross_amount_per_commodity"].sum().nlargest(1)
    top_commodity_name = top_commodity_series.index[0] if not top_commodity_series.empty else "N/A"
    top_commodity_amount = metric_format(top_commodity_series.iloc[0]) if not top_commodity_series.empty else "N/A"

    return {
        "df": period_df,
        "total_amount": total_amount,
        "total_transactions": total_transactions,
        "unique_customers": unique_customers,
        "top_commodity_name": top_commodity_name,
        "top_commodity_amount": top_commodity_amount
    }
