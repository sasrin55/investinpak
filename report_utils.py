import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Any, List

# --- CONSTANTS ---
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f" # e.g., 10,000

# --- DATA LOADING AND CLEANING ---

@st.cache_data(ttl=300)
def load_data(sheet_url: str) -> pd.DataFrame:
    """
    Loads data directly from the CSV export URL of a public Google Sheet 
    (assumes sheet access is set to "Anyone with the link").
    
    Args:
        sheet_url (str): The full CSV export URL for the specific worksheet (GID).
    """
    try:
        # Load the sheet directly into a pandas DataFrame
        df = pd.read_csv(sheet_url)
        
        # Standardize column names (to lowercase and underscores)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # --- Column Mapping and Cleaning ---
        
        # Standardize month/date column names for the pipeline
        if 'month' in df.columns:
            df.rename(columns={'month': 'month_str'}, inplace=True)
        if 'date' in df.columns:
            df.rename(columns={'date': 'date_str'}, inplace=True)

        # Standardize amount column names
        if 'amount' in df.columns:
            df.rename(columns={'amount': 'amount_pkr'}, inplace=True)
            
        # Standardize the commodity list column
        if 'type' in df.columns:
            df.rename(columns={'type': 'commodities_list'}, inplace=True)
            
        # Convert date columns to datetime objects (for main dashboard)
        if 'date_str' in df.columns:
            # We use dayfirst=False for the standard M/D/Y format
            df['date'] = pd.to_datetime(df['date_str'], errors='coerce', dayfirst=False).dt.date
            
        # Convert amount column to numeric, handling commas and fill NaN with 0
        if 'amount_pkr' in df.columns:
            df['amount_pkr'] = pd.to_numeric(
                df['amount_pkr'].astype(str).str.replace(',', ''), errors='coerce'
            ).fillna(0)
            
        # If 'month' column is used (for Finance Historicals), we just clean it here.
        # Conversion to datetime object (Month_DT) is handled on the page level.

        # Filter out rows where essential data is missing
        return df.dropna(how='all')

    except Exception as e:
        st.error(f"Error loading data from the provided URL: {e}")
        return pd.DataFrame()


def explode_commodities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe where a column ('commodities_list') contains a comma-separated 
    list of commodity names and amounts (e.g., '200 Edible oil, 300 Paddy').
    It explodes this into one row per commodity.
    
    Expected columns: 'commodities_list' and 'amount_pkr'.
    
    Returns:
        pd.DataFrame: A new dataframe with one row per commodity.
    """
    df_temp = df.copy()
    
    # 1. Clean the list string
    # We must check if the column exists before trying to access it
    if 'commodities_list' not in df_temp.columns or df_temp['commodities_list'].empty:
        # If no commodities_list, return empty to prevent KeyErrors later
        return pd.DataFrame()
        
    df_temp['commodities_list'] = df_temp['commodities_list'].fillna('')
    
    # 2. Split the list by comma and explode
    df_temp['commodity_items'] = df_temp['commodities_list'].str.split(',').apply(
        lambda x: [item.strip() for item in x if item.strip()]
    )
    df_exploded = df_temp.explode('commodity_items')
    
    # 3. Extract the amount and commodity name
    pattern = r'^\s*(\d+)\s*(.*)'
    
    df_exploded[['amount_split', 'commodity']] = df_exploded['commodity_items'].str.extract(pattern)
    
    df_exploded['amount_split'] = pd.to_numeric(
        df_exploded['amount_split'], errors='coerce'
    ).fillna(0)
    
    # Clean up commodity name
    df_exploded['commodity'] = df_exploded['commodity'].str.strip()
    
    # Calculate the sum of the split amounts for normalization
    sum_split_per_txn = df_exploded.groupby(df_exploded.index)['amount_split'].transform('sum')
    
    # Calculate the gross amount per commodity using the proportion of split amounts 
    # to the total transaction amount (amount_pkr).
    df_exploded['gross_amount_per_commodity'] = np.where(
        (sum_split_per_txn > 0) & (df_exploded['amount_pkr'] > 0),
        df_exploded['amount_split'] / sum_split_per_txn * df_exploded['amount_pkr'],
        df_exploded['amount_pkr'] / df_exploded.groupby(df_exploded.index)['commodity_items'].transform('count')
    )
    
    # Final cleanup and selection
    df_final = df_exploded[df_exploded['commodity'] != ''].reset_index(drop=True)
    
    return df_final


# --- METRIC HELPERS ---

def metric_format(value: float) -> str:
    """Formats a float as a currency string with 0 decimal places."""
    if pd.isna(value):
        return f"{CURRENCY_CODE} 0"
    return f"{CURRENCY_CODE} {value:{CURRENCY_FORMAT}}"


def count_transactions(df: pd.DataFrame, start_date: date, end_date: date) -> int:
    """Counts raw transactions within a date range."""
    if df.empty:
        return 0
    return len(df[(df["date"] >= start_date) & (df["date"] <= end_date)])


def sum_between(df: pd.DataFrame, start_date: date, end_date: date) -> float:
    """Sums the total gross amount (amount_pkr) within a date range."""
    if df.empty:
        return 0.0
    filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    return filtered_df["amount_pkr"].sum()


def get_kpi_metrics(
    raw_df: pd.DataFrame, exploded_df: pd.DataFrame, start_date: date, end_date: date
) -> Dict[str, Any]:
    """
    Calculates key performance indicators for a given date range.
    """
    # 1. Filter the dataframes
    raw_filtered = raw_df[
        (raw_df["date"] >= start_date) & (raw_df["date"] <= end_date)
    ]
    exploded_filtered = exploded_df[
        (exploded_df["date"] >= start_date) & (exploded_df["date"] <= end_date)
    ]

    # 2. Total Sales
    total_amount = raw_filtered["amount_pkr"].sum()

    # 3. Total Transactions (rows in the raw data)
    total_transactions = len(raw_filtered)

    # 4. Unique Customers
    # Note: 'customer_name' column isn't present in Finance Historicals, but is safe here 
    # as this function is primarily used by the main dashboard.
    unique_customers = raw_filtered["customer_name"].nunique() if 'customer_name' in raw_filtered.columns else 0
    
    # 5. Top Commodity
    top_commodity = exploded_filtered.groupby("commodity")[
        "gross_amount_per_commodity"
    ].sum()
    
    top_commodity_name = "N/A"
    top_commodity_amount = "0"

    if not top_commodity.empty:
        top_commodity = top_commodity.sort_values(ascending=False).iloc[0]
        top_commodity_name = top_commodity.name
        top_commodity_amount = metric_format(top_commodity)

    # 6. Return values
    return {
        "df": exploded_filtered,  # Return the exploded filtered DF for summary table rendering
        "total_amount": total_amount,
        "total_transactions": total_transactions,
        "unique_customers": unique_customers,
        "top_commodity_name": top_commodity_name,
        "top_commodity_amount": top_commodity_amount,
    }
