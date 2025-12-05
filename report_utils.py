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
    """
    try:
        df = pd.read_csv(sheet_url)
        
        # Standardize column names (to lowercase and underscores, removing special characters)
        df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') for col in df.columns]
        
        # --- Column Mapping and Cleaning ---
        
        # 1. Map Time/Date Column: 'timestamp' -> 'date_str'
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'date_str'}, inplace=True)
            
        # 2. Map Commodity Column: 'commodity' or 'type' -> 'commodities_list'
        if 'commodity' in df.columns:
            df.rename(columns={'commodity': 'commodities_list'}, inplace=True)
        if 'type' in df.columns:
            df.rename(columns={'type': 'commodities_list'}, inplace=True)

        # 3. Map Amount Column: 'amount' -> 'amount_pkr'
        if 'amount' in df.columns:
            df.rename(columns={'amount': 'amount_pkr'}, inplace=True)
            
        # 4. Map Month Column (for historical sheet only): 'month' -> 'month_str'
        if 'month' in df.columns:
            df.rename(columns={'month': 'month_str'}, inplace=True)


        # --- Final Data Type Conversions ---

        # Convert date columns to datetime objects (for main dashboard)
        if 'date_str' in df.columns:
            df['date'] = pd.to_datetime(df['date_str'], errors='coerce', dayfirst=False).dt.date
            
        # Convert amount column to numeric, handling commas and fill NaN with 0
        if 'amount_pkr' in df.columns:
            df['amount_pkr'] = pd.to_numeric(
                df['amount_pkr'].astype(str).str.replace(',', ''), errors='coerce'
            ).fillna(0)
            
        return df.dropna(how='all')

    except Exception as e:
        st.error(f"Error loading data from the provided URL: {e}")
        return pd.DataFrame()


def explode_commodities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe where a column ('commodities_list') contains a comma-separated 
    list of commodity names and amounts. Explodes this into one row per commodity.
    
    Includes robust logic for proportional or equal splitting of amount_pkr.
    """
    df_temp = df.copy()

    required_cols = ['commodities_list', 'amount_pkr']
    if not all(col in df_temp.columns for col in required_cols):
        return pd.DataFrame(columns=required_cols)

    df_temp['commodities_list'] = df_temp['commodities_list'].fillna('')
    if df_temp['commodities_list'].empty:
         return pd.DataFrame(columns=df_temp.columns)
         
    # 1. Split the list by comma and explode
    df_temp['commodity_items'] = df_temp['commodities_list'].str.split(',').apply(
        lambda x: [item.strip() for item in x if item.strip()]
    )
    df_exploded = df_temp.explode('commodity_items')
    
    # 2. Extract the amount (if present) and commodity name
    # We look for a leading number, followed by the rest of the string as the name.
    pattern = r'^\s*(\d+)\s*(.*)'
    df_exploded[['amount_split', 'commodity']] = df_exploded['commodity_items'].str.extract(pattern)
    
    df_exploded['amount_split'] = pd.to_numeric(
        df_exploded['amount_split'], errors='coerce'
    ).fillna(0)
    
    # Clean up commodity name by removing extra text that was NOT the leading number
    df_exploded['commodity'] = df_exploded['commodity'].str.strip()
    
    # 3. Determine split method and calculate gross_amount_per_commodity
    
    # Calculate the sum of the extracted split amounts for this transaction row
    sum_split_per_txn = df_exploded.groupby(df_exploded.index)['amount_split'].transform('sum')
    
    # Calculate the count of items in the list for equal splitting fallback
    item_count_per_txn = df_exploded.groupby(df_exploded.index)['commodity_items'].transform('count')
    
    # The proportional split (preferred if amounts exist):
    proportional_split = df_exploded['amount_split'] / sum_split_per_txn * df_exploded['amount_pkr']
    
    # The equal split (fallback if no split amounts were extracted OR if sum is 0):
    equal_split = df_exploded['amount_pkr'] / item_count_per_txn

    # Final calculation: Use proportional split if amounts are present and valid (>0); otherwise use equal split.
    df_exploded['gross_amount_per_commodity'] = np.where(
        (sum_split_per_txn > 0) & (df_exploded['amount_pkr'] > 0),
        proportional_split,
        equal_split
    )
    
    # 4. Final Cleanup: Select columns and filter out empty commodity names
    original_cols = list(df.columns)
    new_cols = ['commodity', 'gross_amount_per_commodity']
    
    # Filter out rows where commodity name is empty
    df_final = df_exploded[df_exploded['commodity'] != '']
    
    # Select original columns + new calculation columns
    final_col_selection = original_cols + [col for col in new_cols if col in df_final.columns and col not in original_cols]
    
    df_final = df_final[final_col_selection].reset_index(drop=True)
    
    return df_final


# --- METRIC HELPERS (The rest of the file remains unchanged) ---

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
