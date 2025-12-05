import streamlit as st
import pandas as pd
import numpy as np
import gspread
from datetime import date
from typing import Dict, Any, List

# --- CONSTANTS ---
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f" # e.g., 10,000

# --- DATA LOADING AND CLEANING ---

@st.cache_data(ttl=300)
def load_data(use_cache=True, sheet_name="Historical Data") -> pd.DataFrame:
    """
    Load data from a Google Sheet worksheet.
    
    Args:
        use_cache (bool): Whether to use Streamlit's cache (always True for this function).
        sheet_name (str): The name of the worksheet/tab to load data from.
        
    Returns:
        pd.DataFrame: The loaded and partially cleaned DataFrame.
    """
    try:
        # 1. Connect to Google Sheets
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        spreadsheet_url = st.secrets["spreadsheet_url"]
        
        sh = gc.open_by_url(spreadsheet_url)
        
        # 2. Open the specific worksheet
        worksheet = sh.worksheet(sheet_name) 
        
        # 3. Get all data and convert to DataFrame
        data = worksheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # 4. General Cleaning and Type Conversion
        
        # Drop rows that are entirely empty
        df = df.dropna(how='all')
        
        # Standardize column names (optional, but good practice)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # Convert date column (assuming 'month' for Finance Historical and 'date' for others)
        if 'date' in df.columns:
            # Assumes 'date' column is used in the main sheets, e.g., '11/5/2023'
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False).dt.date
        
        # Convert amount column: remove commas and convert to numeric
        amount_col = None
        if 'amount' in df.columns:
            amount_col = 'amount'
        elif 'amount_pkr' in df.columns:
            amount_col = 'amount_pkr'

        if amount_col:
            df[amount_col] = pd.to_numeric(
                df[amount_col].astype(str).str.replace(',', ''), errors='coerce'
            )
            df[amount_col] = df[amount_col].fillna(0)
            
        # Clean up 'month' or 'commodities_list' (Type) for Finance Historical
        if 'month' in df.columns:
            df['month'] = df['month'].astype(str).str.strip()
        if 'commodities_list' in df.columns:
            df['commodities_list'] = df['commodities_list'].astype(str).str.strip()
            
        return df.dropna(subset=['amount_pkr'])

    except Exception as e:
        # In a real app, you might log this error more formally
        st.error(f"Error loading data from sheet '{sheet_name}': {e}")
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
    df_temp['commodities_list'] = df_temp['commodities_list'].fillna('')
    
    # 2. Split the list by comma and explode
    df_temp['commodity_items'] = df_temp['commodities_list'].str.split(',').apply(
        lambda x: [item.strip() for item in x if item.strip()]
    )
    df_exploded = df_temp.explode('commodity_items')
    
    # 3. Extract the amount and commodity name
    
    # Regex to capture amount (first number) and the rest as commodity name
    # e.g., '200 Edible oil' -> amount=200, name='Edible oil'
    # This regex is robust for the format seen in the screenshot
    pattern = r'^\s*(\d+)\s*(.*)'
    
    df_exploded[['amount_split', 'commodity']] = df_exploded['commodity_items'].str.extract(pattern)
    
    df_exploded['amount_split'] = pd.to_numeric(
        df_exploded['amount_split'], errors='coerce'
    ).fillna(0)
    
    # Clean up commodity name
    df_exploded['commodity'] = df_exploded['commodity'].str.strip()
    
    # Calculate the gross amount per commodity using the split amount as a weight.
    # Total amount is split proportionally IF the row total is not 0
    
    # Calculate the sum of the split amounts for normalization
    sum_split_per_txn = df_exploded.groupby(df_exploded.index)['amount_split'].transform('sum')
    
    # Calculate the gross amount per commodity using the proportion of split amounts 
    # to the total transaction amount (amount_pkr).
    df_exploded['gross_amount_per_commodity'] = np.where(
        (sum_split_per_txn > 0) & (df_exploded['amount_pkr'] > 0),
        df_exploded['amount_split'] / sum_split_per_txn * df_exploded['amount_pkr'],
        df_exploded['amount_pkr'] / df_exploded.groupby(df_exploded.index)['commodity_items'].transform('count')
    )
    
    # Fallback: if sum_split_per_txn is 0 or amount_pkr is 0, distribute equally among items
    # (The numpy where handles the primary case. The else clause is a simple division 
    # for the equal distribution fallback, but the logic above should cover most proportional splits)
    
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
    unique_customers = raw_filtered["customer_name"].nunique()
    
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
