import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Any
from urllib.parse import urlparse, parse_qs # Step 1: Add new import

# --- CONSTANTS ---
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f"  # e.g., 10,000


# --- DATA LOADING HELPERS ---

def normalize_gsheet_url(sheet_url: str) -> str: # Step 2: Add helper function
    """
    Converts a normal Google Sheets URL (edit URL) into a direct CSV export URL.
    If the URL is already a CSV URL or a non-Google URL, it is returned as-is.
    """
    if not sheet_url:
        return sheet_url

    # If it's already an export CSV link, just return it
    if "export?format=csv" in sheet_url:
        return sheet_url

    if "docs.google.com/spreadsheets" in sheet_url:
        parsed = urlparse(sheet_url)
        path_parts = parsed.path.split("/")

        file_id = None
        for i, part in enumerate(path_parts):
            if part == "d" and i + 1 < len(path_parts):
                file_id = path_parts[i + 1]
                break

        # default gid to 0 if missing
        query = parse_qs(parsed.query)
        # Prioritize gid from the original query parameters
        gid = query.get("gid", ["0"])[0] 
        
        # Check for fragment gid if it's an edit URL (like the one provided)
        if "#gid" in parsed.fragment:
            fragment_query = parse_qs(parsed.fragment.lstrip('#'))
            gid = fragment_query.get("gid", [gid])[0]

        if file_id:
            return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid={gid}"

    # Fallback: return unchanged
    return sheet_url


@st.cache_data(ttl=300)
def load_data(sheet_url: str) -> pd.DataFrame: # Step 3: Replace existing load_data
    """
    Loads data from a Google Sheet (edit URL or CSV URL) and normalises it
    for the 'Master for SalesOps' structure:
      - Date          -> date_str / date
      - Customer      -> customer_name
      - Customer type -> customer_type
      - Commodity     -> commodities_list
      - Amount        -> amount_pkr
      - Duration      -> duration (left as-is)
    """
    try:
        # Convert a normal Google Sheets URL into a CSV export URL automatically
        csv_url = normalize_gsheet_url(sheet_url)

        df = pd.read_csv(csv_url)

        # Standardize column names (lowercase + underscores, no parentheses)
        df.columns = [
            col.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            for col in df.columns
        ]

        renames = {}

        # Date column (from 'Master for SalesOps' -> "date")
        if "date" in df.columns:
            renames["date"] = "date_str"
        if "start_date" in df.columns and "date_str" not in renames and "date_str" not in df.columns:
            renames["start_date"] = "date_str"
        if "timestamp" in df.columns and "date_str" not in renames and "date_str" not in df.columns:
            renames["timestamp"] = "date_str"

        # Customer -> customer_name (The phone number)
        if "customer" in df.columns:
            renames["customer"] = "customer_name"

        # Commodity -> commodities_list (used by explode_commodities)
        if "commodity" in df.columns:
            renames["commodity"] = "commodities_list"

        # Amount -> amount_pkr (numeric sales amount)
        if "amount" in df.columns:
            renames["amount"] = "amount_pkr"

        # Month -> month_str (for historical sheets, optional)
        if "month" in df.columns:
            renames["month"] = "month_str"
            
        # The 'customer_type' column does not need a rename as 'customer_type' is already the clean name

        # Apply all renames
        if renames:
            df.rename(columns=renames, inplace=True)

        # ---- Type conversions ----

        # date_str -> date (python date object)
        if "date_str" in df.columns:
            df["date"] = pd.to_datetime(
                df["date_str"],
                errors="coerce",
                dayfirst=True,
                infer_datetime_format=True,
            ).dt.date

        # Ensure amount_pkr is numeric
        if "amount_pkr" in df.columns:
            df["amount_pkr"] = pd.to_numeric(
                df["amount_pkr"].astype(str).str.replace(",", ""),
                errors="coerce",
            ).fillna(0)

        # Drop fully empty rows
        df = df.dropna(how="all")

        return df

    except Exception as e:
        st.error(f"Error loading data from the provided URL: {e}")
        return pd.DataFrame()


def explode_commodities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe where a column ('commodities_list') contains a comma-separated
    list of commodity names and amounts and explodes this into one row per commodity.
    """
    df_temp = df.copy()
    original_cols = list(df_temp.columns)

    # Ensure the app never crashes if columns are missing:
    required_cols = ["commodities_list", "amount_pkr"]
    if not all(col in df_temp.columns for col in required_cols):
        empty_cols = original_cols + ["commodity", "gross_amount_per_commodity"]
        return pd.DataFrame(columns=empty_cols)

    # If list column exists but is entirely empty, return empty with schema
    if df_temp["commodities_list"].isna().all():
        empty_cols = original_cols + ["commodity", "gross_amount_per_commodity"]
        return pd.DataFrame(columns=empty_cols)

    # 1. Clean the list string
    df_temp["commodities_list"] = df_temp["commodities_list"].fillna("")

    # 2. Split the list by comma and explode
    df_temp["commodity_items"] = df_temp["commodities_list"].str.split(",").apply(
        lambda x: [item.strip() for item in x if str(item).strip()]
    )
    df_exploded = df_temp.explode("commodity_items")

    # 3. Extract numeric amount and commodity name from each item
    pattern = r"^\s*(\d+)\s*(.*)"
    df_exploded[["amount_split", "commodity"]] = df_exploded[
        "commodity_items"
    ].str.extract(pattern)

    df_exploded["amount_split"] = pd.to_numeric(
        df_exploded["amount_split"], errors="coerce"
    ).fillna(0)

    # Clean commodity text
    df_exploded["commodity"] = df_exploded["commodity"].astype(str).str.strip()

    # 4. Compute proportional split of transaction amount across commodities
    sum_split_per_txn = df_exploded.groupby(df_exploded.index)[
        "amount_split"
    ].transform("sum")
    count_items_per_txn = df_exploded.groupby(df_exploded.index)[
        "commodity_items"
    ].transform("count")

    df_exploded["gross_amount_per_commodity"] = np.where(
        (sum_split_per_txn > 0) & (df_exploded["amount_pkr"] > 0),
        df_exploded["amount_split"] / sum_split_per_txn * df_exploded["amount_pkr"],
        np.where(
            count_items_per_txn > 0,
            df_exploded["amount_pkr"] / count_items_per_txn,
            0,
        ),
    )

    # 5. Filter out rows where commodity is empty
    df_final = df_exploded[df_exploded["commodity"] != ""].copy()

    # 6. Keep all original columns + new ones
    new_cols = ["commodity", "gross_amount_per_commodity"]
    final_cols = original_cols + [col for col in new_cols if col not in original_cols]

    df_final = df_final[final_cols].reset_index(drop=True)

    # Final check for 'date' column presence (apps rely on this)
    if "date" not in df_final.columns:
        st.warning(
            "Warning: The 'date' column is missing from the exploded data. "
            "Time-based filters and KPIs will not work correctly for this dataset."
        )

    return df_final


# --- METRIC HELPERS ---

def metric_format(value: float) -> str:
    """Formats a float as a currency string with 0 decimal places."""
    if pd.isna(value):
        return f"{CURRENCY_CODE} 0"
    return f"{CURRENCY_CODE} {value:{CURRENCY_FORMAT}}"


def count_transactions(df: pd.DataFrame, start_date: date, end_date: date) -> int:
    """Counts raw transactions within a date range."""
    if df.empty or "date" not in df.columns:
        return 0
    return len(df[(df["date"] >= start_date) & (df["date"] <= end_date)])


def sum_between(df: pd.DataFrame, start_date: date, end_date: date) -> float:
    """Sums the total gross amount (amount_pkr) within a date range."""
    if df.empty or "date" not in df.columns:
        return 0.0
    filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
    return filtered_df.get("amount_pkr", pd.Series(dtype=float)).sum()


def get_kpi_metrics(
    raw_df: pd.DataFrame, exploded_df: pd.DataFrame, start_date: date, end_date: date
) -> Dict[str, Any]:
    """
    Calculates key performance indicators for a given date range.
    Uses 'customer_name' (which holds the phone number) for unique customer count.
    """
    # 1. Filter the dataframes
    if "date" in raw_df.columns:
        raw_filtered = raw_df[
            (raw_df["date"] >= start_date) & (raw_df["date"] <= end_date)
        ]
    else:
        raw_filtered = raw_df.iloc[0:0]

    if "date" in exploded_df.columns:
        exploded_filtered = exploded_df[
            (exploded_df["date"] >= start_date) & (exploded_df["date"] <= end_date)
        ]
    else:
        exploded_filtered = exploded_df.iloc[0:0]

    # 2. Total Sales
    total_amount = raw_filtered.get("amount_pkr", pd.Series(dtype=float)).sum()

    # 3. Total Transactions
    total_transactions = len(raw_filtered)

    # 4. Unique Customers (Determined by phone number, stored in 'customer_name' column)
    unique_customers = (
        raw_filtered["customer_name"].nunique()
        if "customer_name" in raw_filtered.columns
        else 0
    )

    # 5. Top Commodity
    if (
        not exploded_filtered.empty
        and "commodity" in exploded_filtered.columns
        and "gross_amount_per_commodity" in exploded_filtered.columns
    ):
        top_commodity_series = exploded_filtered.groupby("commodity")[
            "gross_amount_per_commodity"
        ].sum()
    else:
        top_commodity_series = pd.Series(dtype=float)

    top_commodity_name = "N/A"
    top_commodity_amount = "0"

    if not top_commodity_series.empty:
        top_sorted = top_commodity_series.sort_values(ascending=False)
        top_commodity_name = top_sorted.index[0]
        top_commodity_amount = metric_format(top_sorted.iloc[0])

    return {
        "df": exploded_filtered,
        "total_amount": total_amount,
        "total_transactions": total_transactions,
        "unique_customers": unique_customers,
        "top_commodity_name": top_commodity_name,
        "top_commodity_amount": top_commodity_amount,
    }
