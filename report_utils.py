import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Any

# --- CONSTANTS ---
CURRENCY_CODE = "PKR"
CURRENCY_FORMAT = ",.0f"  # e.g., 10,000


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
        df = pd.read_csv(sheet_url)

        # Standardize column names (lowercase + underscores, no parentheses)
        df.columns = [
            col.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            for col in df.columns
        ]

        # --- Column mapping / normalisation ---

        # 1) Time column -> date_str
        if "timestamp" in df.columns:
            df.rename(columns={"timestamp": "date_str"}, inplace=True)

        # 2) Commodity columns -> commodities_list
        #    (master sheet: "commodity"; finance sheet: "type")
        if "commodity" in df.columns:
            df.rename(columns={"commodity": "commodities_list"}, inplace=True)
        if "type" in df.columns and "commodities_list" not in df.columns:
            df.rename(columns={"type": "commodities_list"}, inplace=True)

        # 3) Amount column -> amount_pkr
        if "amount" in df.columns:
            df.rename(columns={"amount": "amount_pkr"}, inplace=True)

        # 4) Month column (for historical sheet) -> month_str
        if "month" in df.columns:
            df.rename(columns={"month": "month_str"}, inplace=True)

        # --- Type conversions ---

        # Convert date_str -> date (python date object)
        if "date_str" in df.columns:
            df["date"] = pd.to_datetime(
                df["date_str"], errors="coerce", dayfirst=False
            ).dt.date

        # Convert amount_pkr to numeric, stripping commas
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
    list of commodity names and amounts (e.g., '200 Edible oil, 300 Paddy').
    It explodes this into one row per commodity.

    Expected columns: 'commodities_list' and 'amount_pkr'.

    Returns:
        pd.DataFrame: A new dataframe with one row per commodity and all original
        columns preserved (including 'date' if present), plus:
        - 'commodity'
        - 'gross_amount_per_commodity'
    """
    df_temp = df.copy()
    original_cols = list(df_temp.columns)

    # Ensure the app never crashes if columns are missing:
    required_cols = ["commodities_list", "amount_pkr"]
    if not all(col in df_temp.columns for col in required_cols):
        # Return an empty dataframe with all original columns plus expected new ones
        empty_cols = original_cols + ["commodity", "gross_amount_per_commodity"]
        return pd.DataFrame(columns=empty_cols)

    # Handle case where list column exists but is entirely empty
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
    # Sum split amounts per original transaction (by index)
    sum_split_per_txn = df_exploded.groupby(df_exploded.index)[
        "amount_split"
    ].transform("sum")

    # Where parsing succeeded: use proportional allocation of amount_pkr
    # Else: evenly split amount_pkr across commodity_items count
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
    final_cols = original_cols + [
        col for col in new_cols if col not in original_cols
    ]

    df_final = df_final[final_cols].reset_index(drop=True)

    # Final check for 'date' column presence (apps rely on this)
    if "date" not in df_final.columns:
        # Not fatal, but warn once so you know data won't be time-filterable
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

    # 4. Unique Customers
    unique_customers = (
        raw_filtered["customer_name"].nunique()
        if "customer_name" in raw_filtered.columns
        else 0
    )

    # 5. Top Commodity
    top_commodity_series = (
        exploded_filtered.groupby("commodity")["gross_amount_per_commodity"].sum()
        if not exploded_filtered.empty and "commodity" in exploded_filtered.columns
        else pd.Series(dtype=float)
    )

    top_commodity_name = "N/A"
    top_commodity_amount = "0"

    if not top_commodity_series.empty:
        top_value = top_commodity_series.sort_values(ascending=False).iloc[0]
        top_commodity_name = top_commodity_series.sort_values(
            ascending=False
        ).index[0]
        top_commodity_amount = metric_format(top_value)

    return {
        "df": exploded_filtered,
        "total_amount": total_amount,
        "total_transactions": total_transactions,
        "unique_customers": unique_customers,
        "top_commodity_name": top_commodity_name,
        "top_commodity_amount": top_commodity_amount,
    }
