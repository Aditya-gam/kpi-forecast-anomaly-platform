#!/usr/bin/env python3
"""
combine_datasets.py
-------------------
This script integrates two raw Retail Sales Data (UCI) datasets:
  1. data/raw/online_retail_II.xlsx
  2. data/raw/Online Retail.xlsx

These datasets differ slightly in their column names:
  - The first dataset uses columns: "Invoice", "Price", "Customer ID", etc.
  - The second dataset uses columns: "InvoiceNo", "UnitPrice", "CustomerID", etc.

This script normalizes the column names across datasets by:
  - Renaming "Invoice" in the first dataset to "InvoiceNo".
  - Renaming "Price" in the first dataset to "UnitPrice".
  - Renaming "Customer ID" in the first dataset to "CustomerID".

Once normalized, the two datasets are concatenated (row-wise) and stored as a CSV file
in the `data/interim/` directory under a specified filename (default: combined_retail_data.csv).

Usage:
    python combine_datasets.py [--input1 FILE1] [--input2 FILE2] [--output OUTPUT_FILE]

Arguments:
    --input1    Path to first dataset (default: "data/raw/online_retail_II.xlsx")
    --input2    Path to second dataset (default: "data/raw/Online Retail.xlsx")
    --output    Path to store the combined dataset (default: "data/interim/combined_retail_data.csv")

Environment Variables:
    DATA_PATH   Base directory for raw data (used to construct default file paths).

Dependencies:
    - pandas: For data manipulation and integration.
    - openpyxl: To read Excel files.
    - Standard libraries: os, sys, argparse, logging

Author: Aditya Gambhir
"""

import os
import sys
import argparse
import logging
import pandas as pd

# Configure logging to print info to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default file paths (using DATA_PATH environment variable for raw data)
DATA_PATH = os.getenv("DATA_PATH", "./data/raw")
DEFAULT_INPUT1 = os.path.join(DATA_PATH, "online_retail_II.xlsx")
DEFAULT_INPUT2 = os.path.join(DATA_PATH, "Online Retail.xlsx")
DEFAULT_OUTPUT = os.path.join("data", "interim", "combined_retail_data.csv")


def normalize_columns(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    """
    Normalize column names to a common standard.

    For the first dataset (dataset_label 'dataset1'), expected columns are:
      - "Invoice" should become "InvoiceNo"
      - "Price" should become "UnitPrice"
      - "Customer ID" should become "CustomerID"

    For the second dataset, columns are assumed to already use the standard names.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        dataset_label (str): Identifier for which dataset is being normalized ('dataset1' or 'dataset2').

    Returns:
        pd.DataFrame: DataFrame with normalized column names.
    """
    if dataset_label.lower() == 'dataset1':
        rename_map = {
            "Invoice": "InvoiceNo",
            "Price": "UnitPrice",
            "Customer ID": "CustomerID"
        }
        df = df.rename(columns=rename_map)
        logger.info(f"Normalized column names for dataset1: {rename_map}")
    else:
        logger.info("Dataset2 assumed to have standard column names.")
    return df


def load_and_normalize(input_path: str, dataset_label: str) -> pd.DataFrame:
    """
    Load an Excel dataset and normalize its columns.

    Parameters:
        input_path (str): Path to the Excel file.
        dataset_label (str): Identifier ('dataset1' or 'dataset2').

    Returns:
        pd.DataFrame: Loaded and normalized dataframe.
    """
    try:
        df = pd.read_excel(input_path)
        logger.info(
            f"Loaded dataset '{dataset_label}' from {input_path} with shape: {df.shape}")
    except Exception as e:
        logger.error(
            f"Error loading dataset '{dataset_label}' from {input_path}: {e}")
        sys.exit(1)

    df = normalize_columns(df, dataset_label)

    # Convert InvoiceDate column to datetime if present.
    if 'InvoiceDate' in df.columns:
        try:
            df['InvoiceDate'] = pd.to_datetime(
                df['InvoiceDate'], errors='coerce')
        except Exception as e:
            logger.warning(
                f"Could not convert InvoiceDate to datetime in {dataset_label}: {e}")
    return df


def combine_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combine two dataframes vertically after ensuring both have the same columns.

    Parameters:
        df1 (pd.DataFrame): First dataset.
        df2 (pd.DataFrame): Second dataset.

    Returns:
        pd.DataFrame: The combined dataframe.
    """
    # Identify common columns (intersection) and fill missing ones with NaN
    common_columns = list(set(df1.columns).intersection(set(df2.columns)))
    df1 = df1.reindex(columns=common_columns)
    df2 = df2.reindex(columns=common_columns)
    logger.info(f"Common columns for combination: {common_columns}")

    combined_df = pd.concat([df1, df2], ignore_index=True)
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    return combined_df


def save_combined_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the combined dataframe as a CSV file to the specified output path.

    Parameters:
        df (pd.DataFrame): The combined dataframe.
        output_path (str): The CSV file path to save the data.
    """
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Combined dataset saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving combined dataset: {e}")
        sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Combine two raw Retail Sales Data (UCI) datasets into one CSV file."
    )
    parser.add_argument(
        "--input1",
        type=str,
        default=DEFAULT_INPUT1,
        help=f"Path to the first dataset (default: {DEFAULT_INPUT1})."
    )
    parser.add_argument(
        "--input2",
        type=str,
        default=DEFAULT_INPUT2,
        help=f"Path to the second dataset (default: {DEFAULT_INPUT2})."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to save the combined CSV file (default: {DEFAULT_OUTPUT})."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to load, normalize, combine, and save the datasets.
    """
    args = parse_arguments()
    logger.info("Starting dataset combination process...")

    # Load and normalize each dataset.
    df1 = load_and_normalize(args.input1, "dataset1")
    df2 = load_and_normalize(args.input2, "dataset2")

    # Combine both datasets.
    combined_df = combine_datasets(df1, df2)

    # Save the combined dataset.
    save_combined_dataset(combined_df, args.output)

    logger.info("Dataset combination process completed successfully.")


if __name__ == "__main__":
    main()
