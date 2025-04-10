#!/usr/bin/env python3
"""
examine_dataset.py
-------------------
This script examines the Retail Sales Data (UCI) dataset stored in the raw data directory
(e.g., /data/raw/online_retail_II.xlsx) and generates a comprehensive report that includes:

1. Basic Dataset Overview:
    - Dataset shape, column types, missing and unique values.
    - Statistical summaries for numeric columns.
    - Domain-specific analysis for the 'InvoiceDate' column.
    - A preview of the first five rows.

2. Extended Insights:
    - Revenue calculation (Price x Quantity) for each transaction.
    - Monthly revenue trends with a summary table.
    - Top 10 products by quantity sold and by revenue.
    - Country-level analysis for number of transactions and total revenue.
    - Correlation analysis among numeric features.

3. Visualizations:
    - Distribution histograms for Quantity and Price.
    - Line plot of monthly revenue trends.
    - Correlation heatmap for numeric features.
    
All visualizations are saved in the './reports/plots' directory, and their file paths are
included in the final report.

Usage:
    python examine_dataset.py [--input INPUT_FILE] [--output REPORT_FILE]

Arguments:
    --input     Path to the dataset Excel file.
                (Default: "./data/raw/online_retail_II.xlsx")
    --output    Path where the generated report will be saved.
                (Default: "./data/raw/dataset_report.txt")

Environment Variables:
    DATA_PATH   Base directory for raw data (used as a default if not provided via arguments).

Dependencies:
    - pandas (for data manipulation)
    - matplotlib and seaborn (for plotting)
    - openpyxl (to read Excel files)
    - Standard libraries: os, sys, argparse, logging

Author: Aditya Gambhir
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging for real-time feedback
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define default file paths based on the DATA_PATH environment variable.
DEFAULT_INPUT = os.path.join(
    os.getenv("DATA_PATH", "./data/raw"), "online_retail_II.xlsx")
DEFAULT_OUTPUT = os.path.join(
    os.getenv("DATA_PATH", "./data/reports"), "dataset_report.txt")
PLOTS_DIR = os.path.join("data", "reports", "plots")


def generate_basic_report(df: pd.DataFrame) -> str:
    """
    Generate a basic overview report of the dataset.

    Parameters:
        df (pd.DataFrame): The loaded dataset.

    Returns:
        str: A formatted string containing the basic report.
    """
    lines = []
    divider = "=" * 70
    lines.append("Retail Sales Data (UCI) - Basic Overview Report")
    lines.append(divider)
    lines.append(f"Dataset Shape (rows x columns): {df.shape}\n")

    lines.append("Column Information:")
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        lines.append(
            f" - {col:15s}: dtype={dtype}, missing={missing:5d}, unique={unique:5d}")
    lines.append("")

    # Statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        lines.append("Statistical Summary for Numeric Columns:")
        lines.append(df[numeric_cols].describe().to_string())
        lines.append("")

    # Domain-specific: InvoiceDate analysis
    if 'InvoiceDate' in df.columns:
        lines.append("InvoiceDate Analysis:")
        try:
            df['InvoiceDate'] = pd.to_datetime(
                df['InvoiceDate'], errors='coerce')
            min_date = df['InvoiceDate'].min()
            max_date = df['InvoiceDate'].max()
            lines.append(f" - Earliest Invoice Date: {min_date}")
            lines.append(f" - Latest Invoice Date  : {max_date}\n")
        except Exception as e:
            lines.append(f" - Error converting InvoiceDate: {e}\n")

    # Cancellation pattern based on Invoice column (instead of InvoiceNo)
    if 'Invoice' in df.columns:
        cancellation_count = df['Invoice'].astype(
            str).str.startswith('C').sum()
        total = df.shape[0]
        lines.append("Invoice Cancellation Analysis:")
        lines.append(f" - Total Invoices       : {total}")
        lines.append(
            f" - Cancellation Invoices: {cancellation_count} ({(cancellation_count/total)*100:.2f}%)\n")

    # Data preview
    lines.append("Sample Data (First 5 Rows):")
    lines.append(divider)
    lines.append(df.head().to_string())
    lines.append(divider)

    return "\n".join(lines)


def generate_extended_insights(df: pd.DataFrame) -> str:
    """
    Generate extended insights from the dataset:
      - Compute revenue per transaction.
      - Summarize monthly revenue trends.
      - Identify top 10 products by quantity sold and revenue.
      - Country-level analysis.
      - Correlation analysis among numeric features.

    Parameters:
        df (pd.DataFrame): The loaded dataset.

    Returns:
        str: A formatted string containing extended insights.
    """
    lines = []
    divider = "=" * 70
    lines.append("Extended Insights & Analysis")
    lines.append(divider)

    # Compute revenue per transaction (ensure Price and Quantity exist)
    if 'Price' in df.columns and 'Quantity' in df.columns:
        df['Revenue'] = df['Price'] * df['Quantity']
        lines.append("Revenue Calculation:")
        lines.append(
            " - Revenue (Price x Quantity) has been calculated for each transaction.\n")
    else:
        lines.append(
            "Revenue column could not be computed (Price or Quantity missing).\n")

    # Monthly revenue trends using InvoiceDate
    if 'InvoiceDate' in df.columns:
        df['Month'] = df['InvoiceDate'].dt.to_period("M")
        monthly_rev = df.groupby('Month')['Revenue'].sum().reset_index()
        monthly_rev['Month'] = monthly_rev['Month'].astype(str)
        lines.append("Monthly Revenue Summary:")
        lines.append(monthly_rev.to_string(index=False))
        lines.append("")
    else:
        lines.append(
            "InvoiceDate column missing; cannot compute monthly revenue trends.\n")

    # Top products by quantity and revenue using Description (if available)
    if 'Description' in df.columns:
        product_stats = df.groupby('Description').agg(
            Total_Quantity=('Quantity', 'sum'),
            Total_Revenue=('Revenue', 'sum'),
            # Fixed: using 'Invoice' instead of 'InvoiceNo'
            Transaction_Count=('Invoice', 'count')
        ).reset_index()
        top10_qty = product_stats.sort_values(
            by='Total_Quantity', ascending=False).head(10)
        top10_rev = product_stats.sort_values(
            by='Total_Revenue', ascending=False).head(10)
        lines.append("Top 10 Products by Total Quantity Sold:")
        lines.append(top10_qty.to_string(index=False))
        lines.append("")
        lines.append("Top 10 Products by Total Revenue:")
        lines.append(top10_rev.to_string(index=False))
        lines.append("")
    else:
        lines.append(
            "Description column missing; skipping product analysis.\n")

    # Country-level analysis for transactions and revenue
    if 'Country' in df.columns:
        country_stats = df.groupby('Country').agg(
            Transactions=('Invoice', 'count'),
            Total_Revenue=('Revenue', 'sum')
        ).reset_index().sort_values(by='Total_Revenue', ascending=False)
        lines.append("Country-Level Analysis (by Total Revenue):")
        lines.append(country_stats.to_string(index=False))
        lines.append("")
    else:
        lines.append(
            "Country column missing; skipping country-level analysis.\n")

    # Correlation analysis among numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr()
        lines.append("Correlation Matrix for Numeric Features:")
        lines.append(corr_matrix.to_string())
        lines.append("")
    else:
        lines.append("No numeric columns found for correlation analysis.\n")

    return "\n".join(lines)


def save_plots(df: pd.DataFrame) -> dict:
    """
    Create and save visualizations for key dataset characteristics.
    Plots created:
      1. Quantity Distribution
      2. Price Distribution
      3. Monthly Revenue Trend (if InvoiceDate available)
      4. Correlation Heatmap for numeric features

    Parameters:
        df (pd.DataFrame): The loaded dataset.

    Returns:
        dict: A dictionary mapping plot descriptions to file paths.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_files = {}

    # Plot 1: Distribution of Quantity
    plt.figure(figsize=(8, 6))
    plt.hist(df['Quantity'], bins=50, edgecolor='black', color='skyblue')
    plt.title("Distribution of Quantity")
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    quantity_path = os.path.join(PLOTS_DIR, "quantity_distribution.png")
    plt.savefig(quantity_path, bbox_inches='tight')
    plt.close()
    plot_files["Quantity Distribution"] = quantity_path

    # Plot 2: Distribution of Price
    plt.figure(figsize=(8, 6))
    plt.hist(df['Price'], bins=50, edgecolor='black', color='salmon')
    plt.title("Distribution of Price")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    price_path = os.path.join(PLOTS_DIR, "price_distribution.png")
    plt.savefig(price_path, bbox_inches='tight')
    plt.close()
    plot_files["Price Distribution"] = price_path

    # Plot 3: Monthly Revenue Trend (if InvoiceDate and Revenue exist)
    if 'InvoiceDate' in df.columns and 'Revenue' in df.columns:
        df['Month'] = df['InvoiceDate'].dt.to_period("M")
        monthly_rev = df.groupby('Month')['Revenue'].sum().reset_index()
        monthly_rev['Month'] = monthly_rev['Month'].astype(str)
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_rev['Month'],
                 monthly_rev['Revenue'], marker='o', linestyle='-')
        plt.title("Monthly Revenue Trend")
        plt.xlabel("Month")
        plt.ylabel("Total Revenue")
        plt.xticks(rotation=45)
        monthly_rev_path = os.path.join(PLOTS_DIR, "monthly_revenue_trend.png")
        plt.savefig(monthly_rev_path, bbox_inches='tight')
        plt.close()
        plot_files["Monthly Revenue Trend"] = monthly_rev_path

    # Plot 4: Correlation Heatmap for numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True,
                    cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Correlation Heatmap")
        corr_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        plt.savefig(corr_path, bbox_inches='tight')
        plt.close()
        plot_files["Correlation Heatmap"] = corr_path

    return plot_files


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Examine the Retail Sales Data (UCI) dataset and generate an extended detailed report with insights and plots."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to the dataset Excel file (default: {DEFAULT_INPUT})."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to save the generated report (default: {DEFAULT_OUTPUT})."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to load the dataset, generate a detailed report with extended insights and plots,
    and write the complete report to a file.
    """
    args = parse_arguments()
    logger.info("Starting extended dataset examination.")
    logger.info(f"Loading dataset from: {args.input}")

    try:
        df = pd.read_excel(args.input)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    logger.info("Dataset loaded successfully.")

    # Generate basic and extended sections of the report.
    basic_report = generate_basic_report(df)
    extended_report = generate_extended_insights(df)

    # Create and save plots.
    logger.info("Generating and saving plots...")
    plot_files = save_plots(df)
    plots_section = "Plots Generated:\n" + \
        "\n".join(f" - {desc}: {path}" for desc, path in plot_files.items())

    # Combine all report sections.
    full_report = "\n\n".join([basic_report, extended_report, plots_section])

    # Ensure the output directory exists.
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(full_report)
        logger.info(f"Report written successfully to: {args.output}")
    except Exception as e:
        logger.error(f"Error writing report: {e}")
        sys.exit(1)

    logger.info("Extended dataset examination completed successfully.")


if __name__ == "__main__":
    main()
