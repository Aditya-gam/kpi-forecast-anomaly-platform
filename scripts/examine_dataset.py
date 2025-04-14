#!/usr/bin/env python3
"""
examine_dataset.py
-------------------
This script examines the Retail Sales Data (UCI) dataset and generates a comprehensive report 
with extended insights and additional plots. The script accepts both Excel (.xlsx) and CSV (.csv) 
files based on the file extension provided. It standardizes the dataset’s columns to the final 
list:

    ['StockCode', 'Quantity', 'UnitPrice', 'InvoiceNo', 'CustomerID', 'InvoiceDate', 'Description', 'Country']

The report includes:

1. Basic Dataset Overview:
    - Dataset shape and standardized column information (data types, missing and unique values).
    - Statistical summaries for numeric columns.
    - Domain-specific analysis (e.g., InvoiceDate conversion and range, invoice cancellation).
    - A preview of sample records.

2. Extended Insights:
    - Revenue calculation (UnitPrice x Quantity) for each transaction.
    - Monthly revenue trends with a summary table.
    - Top 10 products by quantity sold and by revenue.
    - Country-level analysis (transaction counts and revenue).
    - Correlation analysis among numeric features.

3. Additional Visualizations:
    - Histograms for Quantity and UnitPrice distributions.
    - Enhanced Monthly Revenue Trend plot with annotations.
    - Correlation heatmap.
    - Scatter plot for UnitPrice vs. Quantity with a regression line.
    - Boxplots for UnitPrice and Quantity.
    - Pairplot for all numeric features.
    - Time series plots of average UnitPrice and average Quantity over months.

All plots are stored in the './data/reports/plots1' directory, and their file paths are included 
in the final report.

Usage:
    python examine_dataset.py [--input INPUT_FILE] [--output REPORT_FILE]

Arguments:
    --input     Path to the dataset file (either .xlsx or .csv). 
                (Default: "./data/raw/online_retail_II.xlsx")
    --output    Path where the generated report will be saved.
                (Default: "./data/reports/dataset_report.txt")

Environment Variables:
    DATA_PATH   Base directory for raw data (used as a default if not provided).

Dependencies:
    - pandas, numpy, matplotlib, seaborn, openpyxl (for Excel files)
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

# Configure logging for real-time feedback.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Final standardized list of column names.
STANDARD_COLUMNS = ['StockCode', 'Quantity', 'UnitPrice', 'InvoiceNo',
                    'CustomerID', 'InvoiceDate', 'Description', 'Country']

# Define default file paths using the DATA_PATH environment variable.
DEFAULT_INPUT = os.path.join(
    os.getenv("DATA_PATH", "./data/raw"), "combined_retail_data.csv")
DEFAULT_OUTPUT = os.path.join(
    os.getenv("DATA_PATH", "./data/reports"), "combined_dataset_report.txt")
PLOTS_DIR = os.path.join("data", "reports", "plots_final")


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from either an Excel (.xlsx) or CSV (.csv) file.

    Parameters:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        ValueError: If file extension is not recognized.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            logger.info(f"Dataset loaded from Excel: {file_path}")
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            logger.info(f"Dataset loaded from CSV: {file_path}")
        else:
            raise ValueError(
                f"Unsupported file extension: {ext}. Use .xlsx or .csv.")
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        sys.exit(1)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the dataframe's columns to match the final list:
    ['StockCode', 'Quantity', 'UnitPrice', 'InvoiceNo', 'CustomerID', 'InvoiceDate', 'Description', 'Country'].

    If some standard columns are missing from the dataframe, they are added with NaN values.
    If there are extra columns, they are dropped.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: DataFrame with standardized columns and in the defined order.
    """
    # Rename known variants.
    rename_map = {
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID"
    }
    df = df.rename(columns=rename_map)

    # Select only the standard columns.
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            logger.warning(
                f"Standard column '{col}' missing in the data; adding it with NaN values.")
            df[col] = np.nan
    df = df[STANDARD_COLUMNS]
    logger.info("Columns standardized.")
    return df


def generate_basic_report(df: pd.DataFrame) -> str:
    """
    Generate a basic overview report of the dataset.

    Parameters:
        df (pd.DataFrame): The loaded and standardized dataset.

    Returns:
        str: A formatted string with the basic report.
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
            f" - {col:12s}: dtype={dtype}, missing={missing:5d}, unique={unique:5d}")
    lines.append("")

    # Statistical summary for numeric columns.
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        lines.append("Statistical Summary for Numeric Columns:")
        lines.append(df[numeric_cols].describe().to_string())
        lines.append("")

    # InvoiceDate analysis.
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

    # Invoice cancellation analysis.
    if 'InvoiceNo' in df.columns:
        cancellation_count = df['InvoiceNo'].astype(
            str).str.startswith('C').sum()
        total = df.shape[0]
        lines.append("Invoice Cancellation Analysis:")
        lines.append(f" - Total Invoices       : {total}")
        lines.append(
            f" - Cancellation Invoices: {cancellation_count} ({(cancellation_count/total)*100:.2f}%)\n")

    # Sample data preview.
    lines.append("Sample Data (First 5 Rows):")
    lines.append(divider)
    lines.append(df.head().to_string())
    lines.append(divider)

    return "\n".join(lines)


def generate_extended_insights(df: pd.DataFrame) -> str:
    """
    Generate extended insights from the dataset:
     - Calculate revenue per transaction.
     - Summarize monthly revenue trends.
     - Identify top 10 products by total quantity and revenue.
     - Country-level analysis for transactions and revenue.
     - Correlation analysis among numeric features.

    Parameters:
        df (pd.DataFrame): The loaded and standardized dataset.

    Returns:
        str: A formatted string with extended insights.
    """
    lines = []
    divider = "=" * 70
    lines.append("Extended Insights & Analysis")
    lines.append(divider)

    # Compute revenue.
    if 'UnitPrice' in df.columns and 'Quantity' in df.columns:
        df['Revenue'] = df['UnitPrice'] * df['Quantity']
        lines.append("Revenue Calculation:")
        lines.append(
            " - Revenue (UnitPrice x Quantity) computed for each transaction.\n")
    else:
        lines.append(
            " - Could not compute Revenue (UnitPrice or Quantity missing).\n")

    # Monthly revenue trends.
    if 'InvoiceDate' in df.columns:
        df['Month'] = df['InvoiceDate'].dt.to_period("M")
        monthly_rev = df.groupby('Month')['Revenue'].sum().reset_index()
        monthly_rev['Month'] = monthly_rev['Month'].astype(str)
        lines.append("Monthly Revenue Summary:")
        lines.append(monthly_rev.to_string(index=False))
        lines.append("")
    else:
        lines.append(
            " - InvoiceDate missing; monthly revenue trends unavailable.\n")

    # Top 10 products.
    if 'Description' in df.columns:
        product_stats = df.groupby('Description').agg(
            Total_Quantity=('Quantity', 'sum'),
            Total_Revenue=('Revenue', 'sum'),
            Transaction_Count=('InvoiceNo', 'count')
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
            " - Description column missing; skipping product analysis.\n")

    # Country-level analysis.
    if 'Country' in df.columns:
        country_stats = df.groupby('Country').agg(
            Transactions=('InvoiceNo', 'count'),
            Total_Revenue=('Revenue', 'sum')
        ).reset_index().sort_values(by='Total_Revenue', ascending=False)
        lines.append("Country-Level Analysis (by Total Revenue):")
        lines.append(country_stats.to_string(index=False))
        lines.append("")
    else:
        lines.append(
            " - Country column missing; skipping country-level analysis.\n")

    # Correlation analysis.
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr()
        lines.append("Correlation Matrix for Numeric Features:")
        lines.append(corr_matrix.to_string())
        lines.append("")
    else:
        lines.append(" - No numeric columns found for correlation analysis.\n")

    return "\n".join(lines)


def save_plots(df: pd.DataFrame) -> dict:
    """
    Create and save a suite of visualizations for the dataset.

    Plots created:
      1. Histogram for Quantity distribution.
      2. Histogram for UnitPrice distribution.
      3. Enhanced Monthly Revenue Trend with annotations.
      4. Correlation Heatmap for numeric features.
      5. Scatter Plot (UnitPrice vs. Quantity) with regression line.
      6. Boxplots for UnitPrice and Quantity.
      7. Pairplot for all numeric features.
      8. Time series of average UnitPrice and average Quantity per month.

    Parameters:
        df (pd.DataFrame): The loaded and standardized dataset.

    Returns:
        dict: Mapping of plot description to file path.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_files = {}

    # Plot 1: Quantity Distribution Histogram.
    plt.figure(figsize=(8, 6))
    plt.hist(df['Quantity'], bins=50, edgecolor='black', color='skyblue')
    plt.title("Distribution of Quantity")
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    quantity_path = os.path.join(PLOTS_DIR, "quantity_distribution.png")
    plt.savefig(quantity_path, bbox_inches='tight')
    plt.close()
    plot_files["Quantity Distribution"] = quantity_path

    # Plot 2: UnitPrice Distribution Histogram.
    plt.figure(figsize=(8, 6))
    plt.hist(df['UnitPrice'], bins=50, edgecolor='black', color='salmon')
    plt.title("Distribution of UnitPrice")
    plt.xlabel("UnitPrice")
    plt.ylabel("Frequency")
    price_path = os.path.join(PLOTS_DIR, "price_distribution.png")
    plt.savefig(price_path, bbox_inches='tight')
    plt.close()
    plot_files["UnitPrice Distribution"] = price_path

    # Plot 3: Enhanced Monthly Revenue Trend.
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
        # Annotate peak revenue.
        peak_idx = monthly_rev['Revenue'].idxmax()
        plt.annotate("Peak Revenue", xy=(monthly_rev.loc[peak_idx, 'Month'], monthly_rev.loc[peak_idx, 'Revenue']),
                     xytext=(peak_idx, monthly_rev['Revenue'].max()*0.9),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        monthly_rev_path = os.path.join(
            PLOTS_DIR, "monthly_revenue_trend_enhanced.png")
        plt.savefig(monthly_rev_path, bbox_inches='tight')
        plt.close()
        plot_files["Monthly Revenue Trend"] = monthly_rev_path

    # Plot 4: Correlation Heatmap.
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

    # Plot 5: Scatter Plot (UnitPrice vs. Quantity) with regression line.
    plt.figure(figsize=(8, 6))
    sns.regplot(x='UnitPrice', y='Quantity', data=df,
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title("Scatter Plot: UnitPrice vs. Quantity")
    scatter_path = os.path.join(PLOTS_DIR, "price_vs_quantity_scatter.png")
    plt.savefig(scatter_path, bbox_inches='tight')
    plt.close()
    plot_files["UnitPrice vs. Quantity Scatter"] = scatter_path

    # Plot 6: Boxplots for UnitPrice and Quantity.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['UnitPrice'], color='lightgreen')
    plt.title("Boxplot: UnitPrice")
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Quantity'], color='lightblue')
    plt.title("Boxplot: Quantity")
    boxplot_path = os.path.join(PLOTS_DIR, "price_quantity_boxplots.png")
    plt.savefig(boxplot_path, bbox_inches='tight')
    plt.close()
    plot_files["UnitPrice & Quantity Boxplots"] = boxplot_path

    # Plot 7: Pairplot for Numeric Features.
    pairplot_path = os.path.join(PLOTS_DIR, "numeric_pairplot.png")
    sns.pairplot(df[numeric_cols], diag_kind="kde")
    plt.savefig(pairplot_path, bbox_inches='tight')
    plt.close()
    plot_files["Numeric Features Pairplot"] = pairplot_path

    # Plot 8: Time Series of Average UnitPrice and Average Quantity per Month.
    if 'InvoiceDate' in df.columns:
        df['Month'] = df['InvoiceDate'].dt.to_period("M")
        monthly_avg = df.groupby('Month').agg(
            Avg_UnitPrice=('UnitPrice', 'mean'),
            Avg_Quantity=('Quantity', 'mean')
        ).reset_index()
        monthly_avg['Month'] = monthly_avg['Month'].astype(str)
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_avg['Month'], monthly_avg['Avg_UnitPrice'],
                 marker='o', linestyle='-', label='Avg UnitPrice')
        plt.plot(monthly_avg['Month'], monthly_avg['Avg_Quantity'],
                 marker='s', linestyle='--', label='Avg Quantity')
        plt.title("Monthly Average UnitPrice & Quantity")
        plt.xlabel("Month")
        plt.ylabel("Average Value")
        plt.xticks(rotation=45)
        plt.legend()
        time_series_path = os.path.join(
            PLOTS_DIR, "monthly_avg_price_quantity.png")
        plt.savefig(time_series_path, bbox_inches='tight')
        plt.close()
        plot_files["Monthly Avg UnitPrice & Quantity"] = time_series_path

    return plot_files


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Examine the Retail Sales Data (UCI) dataset and generate an extended report with insights and additional plots. Accepts both Excel and CSV formats."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to the dataset file (default: {DEFAULT_INPUT})."
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
    Main function to load the dataset (supports both xlsx and csv), standardize the columns,
    generate an extended report with additional plots, and write the report to a file.
    """
    args = parse_arguments()
    logger.info("Starting extended dataset examination.")
    logger.info(f"Loading dataset from: {args.input}")

    # Load dataset based on file extension.
    df = load_dataset(args.input)

    # Standardize column names to the final list.
    df = standardize_columns(df)

    logger.info("Dataset loaded and standardized successfully.")

    basic_report = generate_basic_report(df)
    extended_report = generate_extended_insights(df)

    logger.info("Generating and saving additional plots...")
    additional_plots = save_plots(df)
    plots_section = "Plots Generated:\n" + "\n".join(
        f" - {desc}: {path}" for desc, path in additional_plots.items())

    full_report = "\n\n".join([basic_report, extended_report, plots_section])

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
