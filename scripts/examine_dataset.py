#!/usr/bin/env python3
"""
examine_dataset.py
-------------------
This script examines the Retail Sales Data (UCI) dataset stored in the raw data directory
(e.g., /data/raw/online_retail_II.xlsx) and generates a comprehensive report with extended insights and additional plots.
The report includes:

1. Basic Dataset Overview:
    - Dataset shape, column information (data types, missing & unique values).
    - Statistical summaries for numeric columns.
    - Domain-specific analysis (InvoiceDate conversion & range, invoice cancellation).
    - A preview of sample records.

2. Extended Insights:
    - Revenue calculation (Price x Quantity).
    - Monthly revenue trends summary.
    - Top 10 products by quantity sold and revenue.
    - Country-level analysis (transaction counts and revenue).
    - Correlation analysis among numeric features.

3. Additional Visualizations:
    - Distribution histograms for Quantity and Price.
    - Enhanced Monthly Revenue Trend plot with annotations.
    - Correlation heatmap.
    - Scatter plot for Price vs. Quantity with a regression line.
    - Boxplots for Price and Quantity.
    - Pairplot for all numeric features.
    - Time series plots of average Price and average Quantity over months.
    
All plots are saved in the './data/reports/plots' directory, and their file paths are included in the final report.

Usage:
    python scripts/examine_dataset.py [--input INPUT_FILE] [--output REPORT_FILE]

Arguments:
    --input     Path to the dataset Excel file (default: "./data/raw/online_retail_II.xlsx").
    --output    Path where the generated report will be saved (default: "./data/reports/dataset_report.txt").

Environment Variables:
    DATA_PATH   Base directory for raw data (used as a default if not provided via arguments).

Dependencies:
    - pandas, matplotlib, seaborn, numpy, openpyxl
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

# Define default file paths.
DEFAULT_INPUT = os.path.join(
    os.getenv("DATA_PATH", "./data/raw"), "online_retail_II.xlsx")
DEFAULT_OUTPUT = os.path.join(
    os.getenv("DATA_PATH", "./data/reports"), "dataset_report.txt")
PLOTS_DIR = os.path.join("data", "reports", "plots1")


def generate_basic_report(df: pd.DataFrame) -> str:
    """
    Generate a basic overview report of the dataset.

    Parameters:
        df (pd.DataFrame): The loaded dataset.

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
            f" - {col:15s}: dtype={dtype}, missing={missing:5d}, unique={unique:5d}")
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
        df (pd.DataFrame): The loaded dataset.

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
            " - Revenue (Price x Quantity) computed for each transaction.\n")
    else:
        lines.append(
            " - Could not compute Revenue (Price or Quantity missing).\n")

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
    Create and save a suite of visualizations.

    Plots created:
      1. Histogram for Quantity distribution.
      2. Histogram for Price distribution.
      3. Enhanced Monthly Revenue Trend with annotations.
      4. Correlation Heatmap for numeric features.
      5. Scatter Plot (Price vs. Quantity) with regression line.
      6. Boxplots for Price and Quantity.
      7. Pairplot for all numeric features.
      8. Time series of average Price and average Quantity per month.

    Parameters:
        df (pd.DataFrame): The loaded dataset.

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

    # Plot 2: Price Distribution Histogram.
    plt.figure(figsize=(8, 6))
    plt.hist(df['UnitPrice'], bins=50, edgecolor='black', color='salmon')
    plt.title("Distribution of Price")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    price_path = os.path.join(PLOTS_DIR, "price_distribution.png")
    plt.savefig(price_path, bbox_inches='tight')
    plt.close()
    plot_files["Price Distribution"] = price_path

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
        # Annotate the peak month for additional insight.
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

    # Plot 5: Scatter Plot (Price vs. Quantity) with regression line.
    plt.figure(figsize=(8, 6))
    sns.regplot(x='UnitPrice', y='Quantity', data=df, scatter_kws={
                'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title("Scatter Plot: Price vs. Quantity")
    scatter_path = os.path.join(PLOTS_DIR, "price_vs_quantity_scatter.png")
    plt.savefig(scatter_path, bbox_inches='tight')
    plt.close()
    plot_files["Price vs. Quantity Scatter"] = scatter_path

    # Plot 6: Boxplots for Price and Quantity.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['UnitPrice'], color='lightgreen')
    plt.title("Boxplot: Price")
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Quantity'], color='lightblue')
    plt.title("Boxplot: Quantity")
    boxplot_path = os.path.join(PLOTS_DIR, "price_quantity_boxplots.png")
    plt.savefig(boxplot_path, bbox_inches='tight')
    plt.close()
    plot_files["Price & Quantity Boxplots"] = boxplot_path

    # Plot 7: Pairplot for Numeric Features.
    pairplot_path = os.path.join(PLOTS_DIR, "numeric_pairplot.png")
    sns.pairplot(df[numeric_cols], diag_kind="kde")
    plt.savefig(pairplot_path, bbox_inches='tight')
    plt.close()
    plot_files["Numeric Features Pairplot"] = pairplot_path

    # Plot 8: Time Series of Average Price and Average Quantity per Month.
    if 'InvoiceDate' in df.columns:
        df['Month'] = df['InvoiceDate'].dt.to_period("M")
        monthly_avg = df.groupby('Month').agg(
            Avg_Price=('UnitPrice', 'mean'),
            Avg_Quantity=('Quantity', 'mean')
        ).reset_index()
        monthly_avg['Month'] = monthly_avg['Month'].astype(str)
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_avg['Month'], monthly_avg['Avg_Price'],
                 marker='o', linestyle='-', label='Avg Price')
        plt.plot(monthly_avg['Month'], monthly_avg['Avg_Quantity'],
                 marker='s', linestyle='--', label='Avg Quantity')
        plt.title("Monthly Average Price & Quantity")
        plt.xlabel("Month")
        plt.ylabel("Average Value")
        plt.xticks(rotation=45)
        plt.legend()
        time_series_path = os.path.join(
            PLOTS_DIR, "monthly_avg_price_quantity.png")
        plt.savefig(time_series_path, bbox_inches='tight')
        plt.close()
        plot_files["Monthly Avg Price & Quantity"] = time_series_path

    return plot_files


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Examine the Retail Sales Data (UCI) dataset and generate an extended detailed report with additional insights and improved plots."
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
    Main function to load the dataset, generate an extended report with additional plots,
    and write the report to a file.
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

    basic_report = generate_basic_report(df)
    extended_report = generate_extended_insights(df)

    logger.info("Generating and saving additional plots...")
    additional_plots = save_plots(df)
    plots_section = "Plots Generated:\n" + \
        "\n".join(f" - {desc}: {path}" for desc,
                  path in additional_plots.items())

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
