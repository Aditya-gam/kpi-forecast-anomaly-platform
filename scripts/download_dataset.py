#!/usr/bin/env python3
"""
download_dataset.py
-------------------
This script downloads the Retail Sales Data (UCI) dataset and saves it in the raw
data folder specified by the DATA_PATH environment variable.

The default download URL is set to a known UCI repository URL for the Online Retail II
dataset. You can override this by supplying the --url argument. In addition, you can
override the output file name and location with --output. By default, if a file already
exists at the target location, the script will skip downloading unless the --force flag
is provided.

Usage:
    python download_dataset.py [--url DATASET_URL] [--output OUTPUT_PATH] [--force]

Environment Variables:
    DATA_PATH: The directory where the raw dataset will be stored.
    
Dependencies:
    - requests: For handling HTTP requests.
    - argparse, os, sys, logging: Standard Python libraries.
    
Author: Aditya Gambhir
"""

import os
import sys
import argparse
import logging
import requests

# Configure logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

DEFAULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
DEFAULT_FILENAME = "online_retail_II.xlsx"


def download_dataset(url: str, output_path: str, force: bool = False) -> None:
    """
    Download the dataset from a specified URL and save it to the output path.

    Parameters:
        url (str): The URL from which to download the dataset.
        output_path (str): The file path where the dataset should be saved.
        force (bool): If True, re-download and overwrite the file even if it exists.

    Raises:
        requests.HTTPError: If the HTTP request for the dataset fails.
        Exception: For other exceptions, such as file I/O errors.
    """
    if os.path.exists(output_path) and not force:
        logger.info(
            f"File already exists at {output_path}. Use --force to re-download.")
        return

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Starting download of dataset from {url}")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()  # Raise exception for HTTP errors
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Optionally, print progress information
                        if total_size > 0:
                            percent = downloaded * 100 / total_size
                            logger.info(f"Download progress: {percent:.2f}%")
        logger.info(f"Dataset successfully downloaded to {output_path}")
    except requests.RequestException as req_err:
        logger.error(f"Request failed: {req_err}")
        raise
    except Exception as err:
        logger.error(f"An error occurred during download: {err}")
        raise


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download the Retail Sales Data (UCI) dataset for KPI forecasting and anomaly detection."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"URL to download the dataset from (default: {DEFAULT_URL})."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.getenv("DATA_PATH", "./data/raw"), DEFAULT_FILENAME),
        help="Output file path for saving the dataset (default: DATA_PATH/online_retail_II.xlsx)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and overwrite the file if it already exists."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to download the dataset.
    """
    args = parse_args()

    logger.info("Download Dataset Script Starting")
    logger.info(f"Using URL: {args.url}")
    logger.info(f"Output Path: {args.output}")
    if args.force:
        logger.info(
            "Force flag is set: existing file (if any) will be overwritten.")

    try:
        download_dataset(args.url, args.output, force=args.force)
    except Exception as error:
        logger.error(f"Failed to download dataset: {error}")
        sys.exit(1)
    logger.info("Download dataset script completed successfully.")


if __name__ == "__main__":
    main()
