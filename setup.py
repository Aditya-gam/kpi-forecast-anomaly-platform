# setup.py

import os
from setuptools import find_packages, setup

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # Package name (lowercase, hyphenated)
    name="kpi-forecast-anomaly-platform",
    version="0.1.0",  # Semantic versioning: MAJOR.MINOR.PATCH
    author="Aditya Gambhir",
    author_email="gambhir.aditya19@gmail.com",
    description="End-to-end platform for forecasting financial KPIs and detecting anomalies using ML and AWS.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aditya-gam/kpi-forecast-anomaly-platform",
    project_urls={
        "Documentation": "https://github.com/Aditya-gam/kpi-forecast-anomaly-platform#readme",
        "Source": "https://github.com/Aditya-gam/kpi-forecast-anomaly-platform",
        "Tracker": "https://github.com/Aditya-gam/kpi-forecast-anomaly-platform/issues",
    },
    license="MIT License",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT Software License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Use src/ layout
    include_package_data=True,  # Include files from MANIFEST.in
    install_requires=[
        # Pin versions for reproducibility
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "matplotlib==3.10.1",
        "statsmodels==0.14.4",
        "prophet==1.1.6",
        "tensorflow==2.19.0",
        "boto3==1.37.31",
        "sagemaker==2.243.0",
    ],
    extras_require={
        "dev": [
            "setuptools",
            "wheel",
            "build",
            "pip-tools",
            "pytest",
            "flake8",
            "black",
            "isort",
            "pre-commit",
        ],
    },
    entry_points={
        "console_scripts": [
            "download-data=scripts.download_dataset:main",
            "preprocess-data=scripts.preprocess_data:main",
            "run-forecasting=scripts.run_forecasting:main",
            "detect-anomalies=scripts.detect_anomalies:main",
            "upload-to-s3=scripts.upload_to_s3:main",
        ],
    },
    zip_safe=False,
)
