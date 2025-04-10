#!/bin/bash

# Create directories under the project root
mkdir -p data/{raw,interim,processed}
mkdir -p notebooks
mkdir -p scripts
mkdir -p src/{config,data,features,models,detection,utils}
mkdir -p tests/{unit,integration}
mkdir -p .github/workflows

# Create placeholder files in notebooks/
touch notebooks/01_data_exploration.ipynb
touch notebooks/02_forecasting_model_dev.ipynb
touch notebooks/03_anomaly_detection_dev.ipynb

# Create placeholder files in scripts/
touch scripts/download_dataset.py
touch scripts/preprocess_data.py
touch scripts/run_forecasting.py
touch scripts/detect_anomalies.py
touch scripts/upload_to_s3.py

# Create placeholder files in src/ directories
touch src/data/loader.py
touch src/features/build_features.py

touch src/models/arima_model.py
touch src/models/prophet_model.py
touch src/models/lstm_model.py
touch src/models/utils.py

touch src/detection/isolation_forest.py
touch src/detection/z_score.py

touch src/utils/io.py
touch src/utils/logger.py
touch src/utils/timer.py

# Create placeholder files for tests (optionally add a .placeholder file)
touch tests/unit/.placeholder
touch tests/integration/.placeholder

# Create files at the project root
touch .env.example
touch .gitignore
touch requirements.txt
touch setup.py
touch Makefile
touch Dockerfile
touch docker-compose.yml
touch README.md
touch LICENSE

# Create the GitHub Actions workflow file
touch .github/workflows/ci.yml

echo "Project structure created successfully."
