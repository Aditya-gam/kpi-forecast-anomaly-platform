###############################################################################
# Production-Level Makefile
#
# This Makefile automates a number of common development, testing, and release
# tasks for the kpi-forecast-anomaly-platform project. The targets include:
#
#   - help         : Display this help message with a list of available targets.
#   - venv         : Create the virtual environment (if it does not already exist).
#   - install      : Upgrade pip and install all project dependencies including dev.
#   - lint         : Run code linters (flake8, black, isort) to enforce code style.
#   - test         : Run the test suite using pytest.
#   - format       : Auto-format source code using black and isort.
#   - build        : Build source and wheel distribution packages.
#   - publish      : Publish distributions to PyPI via twine (requires proper config).
#   - clean        : Remove build artifacts and cache directories.
#   - docker-build : Build the Docker image for the application.
#   - docker-run   : Run the Docker container locally.
#   - precommit    : Run pre-commit hooks over all files.
#   - docs         : Build HTML documentation using Sphinx (if configured).
#
# Before running any target, ensure that your virtual environment is ready or use
# the "install" target which depends on creating the environment first.
###############################################################################

# Variables
PYTHON_VERSION     = python3
VENV_DIR           = .venv
VENV_BIN           = $(VENV_DIR)/bin
PIP                = $(VENV_BIN)/pip
PYTHON             = $(VENV_BIN)/python
FLAKE8             = $(VENV_BIN)/flake8
BLACK              = $(VENV_BIN)/black
ISORT              = $(VENV_BIN)/isort
PYTEST             = $(VENV_BIN)/pytest
TWINE              = $(VENV_BIN)/twine
DOCKER_IMAGE       = kpi-forecast-anomaly-platform

# List of targets that are not files
.PHONY: help venv install lint test format build publish clean docker-build docker-run precommit docs

###############################################################################
# help: Display help information for available Makefile targets
###############################################################################
help:
	@echo "Available targets:"
	@echo "  help         - Display this help information."
	@echo "  venv         - Create the virtual environment (if not present)."
	@echo "  install      - Install project dependencies (including dev extras)."
	@echo "  lint         - Run linters (flake8, black, isort) to check code style."
	@echo "  test         - Run the test suite using pytest."
	@echo "  format       - Auto-format source code using black and isort."
	@echo "  build        - Build source and wheel distributions."
	@echo "  publish      - Publish the package to PyPI (requires twine and config)."
	@echo "  clean        - Remove build artifacts and temporary files."
	@echo "  docker-build - Build the Docker image for the application."
	@echo "  docker-run   - Run the Docker container locally."
	@echo "  precommit    - Run pre-commit hooks over all files."
	@echo "  docs         - Build HTML documentation with Sphinx (if configured)."

###############################################################################
# venv: Create a Python virtual environment if it does not already exist.
###############################################################################
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		$(PYTHON_VERSION) -m venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists."; \
	fi

###############################################################################
# install: Create the virtual environment and install all dependencies.
# This target upgrades pip and then installs your package with dev extras.
###############################################################################
install: venv
	@echo "Upgrading pip and installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -e .[dev]

###############################################################################
# lint: Run code linters to enforce code style and catch potential issues.
# This target uses flake8 for static analysis, black and isort in check mode.
###############################################################################
lint: venv
	@echo "Running linters..."
	@$(FLAKE8) src tests
	@$(BLACK) --check src tests
	@$(ISORT) --check-only src tests

###############################################################################
# test: Execute the test suite using pytest.
###############################################################################
test: venv
	@echo "Running tests..."
	@$(PYTEST) --maxfail=1 --disable-warnings -q

###############################################################################
# format: Automatically format code using black and isort.
# This target rewrites your source code to follow style guidelines.
###############################################################################
format: venv
	@echo "Formatting code with black and isort..."
	@$(BLACK) src tests
	@$(ISORT) src tests

###############################################################################
# build: Clean previous builds and create source and wheel distributions.
###############################################################################
build: venv
	@echo "Cleaning previous builds..."
	@$(MAKE) clean
	@echo "Building source and wheel distributions..."
	@$(PYTHON) setup.py sdist bdist_wheel

###############################################################################
# publish: Upload distribution packages to PyPI using twine.
# Note: Ensure that your .pypirc is properly configured and you have valid credentials.
###############################################################################
publish: build venv
	@echo "Publishing packages to PyPI..."
	@$(TWINE) upload dist/*

###############################################################################
# clean: Remove build artifacts, caches, and temporary files.
###############################################################################
clean:
	@echo "Cleaning build artifacts, cache directories, and temporary files..."
	@rm -rf build dist *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +

###############################################################################
# docker-build: Build the Docker image for the application.
###############################################################################
docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE)..."
	@docker build -t $(DOCKER_IMAGE) .

###############################################################################
# docker-run: Run the Docker container locally.
# Maps port 8080 from the container to localhost.
###############################################################################
docker-run:
	@echo "Running Docker container from image: $(DOCKER_IMAGE)..."
	@docker run --rm -it -p 8080:8080 $(DOCKER_IMAGE)

###############################################################################
# precommit: Execute pre-commit hooks to validate code before commit.
###############################################################################
precommit: venv
	@echo "Running pre-commit hooks..."
	@$(VENV_BIN)/pre-commit run --all-files

###############################################################################
# docs: Build project documentation using Sphinx.
# If you have a docs/ directory with Sphinx configuration, this target builds HTML docs.
###############################################################################
docs: venv
	@echo "Building documentation..."
	@if [ -d "docs" ]; then \
		$(VENV_BIN)/sphinx-build -b html docs docs/_build/html; \
	else \
		echo "Docs directory not found. Skipping docs build."; \
	fi
