###############################################################################
# Stage 1: Builder
#
# This stage installs any necessary build dependencies, compiles wheels for
# Python packages from requirements.txt, and prepares a cache of dependency
# artifacts. This approach reduces the final image size by only copying the built
# packages to the final image.
###############################################################################
FROM python:3.11-slim AS builder

# Install system-level build dependencies. The "--no-install-recommends" flag
# helps reduce the final size of the image by installing only essential packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for dependency building.
WORKDIR /install

# Copy requirements and setup files first to leverage Docker's cache.
# You can also copy other relevant build files if needed.
COPY requirements.txt setup.py ./

# Upgrade pip and build wheels for all dependencies.
# Wheels are pre-built packages that can be installed faster in the runtime image.
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/install/wheels -r requirements.txt

###############################################################################
# Stage 2: Production Runtime
#
# This stage creates the final runtime container. It copies only the built wheels 
# (from the builder stage) and the application code so that build tools and other
# extraneous files are not included, resulting in a smaller and more secure image.
###############################################################################
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and to
# ensure stdout and stderr are flushed immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a new non-root user for running the application.
# Running as a non-root user in production containers improves security.
RUN useradd --create-home appuser

# Set the working directory inside the container.
WORKDIR /app

# Copy built dependency wheels from the builder stage.
COPY --from=builder /install/wheels /wheels

# Upgrade pip and install the dependencies using the pre-built wheels.
# The "--no-index" flag forces pip to look only locally for packages, which speeds up installation.
RUN pip install --upgrade pip && \
    pip install --no-index --find-links=/wheels -r requirements.txt

# Copy the rest of the application code into the container.
COPY . .

# Adjust file permissions to allow non-root user access.
RUN chown -R appuser:appuser /app

# Switch to the non-root user.
USER appuser

# Expose the port your application will run on. Adjust the port as required.
EXPOSE 8080

# Set the default command for the container.
# In this example, we run the package as a module using the __main__.py in the src/ folder.
# Alternatively, you could point directly to a command-line script declared in your setup.py.
CMD ["python", "-m", "src"]
