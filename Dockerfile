FROM python:3.12

# Set the workdir
WORKDIR /app

# Minimal software stack
RUN apt-get update && \
    apt-get install -y nano less htop git libhdf5-serial-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy all application code
COPY . .

# Install the package with all dependencies (hepattn, hgcal-offline-reco, dev tools, and flash-attention)
RUN echo "=======================================" && \
    echo "Installing hgcal-offline-reco with dependencies" && \
    echo "Current time:" $(date) && \
    echo "=======================================" && \
    pip install --no-cache-dir -e .[dev,fa2] --verbose && \
    echo "Installation complete!"

