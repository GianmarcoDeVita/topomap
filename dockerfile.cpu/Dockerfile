FROM python:3.8-slim

# Install system packages (adjust if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy your requirements.txt into the container
COPY requirements.txt .

# Create a virtual environment inside /workspace/.venv
RUN python -m venv /workspace/.venv

# Upgrade pip inside the venv and install packages
RUN /workspace/.venv/bin/pip install --upgrade pip
RUN /workspace/.venv/bin/pip install -r requirements.txt

# Optionally, set environment variables so the venv is used by default
ENV PATH="/workspace/.venv/bin:$PATH"

# Copy your project files
COPY . .

# Auto-activate venv when container starts
RUN echo 'source /workspace/.venv/bin/activate' >> ~/.bashrc

CMD ["/bin/bash"]