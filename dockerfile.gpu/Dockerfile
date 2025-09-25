FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install Python 3.8 and required system tools
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3.8-venv python3.8-distutils \
    wget curl git build-essential ca-certificates \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python and pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Create venv and install pip manually for Python 3.8
RUN python -m venv /workspace/.venv || true && \
    curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && \
    /workspace/.venv/bin/python get-pip.py && \
    /workspace/.venv/bin/pip install --upgrade pip && \
    rm get-pip.py

# Install PyTorch (CUDA 11.8)
RUN . /workspace/.venv/bin/activate && \
    pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN . /workspace/.venv/bin/activate && pip install -r /workspace/requirements.txt

# Auto-activate venv when container starts
RUN echo 'source /workspace/.venv/bin/activate' >> ~/.bashrc

# Reset working directory
WORKDIR /workspace

CMD ["/bin/bash"]