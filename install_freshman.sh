#!/bin/bash

# Freshman Environment Setup Script (Python 3.11 for GRF Compatibility)
# Uses 'uv pip' for fast, individual package installation.

export VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating virtual environment..."
    uv venv $VENV_DIR --python 3.10
fi

source $VENV_DIR/bin/activate

echo "[*] Installing dependencies individually..."

# Core Libraries
uv pip install "numpy<2"
uv pip install "torch --index-url https://download.pytorch.org/whl/cpu"
uv pip install "pygame>=1.9.6"
uv pip install "matplotlib"
uv pip install "easydict"

# Environments
uv pip install "gymnasium[all]"
uv pip install "highway-env"
uv pip install "DI-engine"
uv pip install "psutil"

# Google Research Football (Compilation might require system libs)
# We use --no-build-isolation to ensure it sees the venv's psutil if needed
uv pip install "gfootball" --no-build-isolation || echo "[!] gfootball failed to install. Ensure system libs are present."

echo "[*] Installation sequence completed."
