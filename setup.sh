#!/bin/bash 

echo "============================================================"
echo " PLGA Drug Delivery Optimizer - Setup"
echo "============================================================"

# Detect environment 
if [ -n "$CODESPACES" ]; then
  ENV_TYPE="codespaces"
else
  ENV_TYPE="local"
fi
echo ""
#!/usr/bin/env bash
set -e

# Install Miniconda if not already installed
if [ ! -d "/opt/conda" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /opt/conda
    rm miniconda.sh
fi

# Create your environment if missing
if ! /opt/conda/bin/conda env list | grep -q "plga_venv"; then
    echo "Creating conda environment plga_venv..."
    /opt/conda/bin/conda create -y -n plga_venv python=3.9
fi

# Install RDKit
echo "Installing RDKit..."
/opt/conda/bin/conda install -y -n plga_venv -c conda-forge rdkit


echo ""
echo "Environment detected: $ENV_TYPE"

# Local: create and activate a conda env 
if [ "$ENV_TYPE" = "local" ]; then
  ENV_NAME="plga_venv"

  if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda env '$ENV_NAME' already exists, skipping creation."
  else
    echo "Creating conda env '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.11 -y
  fi

  # Activate inside the script
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"

  echo "Active env: $CONDA_DEFAULT_ENV"
fi

# ── Install RDKit via conda 
echo ""
echo "Installing RDKit via conda..."
conda install -c conda-forge rdkit -y

# ── Install remaining dependencies via pip 
echo ""
echo "Installing Python packages..."
pip install --quiet --upgrade pip
pip install -r requirements.txt

# ── Train models 
echo ""
echo "Training models..."
python main.py

echo ""
echo "===================================================================="
echo " Setup complete!"

if [ "$ENV_TYPE" = "local" ]; then
  echo ""
  echo " To run the optimizer:"
  echo "   python plga_optimizer.py"
fi
echo "===================================================================="
