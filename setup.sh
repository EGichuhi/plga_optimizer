#!/bin/bash
# setup.sh - Runs automatically in Codespaces

echo "============================================================"
echo " PLGA Drug Delivery Optimizer - Setup"
echo "============================================================"

# Install RDKit via conda (works in Codespaces)
echo ""
echo "Installing RDKit via conda..."
conda install -c conda-forge rdkit -y

# Install other dependencies via pip (requirements.txt should NOT have rdkit-pypi)
echo ""
echo "Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Train models
echo ""
echo "Training models..."
python main.py

echo ""
echo "Setup complete!"
echo ""
echo "Run: python cl_optimizer.py"