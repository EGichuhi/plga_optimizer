#!/bin/bash
# setup.sh

echo "============================================================"
echo "PLGA Drug Delivery Optimizer - Setup"
echo "============================================================"

# Install RDKit
echo ""
echo "Installing RDKit..."
conda install -c conda-forge rdkit -y

# Install everything else from requirements.txt
echo ""
echo "Installing Python packages..."
pip install -r requirements.txt

# Train models
echo ""
echo "Training models..."
python main.py

echo ""
echo " Setup complete!"
echo ""
echo "Run: python cl_optimizer.py"