#!/bin/bash
set -e

echo "Setting up PLGA Optimizer environment..."

# Create venv
python -m venv .venv

# Activate it and install deps
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Make run.sh executable
chmod +x run.sh

# Auto-activate venv in every new terminal
echo 'source /workspaces/plga_optimizer/.venv/bin/activate' >> ~/.bashrc

echo "Setup complete!"