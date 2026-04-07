#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up PLGA Optimizer environment..."

# Clean recreate venv
rm -rf plga_venv
python -m venv plga_venv

# Activate and install
source plga_venv/bin/activate
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# Make scripts executable
chmod +x run.sh

# Run main.py once during setup
echo "Running initial setup script..."
python main.py

echo "Setup complete! Run ./run.sh to start the optimizer."