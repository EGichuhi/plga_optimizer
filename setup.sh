#!/bin/bash
set -e

echo "Setting up PLGA Optimizer environment..."

# Create venv (using a consistent name)
python -m venv plga_venv

# Activate and install
source plga_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Make run.sh executable
chmod +x run.sh

# Create an activation script that auto-runs
echo '#!/bin/bash' > activate.sh
echo 'source $(pwd)/plga_venv/bin/activate' >> activate.sh
chmod +x activate.sh

echo "Setup complete!"
echo ""
echo "To activate environment, run: source activate.sh"
echo "Or just run: ./run.sh"