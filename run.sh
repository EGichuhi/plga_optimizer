#!/bin/bash
# run.sh - Launch the optimizer

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN} PLGA Drug Delivery Optimizer${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if setup needs to run
if [ ! -d "plga_env" ] && [ ! -f "models/particle_size_model.pkl" ]; then
    echo "First time setup. Running setup.sh..."
    bash setup.sh
fi

# Run the optimizer
echo ""
python cl_optimizer.py