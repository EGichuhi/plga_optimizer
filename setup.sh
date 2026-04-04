#!/bin/bash
# setup.sh - Runs automatically in Codespaces

echo "============================================================"
echo " PLGA Drug Delivery Optimizer - Setup"
echo "============================================================"

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in Codespaces
if [ -n "$CODESPACES" ]; then
    echo -e "${BLUE} Running in GitHub Codespaces${NC}"
fi

# Install RDKit via conda (works in Codespaces)
echo ""
echo -e "${GREEN} Installing RDKit via conda...${NC}"
conda install -c conda-forge rdkit -y

# Install other dependencies via pip
echo ""
echo -e "${GREEN} Installing other Python dependencies...${NC}"
pip install --quiet --upgrade pip
pip install --quiet pandas numpy scikit-learn matplotlib seaborn joblib streamlit

# Check if models already exist
if [ -f "models/particle_size_model.pkl" ]; then
    echo -e "${GREEN}✓ Models already trained${NC}"
else
    echo ""
    echo -e "${GREEN} Training models (first time only)...${NC}"
    python main.py
fi

echo ""
echo -e "${GREEN} Setup complete!${NC}"
echo ""
echo "To start the optimizer, run:"
echo -e "${BLUE}  python cl_optimizer.py${NC}"
echo ""
echo "Or for the web interface:"
echo -e "${BLUE}  streamlit run streamlit_app.py${NC}"