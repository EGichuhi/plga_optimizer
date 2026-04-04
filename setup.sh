#!/bin/bash
# setup.sh - Runs automatically in Codespaces

echo "============================================================"
echo "🔬 PLGA Drug Delivery Optimizer - Codespaces Setup"
echo "============================================================"

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in Codespaces
if [ -n "$CODESPACES" ]; then
    echo -e "${BLUE}📦 Running in GitHub Codespaces${NC}"
fi

# Install dependencies
echo ""
echo -e "${GREEN}📦 Installing Python dependencies...${NC}"
pip install --quiet --upgrade pip
pip install --quiet pandas numpy scikit-learn matplotlib seaborn rdkit-pypi joblib streamlit

# Check if models already exist
if [ -f "models/particle_size_model.pkl" ]; then
    echo -e "${GREEN}✓ Models already trained${NC}"
else
    echo ""
    echo -e "${GREEN}🚀 Training models (first time only)...${NC}"
    python main.py
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "To start the optimizer, run:"
echo -e "${BLUE}  ./run.sh${NC}"
echo ""
echo "Or for the web interface:"
echo -e "${BLUE}  streamlit run streamlit_app.py${NC}"