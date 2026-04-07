#!/bin/bash
source plga_venv/bin/activate
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0