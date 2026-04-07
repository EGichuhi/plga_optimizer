import streamlit as st
import subprocess

st.title("PLGA Drug Delivery Optimizer")

if st.button("Run Optimizer"):
    with st.spinner("Running..."):
        result = subprocess.run(
            ["python", "plga_optimizer.py"],
            capture_output=True,
            text=True
        )
        st.code(result.stdout)
        if result.stderr:
            st.error(result.stderr)