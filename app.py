import streamlit as st
import subprocess
import sys
import os

st.title("PLGA Drug Delivery Optimizer")

# Check if the optimizer file exists
optimizer_file = "plga_optimizer.py"
if not os.path.exists(optimizer_file):
    st.error(f"❌ Error: '{optimizer_file}' not found in {os.getcwd()}")
    st.info(f"Please make sure '{optimizer_file}' is in the same directory as this Streamlit app")
    st.stop()

st.info(f"✅ Found '{optimizer_file}' in current directory")

if st.button("Run with Live Output"):
    try:
        # Create placeholders for output
        output_placeholder = st.empty()
        error_placeholder = st.empty()
        
        # Start the process
        process = subprocess.Popen(
            [sys.executable, optimizer_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            cwd=os.getcwd()
        )
        
        # Collect output
        output_text = ""
        error_text = ""
        
        # Show running indicator
        status_placeholder = st.info("🔄 Running optimizer...")
        
        # Read stdout line by line
        for line in process.stdout:
            output_text += line
            output_placeholder.code(output_text, language="python")
        
        # Wait for process to complete
        return_code = process.wait(timeout=60)
        
        # Read any remaining stderr
        error_text = process.stderr.read()
        
        if return_code == 0:
            status_placeholder.success("✅ Optimizer completed successfully!")
        else:
            status_placeholder.error(f"❌ Optimizer exited with code {return_code}")
        
        if error_text:
            error_placeholder.error(f"Errors/Warnings:\n{error_text}")
            
    except subprocess.TimeoutExpired:
        process.kill()
        st.error("❌ Optimizer timed out after 60 seconds")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Add debugging information
st.divider()
with st.expander("🔧 Debugging Information"):
    st.write(f"**Current Directory:** {os.getcwd()}")
    st.write(f"**Python Executable:** {sys.executable}")
    st.write(f"**Python Version:** {sys.version}")
    
    # Check if file is readable
    if os.path.exists(optimizer_file):
        st.write(f"**File Size:** {os.path.getsize(optimizer_file)} bytes")
        st.write(f"**File Permissions:** {oct(os.stat(optimizer_file).st_mode)[-3:]}")
        
        # Show first few lines of the optimizer file
        try:
            with open(optimizer_file, 'r') as f:
                first_lines = f.readlines()[:10]
                st.write("**First 10 lines of optimizer:**")
                st.code(''.join(first_lines), language="python")
        except:
            st.write("Could not read file contents")
    else:
        st.write(f"**File '{optimizer_file}' not found**")
    
    st.write("**Environment Variables (PYTHONPATH):**")
    st.code(os.environ.get('PYTHONPATH', 'Not set'))