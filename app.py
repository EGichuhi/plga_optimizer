import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append('.')

# Import directly from the module
from plga_optimizer import PLGAOptimizer

# Page configuration
st.set_page_config(
    page_title="PLGA Drug Delivery Optimizer",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling with prominent button
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    h1 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    
    h2, h3 {
        color: #2c3e50;
    }
    
    /* Prominent Run Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #145a8d;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    /* Primary button specific styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f77b4 0%, #145a8d 100%);
    }
    
    /* Placeholder styling */
    .placeholder-container {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 2rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Success message styling */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Results container */
    .results-container {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

st.title("💊 PLGA Drug Delivery Optimizer")
st.markdown("**Machine Learning-Optimized Polymer-Based Drug Delivery Formulations**")
st.markdown("---")

# Initialize session state for results
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'optimization_run' not in st.session_state:
    st.session_state.optimization_run = False
if 'current_drug' not in st.session_state:
    st.session_state.current_drug = None
if 'optimization_triggered' not in st.session_state:
    st.session_state.optimization_triggered = False

# Initialize optimizer with caching
@st.cache_resource
def load_optimizer():
    """Load the optimizer with caching to avoid reloading"""
    try:
        optimizer = PLGAOptimizer()
        return optimizer
    except Exception as e:
        st.error(f"Failed to load optimizer: {str(e)}")
        return None

# Load optimizer
with st.spinner("Loading models and drug database..."):
    optimizer = load_optimizer()

if optimizer is None:
    st.stop()

# Get all available drugs
available_drugs = optimizer.list_available_drugs()

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Optimization Settings")
    st.markdown("---")
    
    # Drug selection
    st.markdown("### 🔍 Select Drug")
    
    # Search/filter
    search_term = st.text_input("Search drug", placeholder="Type drug name...", key="drug_search")
    
    if search_term:
        filtered_drugs = [drug for drug in available_drugs if search_term.lower() in drug.lower()]
        st.caption(f"Found {len(filtered_drugs)} drugs")
    else:
        filtered_drugs = available_drugs
    
    if filtered_drugs:
        drug_name = st.selectbox(
            "Select drug",
            options=filtered_drugs,
            help=f"Choose from {len(available_drugs)} available drugs"
        )
    else:
        st.warning(f"No drugs found matching '{search_term}'")
        drug_name = st.selectbox("Select drug", options=available_drugs)
    
    st.markdown("---")
    
    # Optimization parameters
    st.markdown("### ⚙️ Parameters")
    
    min_ee = st.slider(
        "Minimum EE (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=5.0,
        help="Minimum acceptable encapsulation efficiency"
    )
    
    # Fixed max size to 500nm
    max_size = st.slider(
        "Maximum particle size (nm)",
        min_value=50.0,
        max_value=500.0,
        value=300.0,
        step=25.0,
        help="Maximum acceptable particle size (max 500nm)"
    )
    
    priority_options = {
        "balanced": "Balanced (40% Size + 30% EE + 30% LC)",
        "size": "Minimize Particle Size",
        "ee": "Maximize Encapsulation Efficiency",
        "lc": "Maximize Loading Capacity"
    }
    
    priority = st.selectbox(
        "Optimization priority",
        options=list(priority_options.keys()),
        format_func=lambda x: priority_options[x]
    )
    
    show_top = st.number_input(
        "Number of recommendations",
        min_value=1,
        max_value=50,
        value=10,
        step=5
    )
    
    auto_save = st.checkbox("Auto-save results to CSV", value=True)
    
    st.markdown("---")
    
    # Database stats
    st.markdown("### 📊 Database")
    st.metric("Available drugs", len(available_drugs))
    st.metric("Total formulations", 433)
    st.metric("Search combinations", 448)

# Main content area
# Prominent Run Optimization button at the top of main area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_optimization = st.button("🚀 RUN OPTIMIZATION", type="primary", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Show current selection summary
if drug_name:
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info(f"📌 **Selected Drug:** {drug_name}")
    with col_info2:
        st.info(f"⚙️ **Min EE:** {min_ee}% | **Max Size:** {max_size} nm")
    with col_info3:
        st.info(f"🎯 **Priority:** {priority.capitalize()}")

st.markdown("---")

# Handle optimization when button is clicked
if run_optimization:
    if not drug_name:
        st.error("❌ Please select a drug from the sidebar")
    else:
        st.session_state.optimization_triggered = True
        with st.spinner(f"🔍 Searching 448 formulation combinations for {drug_name}..."):
            try:
                # Run optimization
                results_df = optimizer.recommend(
                    drug_name=drug_name,
                    min_ee=min_ee,
                    max_size=max_size,
                    priority=priority,
                    show_top=show_top,
                    auto_save=auto_save
                )
                
                # Store results in session state
                st.session_state.results_df = results_df
                st.session_state.optimization_run = True
                st.session_state.current_drug = drug_name
                st.session_state.current_min_ee = min_ee
                st.session_state.current_max_size = max_size
                st.session_state.current_priority = priority
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.session_state.optimization_run = False
                st.session_state.results_df = None
                import traceback
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

# Display results if optimization has been run
if st.session_state.optimization_run and st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    drug_name_display = st.session_state.current_drug
    
    if results_df.empty:
        st.warning("⚠️ No formulations found meeting your constraints")
        st.info("**Suggestions:**\n- Lower the minimum EE requirement\n- Increase the maximum particle size\n- Try a different drug")
        
        # Add button to reset
        col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
        with col_reset2:
            if st.button("🔄 Start New Optimization", use_container_width=True):
                st.session_state.optimization_run = False
                st.session_state.results_df = None
                st.session_state.optimization_triggered = False
                st.rerun()
    else:
        # Success message
        st.success(f"✅ Optimization complete! Found {len(results_df)} formulations meeting criteria")
        
        # Best formulation card
        st.markdown("## 🏆 Top Recommended Formulation")
        
        best = results_df.iloc[0]
        
        # Display best formulation in columns
        col_params, col_preds = st.columns(2)
        
        with col_params:
            st.markdown("**📋 Formulation Parameters**")
            st.markdown(f"• **Polymer MW:** `{best['polymer_MW (kDa)']:.0f} kDa`")
            st.markdown(f"• **LA/GA Ratio:** `{best['LA/GA ratio']:.2f}`")
            st.markdown(f"• **Drug/Polymer:** `{best['drug/polymer']:.3f}`")
            st.markdown(f"• **Optimization Score:** `{best['Score']:.1f}`")
        
        with col_preds:
            st.markdown("**📈 Predicted Performance**")
            st.markdown(f"• **Particle Size:** `{best['Size (nm)']:.1f} nm`")
            st.markdown(f"• **Encapsulation Efficiency:** `{best['EE (%)']:.1f}%`")
            st.markdown(f"• **Loading Capacity:** `{best['LC (%)']:.1f}%`")
        
        # Drug properties (without logP)
        try:
            drug_props = optimizer.get_drug_properties(drug_name_display)
            if drug_props and drug_props.get('mol_MW') and not np.isnan(drug_props.get('mol_MW')):
                with st.expander("📊 Drug Properties", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Molecular Weight", f"{drug_props['mol_MW']:.1f} g/mol")
                        st.metric("TPSA", f"{drug_props['mol_TPSA']:.1f} Å²" if drug_props['mol_TPSA'] else "N/A")
                    with col_b:
                        st.metric("Melting Point", f"{drug_props['mol_melting_point']:.1f} °C" if drug_props['mol_melting_point'] else "N/A")
                        st.metric("H-bond Acceptors", int(drug_props['mol_Hacceptors']) if drug_props['mol_Hacceptors'] else "N/A")
                    with col_c:
                        st.metric("H-bond Donors", int(drug_props['mol_Hdonors']) if drug_props['mol_Hdonors'] else "N/A")
                        st.metric("Heteroatoms", int(drug_props['mol_heteroatoms']) if drug_props['mol_heteroatoms'] else "N/A")
        except Exception as e:
            pass
        
        st.markdown("---")
        
        # All recommendations table
        st.markdown(f"## 📋 Top {min(show_top, len(results_df))} Formulations")
        
        # Format dataframe for display
        display_df = results_df.head(show_top).copy()
        display_df['polymer_MW (kDa)'] = display_df['polymer_MW (kDa)'].astype(int)
        
        # Rename columns for cleaner display
        display_df.columns = [
            'Polymer MW (kDa)', 'LA/GA Ratio', 'Drug/Polymer',
            'Size (nm)', 'EE (%)', 'LC (%)', 'Score'
        ]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Polymer MW (kDa)": st.column_config.NumberColumn("Polymer MW (kDa)", format="%d"),
                "LA/GA Ratio": st.column_config.NumberColumn("LA/GA Ratio", format="%.2f"),
                "Drug/Polymer": st.column_config.NumberColumn("Drug/Polymer", format="%.3f"),
                "Size (nm)": st.column_config.NumberColumn("Size (nm)", format="%.1f"),
                "EE (%)": st.column_config.NumberColumn("EE (%)", format="%.1f"),
                "LC (%)": st.column_config.NumberColumn("LC (%)", format="%.1f"),
                "Score": st.column_config.NumberColumn("Score", format="%.1f"),
            },
            hide_index=True
        )
        
        # Download and reset buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{drug_name_display.replace(' ', '_')}_{st.session_state.current_priority}_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_btn3:
            if st.button("🔄 New Optimization", use_container_width=True):
                st.session_state.optimization_run = False
                st.session_state.results_df = None
                st.session_state.optimization_triggered = False
                st.rerun()
        
        # Debug info in expander (remove in production)
        with st.expander("ℹ️ Optimization Info"):
            st.write(f"Drug: {drug_name_display}")
            st.write(f"Min EE: {st.session_state.current_min_ee}%")
            st.write(f"Max Size: {st.session_state.current_max_size} nm")
            st.write(f"Priority: {st.session_state.current_priority}")
            st.write(f"Results found: {len(results_df)}")

# Show placeholder only if no optimization has been run
elif not st.session_state.optimization_triggered and not st.session_state.optimization_run:
    st.markdown("""
    <div class="placeholder-container">
        <h3 style="color: #1f77b4; margin-bottom: 1rem;">✨ Ready to Optimize</h3>
        <p style="color: #495057; font-size: 1.1rem; margin-bottom: 1rem;">
            Click the <strong style="color: #1f77b4;">RUN OPTIMIZATION</strong> button above to find the best PLGA formulation for your selected drug.
        </p>
        <p style="color: #6c757d; margin-top: 1rem;">
            The optimizer will search through <strong>448 formulation combinations</strong> to find optimal parameters.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show available drugs preview in an expander
    with st.expander("📋 View Available Drugs in Database"):
        # Show drugs in columns
        drugs_per_column = 20
        num_columns = (len(available_drugs) + drugs_per_column - 1) // drugs_per_column
        
        cols = st.columns(min(num_columns, 3))
        for idx, drug in enumerate(available_drugs):
            col_idx = idx % len(cols)
            cols[col_idx].write(f"• {drug}")
        
        st.caption(f"Total: {len(available_drugs)} unique drugs available for optimization")

# Information section at the bottom
st.markdown("---")
with st.expander("ℹ️ About the PLGA Optimizer"):
    st.markdown("""
    ### How it works
    
    The PLGA Drug Delivery Optimizer uses machine learning models trained on experimental data to predict optimal formulation parameters for polymer-based drug delivery systems.
    
    **Models predict:**
    - **Particle Size (nm)** - Critical for cellular uptake and biodistribution
    - **Encapsulation Efficiency (%)** - Amount of drug successfully encapsulated
    - **Loading Capacity (%)** - Drug payload per polymer mass
    
    **Search space:**
    - **Polymer MW:** 10, 20, 30, 50, 75, 100, 150 kDa
    - **LA/GA Ratio:** 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0
    - **Drug/Polymer Ratio:** 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
    
    **Scoring:**
    - **Balanced:** 40% Size + 30% EE + 30% LC
    - **Size-focused:** Prioritizes smaller particles
    - **EE-focused:** Prioritizes higher encapsulation
    - **LC-focused:** Prioritizes higher loading capacity
    
    **Note:** Maximum particle size is capped at 500nm for optimal drug delivery applications.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "PLGA Drug Delivery Optimizer | Powered by Machine Learning</p>",
    unsafe_allow_html=True
)