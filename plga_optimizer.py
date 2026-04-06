#!/usr/bin/env python3
"""
PLGA Drug Delivery Optimizer - Single File CLI (Front-end only)

This script loads the models and preprocessing pipeline 
trained by your train.py / model_pipeline.py / main.py.

It does NOT train or rebuild anything — it only performs optimization 
using the saved artifacts from training + test process.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import joblib

# Add current directory to path (so src.feature_engineering can be found)
sys.path.append('.')

from src.feature_engineering import (
    ALL_SCALAR_FEATURES,
    filter_valid_fingerprints,
    generate_morgan_fingerprints,
    create_engineered_features,
    build_full_feature_matrix,
)


class PLGAOptimizer:

    def __init__(self, models_path='models/', data_path='data/processed/'):
        self.models_path = models_path
        self.data_path = data_path
        self.results_path = 'results/'
        os.makedirs(self.results_path, exist_ok=True)

        print("→ Loading models and pipeline  \n")

        # Load the three models saved during training
        self.models = {}
        for target in ['particle_size', 'EE', 'LC']:
            model_path = os.path.join(models_path, f'{target}_model.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"\nModel not found: {model_path}\n\n"
                    f"Please run your training script first: python main.py\n"
                )
            self.models[target] = joblib.load(model_path)
            print(f"✓ Loaded {target.replace('_', ' ')} model")

        # Load the preprocessing pipeline (fitted ONLY on training data during main.py)
        pipe_path = os.path.join(models_path, 'preprocessing_pipeline.pkl')
        if not os.path.exists(pipe_path):
            raise FileNotFoundError(
                f"\nPreprocessing pipeline not found: {pipe_path}\n\n"
                f"Your training script must save the fitted pipeline after training."
            )
        self.preprocess_pipe = joblib.load(pipe_path)
        print("✓ Loaded preprocessing pipeline (fitted on train set only)")

        # Load reference data (used for drug properties and fingerprints)
        df_path = os.path.join(data_path, 'df13_with_features.csv')
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Reference data not found: {df_path}")

        self.df13 = pd.read_csv(df_path)

        # Load cached fingerprints or regenerate
        fp_path = os.path.join(data_path, 'morgan_fingerprints.npy')
        if os.path.exists(fp_path):
            fp_matrix = np.load(fp_path)
            self.df13['morgan_fingerprint'] = list(fp_matrix)
            print("✓ Loaded Morgan fingerprints cache")
        else:
            print("Fingerprint cache missing — regenerating from SMILES...")
            self.df13 = generate_morgan_fingerprints(self.df13, smiles_col='canonical_SMILES')
            self.df13 = filter_valid_fingerprints(self.df13)

        print(f"\n✓ Ready! Database contains {len(self.df13)} formulations "
              f"({self.df13['small_molecule_name'].nunique()} drugs)\n")

    def get_drug_properties(self, drug_name):
        """Pull drug properties from the reference data (same as used in training)."""
        drug_data = self.df13[self.df13['small_molecule_name'].str.lower() == drug_name.lower()]

        drug_cols = ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point',
                     'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'canonical_SMILES']

        if len(drug_data) > 0:
            actual_name = drug_data['small_molecule_name'].iloc[0]
            props = {col: drug_data[col].mean() for col in drug_cols[:-1] if col in drug_data.columns}
            props['canonical_SMILES'] = drug_data['canonical_SMILES'].iloc[0]
            print(f"✓ Found '{actual_name}' in training/reference data")
            return props
        else:
            print(f" Drug '{drug_name}' not in database — using dataset averages from training data.")
            props = {col: self.df13[col].mean() for col in drug_cols[:-1] if col in self.df13.columns}
            props['canonical_SMILES'] = None
            return props

    def _build_candidate_df(self, polymer_MW_kDa, LA_GA, drug_polymer, drug_props):
        """Convert polymer MW from kDa to Da for model prediction"""
        polymer_MW_Da = polymer_MW_kDa * 1000  # Convert kDa to Da
        
        row = {
            'polymer_MW': polymer_MW_Da,  # Store in Da for the model
            'LA/GA': LA_GA,
            'mol_MW': drug_props.get('mol_MW'),
            'mol_logP': drug_props.get('mol_logP'),
            'mol_TPSA': drug_props.get('mol_TPSA'),
            'mol_melting_point': drug_props.get('mol_melting_point'),
            'mol_Hacceptors': drug_props.get('mol_Hacceptors'),
            'mol_Hdonors': drug_props.get('mol_Hdonors'),
            'mol_heteroatoms': drug_props.get('mol_heteroatoms'),
            'drug/polymer': drug_polymer,
            'surfactant_concentration': 1.5,
            'surfactant_HLB': 15.0,
            'aqueous/organic': 1.0,
            'pH': 7.0,
            'solvent_polarity_index': 0.8,
            'canonical_SMILES': drug_props.get('canonical_SMILES'),
        }

        df_candidate = pd.DataFrame([row])
        df_candidate = generate_morgan_fingerprints(df_candidate, smiles_col='canonical_SMILES')
        df_candidate = filter_valid_fingerprints(df_candidate)
        if len(df_candidate) == 0:
            return None

        df_candidate = create_engineered_features(df_candidate)
        return df_candidate

    def _predict(self, df_candidate):
        available_scalar = [f for f in ALL_SCALAR_FEATURES if f in df_candidate.columns]
        X_raw, _, _ = build_full_feature_matrix(df_candidate, scalar_features=available_scalar)

        # Use the exact same pipeline that was fitted on the train set
        X_proc = self.preprocess_pipe.transform(X_raw)

        size = self.models['particle_size'].predict(X_proc)[0]
        ee   = self.models['EE'].predict(X_proc)[0]
        lc   = self.models['LC'].predict(X_proc)[0]

        return size, ee, lc

    def save_results(self, results_df, drug_name, priority, min_ee, max_size):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{drug_name.replace(' ', '_')}_{priority}_{timestamp}.csv"
        filepath = os.path.join(self.results_path, filename)

        header = f"""# PLGA Optimization Results
# Drug: {drug_name}
# Priority: {priority}
# Min EE: {min_ee}%
# Max Size: {max_size} nm
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Total formulations evaluated: {len(results_df)}
# Models trained on train set | Evaluated on test set during main.py
#\n"""

        with open(filepath, 'w') as f:
            f.write(header)

        results_df.to_csv(filepath, mode='a', index=False)
        print(f"✓ Results saved → {filepath}")

    def recommend(self, drug_name, min_ee=30, max_size=500, priority='balanced',
                  show_top=10, auto_save=True):
        print(f"\n{'='*65}")
        print(f" OPTIMIZING PLGA FORMULATION FOR → {drug_name.upper()}")
        print(f"{'='*65}")

        drug_props = self.get_drug_properties(drug_name)

        # Search space in kDa (more intuitive for PLGA polymers)
        search_space = {
            'polymer_MW_kDa': [10, 20, 30, 50, 75, 100, 150],  # kDa
            'LA/GA': [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            'drug/polymer': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        }

        total = len(search_space['polymer_MW_kDa']) * len(search_space['LA/GA']) * len(search_space['drug/polymer'])
        print(f"Searching {total:,} candidate formulations using the trained models...")

        results = []
        for mw_kDa, la_ga, dp in product(search_space['polymer_MW_kDa'], search_space['LA/GA'], search_space['drug/polymer']):
            df_candidate = self._build_candidate_df(mw_kDa, la_ga, dp, drug_props)
            if df_candidate is None:
                continue

            size, ee, lc = self._predict(df_candidate)

            if ee >= min_ee and size <= max_size:
                size_score = max(0.0, min(1.0, (500 - size) / 500))
                ee_score = ee / 100.0
                lc_score = lc / 30.0

                if priority == 'size':
                    score = size_score * 100
                elif priority == 'ee':
                    score = ee_score * 100
                elif priority == 'lc':
                    score = lc_score * 100
                else:  # balanced
                    score = (size_score * 0.4 + ee_score * 0.3 + lc_score * 0.3) * 100

                results.append({
                    'polymer_MW (kDa)': mw_kDa,  # Display in kDa
                    'LA/GA ratio': la_ga,
                    'drug/polymer': round(dp, 3),
                    'Size (nm)': round(size, 1),
                    'EE (%)': round(ee, 1),
                    'LC (%)': round(lc, 1),
                    'Score': round(score, 2),
                })

        if not results:
            print("\n No formulations met your constraints.")
            print("   Try lowering min_EE or raising max_size.")
            return pd.DataFrame()

        results_df = pd.DataFrame(results).sort_values('Score', ascending=False)

        print(f"\n TOP {min(show_top, len(results_df))} RECOMMENDATIONS")
        print("=" * 65)
        print(results_df.head(show_top).to_string(index=False))

        best = results_df.iloc[0]
        print(f"\n{'='*65}")
        print(" BEST RECOMMENDED FORMULATION")
        print(f"{'='*65}")
        print(f" Polymer MW      : {best['polymer_MW (kDa)']} kDa ({best['polymer_MW (kDa)']*1000:,} Da)")
        print(f" LA/GA ratio     : {best['LA/GA ratio']}")
        print(f" Drug/Polymer    : {best['drug/polymer']}")
        print(f"\n Predicted performance (using models from train/test):")
        print(f" • Particle size     : {best['Size (nm)']} nm")
        print(f" • Encapsulation EE  : {best['EE (%)']}%")
        print(f" • Loading capacity  : {best['LC (%)']}%")

        if auto_save:
            self.save_results(results_df, drug_name, priority, min_ee, max_size)

        return results_df

    def list_available_drugs(self):
        return sorted(self.df13['small_molecule_name'].unique())


# ====================== CLI Front-end ======================

def print_banner():
    print("=" * 75)
    print("          PLGA DRUG DELIVERY OPTIMIZER")
    print("=" * 75)
    print("   Uses models trained & validated on train/test split")
    print("   (main.py / train.py / model_pipeline.py)")
    print("=" * 75)


def get_user_preferences():
    print("\nOptimization Priority:")
    print("  1. Balanced (default)")
    print("  2. Minimize particle size")
    print("  3. Maximize EE (encapsulation efficiency)")
    print("  4. Maximize LC (loading capacity)")

    choice = input("\nChoose (1-4) [default: 1]: ").strip()
    return {'1': 'balanced', '2': 'size', '3': 'ee', '4': 'lc'}.get(choice, 'balanced')


def main():
    print_banner()

    try:
        optimizer = PLGAOptimizer()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"\nError initializing optimizer: {e}")
        sys.exit(1)

    while True:
        print("\n" + "=" * 75)
        drug_name = input("Enter drug name (or 'list' to see available, 'quit' to exit): ").strip()

        if drug_name.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!\n")
            break

        if drug_name.lower() == 'list':
            drugs = optimizer.list_available_drugs()
            print("\nAvailable drugs (from training data):")
            for i, drug in enumerate(drugs, 1):
                print(f"  {i:2d}. {drug}")
            continue

        if not drug_name:
            continue

        priority = get_user_preferences()

        min_ee_input = input("\nMinimum EE (%) [default 30]: ").strip()
        max_size_input = input("Maximum size (nm) [default 500]: ").strip()

        min_ee = float(min_ee_input) if min_ee_input else 30.0
        max_size = float(max_size_input) if max_size_input else 500.0

        optimizer.recommend(
            drug_name=drug_name,
            min_ee=min_ee,
            max_size=max_size,
            priority=priority,
            show_top=10,
            auto_save=True
        )

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()