import pandas as pd
import numpy as np
import joblib

print("Loading models...")
models = {}
for target in ['particle_size', 'EE', 'LC']:
    models[target] = joblib.load(f'models/{target}_model.pkl')
    print(f"✓ Loaded {target} model")

print("\nLoading data...")
df13 = pd.read_csv('data/processed/df13_with_features.csv')
print(f"✓ Loaded {len(df13)} formulations")

# Define ONLY numeric features (exclude string columns)
numeric_features = [
    'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 
    'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 
    'mol_heteroatoms', 'drug/polymer', 'surfactant_concentration', 
    'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index',
    'LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa', 
    'drug_logP_normalized', 'polymer_drug_affinity', 'la_ga_drug_logP'
]

# Check which features exist in df13
available_features = [f for f in numeric_features if f in df13.columns]
print(f"Using {len(available_features)} numeric features")

# Take first row and select only numeric features
test_row = df13[available_features].iloc[0:1]

print(f"\nFeature matrix shape: {test_row.shape}")
print(f"Data types: {test_row.dtypes.value_counts()}")

print("\nTesting prediction for first formulation...")
for target, model in models.items():
    pred = model.predict(test_row)[0]
    actual = df13[target].iloc[0]
    print(f"{target}: Predicted={pred:.1f}, Actual={actual:.1f}")

print("\nModels work! ")