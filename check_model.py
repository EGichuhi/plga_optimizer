import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('models/particle_size_model.pkl')
df13 = pd.read_csv('data/processed/df13_with_features.csv')

# Test with different inputs
features = [
    'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 
    'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 
    'mol_heteroatoms', 'drug/polymer', 'surfactant_concentration', 
    'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index',
    'LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa', 
    'drug_logP_normalized', 'polymer_drug_affinity', 'la_ga_drug_logP'
]

# Test with extreme values
test_cases = [
    [10000, 0.5, 300, 2.0, 60, 150, 4, 2, 6, 0.1, 1.5, 15, 1, 7, 0.8, 0.33, 4, 10, 0, 0, 0],
    [100000, 4.0, 500, 4.0, 120, 250, 8, 4, 12, 0.3, 3.0, 15, 1, 7, 0.8, 0.8, 5, 100, 1, 5, 4],
]

for i, test in enumerate(test_cases):
    pred = model.predict([test])[0]
    print(f"Test {i+1}: {pred:.1f} nm")

# Check feature ranges in training data
print("\nTraining data ranges:")
for f in features[:5]:
    print(f"  {f}: {df13[f].min():.2f} - {df13[f].max():.2f}")