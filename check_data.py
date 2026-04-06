import os
import pandas as pd

print("="*60)
print("Checking Data Files")
print("="*60)

data_path = 'data/processed/'
files = ['dataset1.csv', 'dataset2.csv', 'dataset3.csv', 'dataset4.csv']

print(f"\nLooking for files in: {data_path}")
print("-" * 40)

for file in files:
    full_path = os.path.join(data_path, file)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        print(f" {file} ({size:,} bytes)")
    else:
        print(f" {file} - NOT FOUND")
        print(f"  Tried: {os.path.abspath(full_path)}")

print("-" * 40)
print("\n Checking if df1 and df3 were merged into df13")

# Load dataset1 and dataset3 to check merge
try:
    df1 = pd.read_csv(os.path.join(data_path, 'dataset1.csv'))
    df3 = pd.read_csv(os.path.join(data_path, 'dataset3.csv'))
    df13 = pd.read_csv(os.path.join(data_path, 'df13.csv'))
    df13_with_smiles = pd.read_csv(os.path.join(data_path, 'df13_with_smiles.csv'))
    
    # Check if df13 has rows from both df1 and df3 (horizontal concat, same number of rows)
    if len(df13) == len(df1) and len(df13) == len(df3):
        print("\n Merge check: PASSED - df13 has the same number of rows as df1 and df3")
    else:
        print("\n Merge check: FAILED - df13 does not have matching row count")
    print("-" * 40)

    # Check if canonical_SMILES column is present in df13_with_smiles
    if 'canonical_SMILES' in df13_with_smiles.columns:
        print(" Smiles check: PASSED - 'canonical_SMILES' column found in df13_with_smiles")
        print(f" Smiles column has {df13_with_smiles['canonical_SMILES'].notna().sum()} non-null values")
    else:
        print(" Smiles check: FAILED - 'canonical_SMILES' column not found in df13_with_smiles")
       
    print(f" df13_with_smiles shape: {df13_with_smiles.shape}")
    print("-" * 40)

    # Check for df13_with_features
    df13_with_features_path = os.path.join(data_path, 'df13_with_features.csv')
    if os.path.exists(df13_with_features_path):
        df_features = pd.read_csv(df13_with_features_path)
        print(" Features check: PASSED - df13_with_features.csv exists")
        print(f"  df13_with_features shape: {df_features.shape}")
        
        # Check for scalar features
        scalar_features = ['polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 
                           'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms', 'drug/polymer', 
                           'surfactant_concentration', 'surfactant_HLB', 'aqueous/organic', 'pH', 
                           'solvent_polarity_index', 'LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa', 
                           'drug_logP_normalized', 'polymer_drug_affinity', 'la_ga_drug_logP']
        missing_scalars = [f for f in scalar_features if f not in df_features.columns]
        if not missing_scalars:
            print("  Scalar features: PASSED - all required scalar features present")
        else:
            print(f"  Scalar features: FAILED - missing {missing_scalars}")
        print("-" * 40)

        # Check for fingerprint
        if 'morgan_fingerprint' in df_features.columns:
            print("  Fingerprint check: PASSED - 'morgan_fingerprint' column found")
            fp_count = df_features['morgan_fingerprint'].notna().sum()
            print(f"    Fingerprints generated for {fp_count}/{len(df_features)} rows")
        else:
            print("  Fingerprint check: FAILED - 'morgan_fingerprint' column not found")
        
        print(f"  df13_with_features columns: {df_features.columns.tolist()[:10]}...")  
    else:
        print(" Features check: FAILED - df13_with_features.csv not found")
    
    
except Exception as e:
    print(f" Error during checks: {e}")

print("\n" + "="*60)
