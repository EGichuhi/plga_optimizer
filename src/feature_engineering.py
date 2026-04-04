# src/feature_engineering.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from sklearn.impute import SimpleImputer

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')

# Configuration for features


POLYMER_FEATURES = ['polymer_MW', 'LA/GA']
DRUG_FEATURES = ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 
                 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms']
FORMULATION_FEATURES = ['drug/polymer', 'surfactant_concentration', 'surfactant_HLB']
PROCESS_FEATURES = ['aqueous/organic', 'pH', 'solvent_polarity_index']
ENGINEERED_FEATURES = ['LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa', 
                       'drug_logP_normalized', 'polymer_drug_affinity', 'la_ga_drug_logP']

ALL_SCALAR_FEATURES = POLYMER_FEATURES + DRUG_FEATURES + FORMULATION_FEATURES + PROCESS_FEATURES + ENGINEERED_FEATURES

# SMILES 

def add_smiles_from_lookup(df13, df2, df4=None):
    """
    Add canonical SMILES to df13 by looking up from df2.
    df4 is optional - only used if df2 doesn't have the SMILES.
    """
    smiles_lookup = {}
    
    for _, row in df2.iterrows():
        if pd.notna(row.get('small_molecule_name')) and pd.notna(row.get('canonical_SMILES')):
            smiles_lookup[row['small_molecule_name']] = row['canonical_SMILES']
    
    # Only use df4 if provided and SMILES missing in df2
    if df4 is not None and 'canonical_SMILES' in df4.columns:
        for _, row in df4.iterrows():
            name = row.get('small_molecule_name')
            smiles = row.get('canonical_SMILES')
            if pd.notna(name) and pd.notna(smiles) and name not in smiles_lookup:
                smiles_lookup[name] = smiles
    
    df13_with_smiles = df13.copy()
    df13_with_smiles['canonical_SMILES'] = df13_with_smiles['small_molecule_name'].map(smiles_lookup)
    
    matched = df13_with_smiles['canonical_SMILES'].notna().sum()
    print(f"Added SMILES to {matched}/{len(df13_with_smiles)} rows ({matched/len(df13_with_smiles)*100:.1f}%)")
    
    return df13_with_smiles

# Fingerprint generation using RDKit's Morgan fingerprints 

def _safe_get_fingerprint(smiles, radius=2, nBits=2048):
    """Internal function - generates fingerprint for a single SMILES"""
    if pd.isna(smiles):
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = mfp_gen.GetFingerprint(mol)
        return list(fp)
    except Exception:
        return None

def generate_morgan_fingerprints(df, smiles_col='canonical_SMILES', radius=2, nBits=2048):
    """Generate Morgan fingerprints for all rows"""
    df_with_fp = df.copy()
    df_with_fp['morgan_fingerprint'] = df_with_fp[smiles_col].apply(
        lambda x: _safe_get_fingerprint(x, radius, nBits)
    )
    
    valid_count = df_with_fp['morgan_fingerprint'].notna().sum()
    print(f"Generated fingerprints for {valid_count}/{len(df_with_fp)} molecules")
    
    if valid_count > 0:
        sample_fp = df_with_fp['morgan_fingerprint'].dropna().iloc[0]
        print(f"  Length: {len(sample_fp)} bits")
        print(f"  Density: {np.mean(sample_fp):.3f}")
    
    return df_with_fp

# Feature engineering

def create_engineered_features(df):
    """Create engineered features for PLGA optimization"""
    df_eng = df.copy()
    
    # Polymer-derived
    df_eng['LA_mol_fraction'] = df_eng['LA/GA'] / (1 + df_eng['LA/GA'])
    df_eng['log_polymer_MW'] = np.log10(df_eng['polymer_MW'])
    df_eng['polymer_MW_kDa'] = df_eng['polymer_MW'] / 1000
    
    # Drug-derived (normalized)
    df_eng['drug_logP_normalized'] = (
        (df_eng['mol_logP'] - df_eng['mol_logP'].mean()) / 
        df_eng['mol_logP'].std()
    )
    
    # Interaction features
    df_eng['polymer_drug_affinity'] = df_eng['log_polymer_MW'] * df_eng['drug_logP_normalized']
    df_eng['la_ga_drug_logP'] = df_eng['LA_mol_fraction'] * df_eng['drug_logP_normalized']
    
    print(f"Created {len(ENGINEERED_FEATURES)} engineered features")
    return df_eng

# Feature matrix preparation

def prepare_feature_matrix(df, feature_list):
    """Extract and impute features"""
    missing_counts = df[feature_list].isnull().sum()
    
    if missing_counts.sum() > 0:
        print(f"Missing values found, imputing with median...")
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(df[feature_list])
        return X, imputer
    else:
        print("No missing values found")
        return df[feature_list].values, None