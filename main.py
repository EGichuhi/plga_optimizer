import pandas as pd
import numpy as np
import os

from src.data_loader import DataLoader
from src.feature_engineering import (
    generate_morgan_fingerprints,
    create_engineered_features,
    prepare_feature_matrix,
    ALL_SCALAR_FEATURES
)
from src.model_pipeline import MLPipeline
from src.config import TARGETS


def check_and_prepare_data(loader):
    """Check if data files exist, if not, create them"""
    
    # Check if we already have the final dataset
    if os.path.exists(os.path.join(loader.data_path, 'df13_with_features.csv')):
        print("✓ Found existing df13_with_features.csv")
        return loader.load_df13_with_features()
    
    # If not, build from scratch
    print(" Preparing data for first time use...")
    
    # Step 1: Build df13 (D1 + D3 concat)
    try:
        df13 = loader.load_df13()
        print("✓ Found existing df13.csv")
    except FileNotFoundError:
        print("Building df13.csv from raw data...")
        df13 = loader.build_df13()
    
    # Step 2: Add SMILES
    try:
        df13_with_smiles = loader.load_df13_with_smiles()
        print("✓ Found existing df13_with_smiles.csv")
    except FileNotFoundError:
        print("Adding SMILES to create df13_with_smiles.csv...")
        df13_with_smiles = loader.build_df13_with_smiles()
    
    # Return the dataset with SMILES (fingerprints will be added in main)
    return df13_with_smiles


def main():
    print("="*60)
    print(" PLGA Formulation Optimization Pipeline")
    print("="*60)
    print("This will prepare your data, train models, and save everything.")
    print("="*60)
    
    # Initialize loader
    loader = DataLoader(data_path='data/processed/')
    
    # 1. Check and prepare data (automatically handles everything)
    print("\n[1/6] Checking and preparing data...")
    df13 = check_and_prepare_data(loader)
    print(f"   Dataset shape: {df13.shape}")
    print(f"   Columns: {df13.columns.tolist()[:5]}...")
    
    # 2. Generate Morgan fingerprints
    print("\n[2/6] Generating Morgan fingerprints...")
    if 'morgan_fingerprint' not in df13.columns:
        df13 = generate_morgan_fingerprints(df13, smiles_col='canonical_SMILES')
    else:
        print("   ✓ Fingerprints already exist")
    
    # 3. Create engineered features
    print("\n[3/6] Creating engineered features...")
    engineered_cols = ['LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa', 
                       'drug_logP_normalized', 'polymer_drug_affinity', 'la_ga_drug_logP']
    
    if not all(col in df13.columns for col in engineered_cols):
        df13 = create_engineered_features(df13)
    else:
        print("   ✓ Engineered features already exist")
    
    # 4. Prepare feature matrix
    print("\n[4/6] Preparing feature matrix...")
    available_features = [f for f in ALL_SCALAR_FEATURES if f in df13.columns]
    print(f"   Using {len(available_features)} features")
    
    X, imputer = prepare_feature_matrix(df13, available_features)
    y_dict = {target: df13[target].values for target in TARGETS if target in df13.columns}
    
    # 5. Train and evaluate models
    print("\n[5/6] Training and evaluating models...")
    pipeline = MLPipeline()
    X_train, X_test, y_train_dict, y_test_dict = pipeline.split_data(X, y_dict)
    pipeline.train_models(X_train, y_train_dict, feature_names=available_features)
    pipeline.evaluate(X_test, y_test_dict)
    
    # 6. Save everything
    print("\n[6/6] Saving results...")
    pipeline.save_models()
    df13.to_csv('data/processed/df13_with_features.csv', index=False)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
    print("\nWhat was saved:")
    print("   • models/particle_size_model.pkl")
    print("   • models/EE_model.pkl")
    print("   • models/LC_model.pkl")
    print("   • data/processed/df13.csv (D1 + D3 concat)")
    print("   • data/processed/df13_with_smiles.csv (with SMILES)")
    print("   • data/processed/df13_with_features.csv (final for ML)")
    print("\nYou can now run:")
    print("   python cl_optimizer.py")
    print("="*60)
    
    return pipeline, df13


if __name__ == "__main__":
    pipeline, df13 = main()