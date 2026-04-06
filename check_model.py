# check_model.py
import numpy as np
import pandas as pd
import joblib
from src.model_pipeline import MLPipeline
from src.config import TARGETS

def test_ml_pipeline():
    """Test that the MLPipeline trains and evaluates correctly."""
    
    print("="*60)
    print("TESTING ML PIPELINE")
    print("="*60)
    
    # Create synthetic test data (21 raw features)
    n_samples = 100
    np.random.seed(42)
    
    # Simulate raw data (21 features as in your check_model)
    raw_data = pd.DataFrame({
        'PLGA_dl': np.random.uniform(0.5, 1.0, n_samples),
        'PLGA_Mw': np.random.uniform(10000, 50000, n_samples),
        'PLGA_LA_GA': np.random.uniform(25, 75, n_samples),
        'drug_logP': np.random.uniform(0, 5, n_samples),
        'drug_Mw': np.random.uniform(100, 500, n_samples),
        'drug_pKa': np.random.uniform(2, 10, n_samples),
        'drug_HBD': np.random.randint(0, 5, n_samples),
        'drug_HBA': np.random.randint(0, 8, n_samples),
        'drug_RB': np.random.randint(0, 10, n_samples),
        'polymer_mass': np.random.uniform(50, 150, n_samples),
        'drug_mass': np.random.uniform(5, 20, n_samples),
        'solvent_volume': np.random.uniform(1, 5, n_samples),
        'solvent_type': np.random.choice(['DCM', 'Ethyl acetate', 'Acetone'], n_samples),
        'surfactant_type': np.random.choice(['PVA', 'Pluronic', 'SDS'], n_samples),
        'surfactant_concentration': np.random.uniform(0.5, 3, n_samples),
        'mixing_speed': np.random.uniform(500, 2000, n_samples),
        'mixing_time': np.random.uniform(1, 10, n_samples),
        'injection_rate': np.random.uniform(0.5, 5, n_samples),
        'evaporation_rate': np.random.uniform(5, 20, n_samples),
        'temperature': np.random.uniform(20, 30, n_samples),
        'volume_ratio': np.random.uniform(5, 20, n_samples),
    })
    
    # Simulate target variables
    y_data = pd.DataFrame({
        'encapsulation_efficiency': np.random.uniform(60, 95, n_samples),
        'particle_size': np.random.uniform(100, 300, n_samples),
        'pdi': np.random.uniform(0.1, 0.4, n_samples),
        'zeta_potential': np.random.uniform(-30, -10, n_samples),
        'loading_capacity': np.random.uniform(5, 20, n_samples),
    })
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train_raw = raw_data.iloc[:split_idx]
    X_test_raw = raw_data.iloc[split_idx:]
    y_train = y_data.iloc[:split_idx]
    y_test = y_data.iloc[split_idx:]
    
    print(f"\n✓ Created synthetic data:")
    print(f"  Training samples: {len(X_train_raw)}")
    print(f"  Test samples: {len(X_test_raw)}")
    print(f"  Raw features: {X_train_raw.shape[1]}")
    print(f"  Targets: {list(y_train.columns)}")
    
    # Create and fit preprocessing pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Identify column types
    categorical_cols = ['solvent_type', 'surfactant_type']
    numerical_cols = [col for col in X_train_raw.columns if col not in categorical_cols]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
        ]
    )
    
    # Fit on training data only
    X_train_preprocessed = preprocessor.fit_transform(X_train_raw)
    X_test_preprocessed = preprocessor.transform(X_test_raw)
    
    print(f"\n✓ Applied preprocessing:")
    print(f"  Training features: {X_train_preprocessed.shape[1]}")
    print(f"  Test features: {X_test_preprocessed.shape[1]}")
    
    # Convert y to dictionary format expected by MLPipeline
    y_train_dict = {col: y_train[col].values for col in TARGETS if col in y_train.columns}
    y_test_dict = {col: y_test[col].values for col in TARGETS if col in y_test.columns}
    
    # Initialize and train pipeline
    pipeline = MLPipeline()
    
    # Train models
    pipeline.train_models(
        X_train_preprocessed, 
        y_train_dict, 
        feature_names=preprocessor.get_feature_names_out()
    )
    
    # Cross-validate
    pipeline.cross_validate_models(X_train_preprocessed, y_train_dict, cv=3)
    
    # Evaluate on test set
    pipeline.evaluate(X_test_preprocessed, y_test_dict)
    
    # Get feature importance
    scalar_df, fp_df = pipeline.get_feature_importance(top_n_scalar=5, top_n_fp=5)
    
    # Save models
    pipeline.save_models(preprocess_pipe=preprocessor, path='test_models/')
    
    # Test loading models back
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    new_pipeline = MLPipeline()
    new_pipeline.load_models(path='test_models/')
    
    # Make prediction on a single sample
    print("\n" + "="*60)
    print("TESTING SINGLE PREDICTION")
    print("="*60)
    
    # Single raw sample (21 features)
    single_sample = pd.DataFrame([{
        'PLGA_dl': 0.75,
        'PLGA_Mw': 20000,
        'PLGA_LA_GA': 50,
        'drug_logP': 2.5,
        'drug_Mw': 300,
        'drug_pKa': 7.4,
        'drug_HBD': 2,
        'drug_HBA': 4,
        'drug_RB': 6,
        'polymer_mass': 100,
        'drug_mass': 10,
        'solvent_volume': 2,
        'solvent_type': 'DCM',
        'surfactant_type': 'PVA',
        'surfactant_concentration': 1,
        'mixing_speed': 1000,
        'mixing_time': 5,
        'injection_rate': 1,
        'evaporation_rate': 10,
        'temperature': 25,
        'volume_ratio': 10
    }])
    
    # Apply preprocessing
    single_preprocessed = preprocessor.transform(single_sample)
    
    # Make predictions through the loaded pipeline
    print("\n  Predictions for single sample:")
    for target in TARGETS:
        if target in new_pipeline.models:
            pred = new_pipeline.models[target].predict(single_preprocessed)[0]
            print(f"    {target:25s}: {pred:.2f}")
    
    # Verify that prediction works (no error)
    assert len(new_pipeline.models) > 0, "Models not loaded properly"
    print("\n✓ All tests passed!")
    
    # Clean up test files
    import shutil
    if os.path.exists('test_models'):
        shutil.rmtree('test_models')
        print("✓ Cleaned up test files")

if __name__ == "__main__":
    import os
    test_ml_pipeline()