print("="*50)
print("Testing Imports")
print("="*50)

try:
    from src.data_loader import DataLoader
    print("✓ DataLoader imported successfully")
except Exception as e:
    print(f"✗ DataLoader failed: {e}")

try:
    from src.feature_engineering import (
        add_smiles_from_lookup,
        generate_morgan_fingerprints,
        create_engineered_features,
        prepare_feature_matrix,
        ALL_SCALAR_FEATURES
    )
    print("✓ feature_engineering imported successfully")
except Exception as e:
    print(f"✗ feature_engineering failed: {e}")

try:
    from src.model_pipeline import MLPipeline
    print("✓ model_pipeline imported successfully")
except Exception as e:
    print(f"✗ model_pipeline failed: {e}")

try:
    from src.config import MODEL_CONFIG, TARGETS
    print("✓ config imported successfully")
except Exception as e:
    print(f"✗ config failed: {e}")

print("\n" + "="*50)
print("All imports completed!")
