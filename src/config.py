# src/config.py

# Data paths
DATA_PATH = {
    'df1': 'data/processed/dataset1.csv',
    'df2': 'data/processed/dataset2.csv',
    'df3': 'data/processed/dataset3.csv',
    'df4': 'data/processed/dataset4.csv'
}

# Model parameters - ADD ALL THESE KEYS
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': None,           
    'min_samples_split': 2,     
    'min_samples_leaf': 1,      
    'cv_folds': 5
}

# Target variables
TARGETS = ['particle_size', 'EE', 'LC']