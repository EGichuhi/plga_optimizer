import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression

FEATURES_DIR = Path("data/features")
DATA_DIR     = Path("data/processed")
TARGETS      = ['particle_size', 'EE', 'LC']


X            = np.load(FEATURES_DIR / "X_processed.npy")
feature_names = json.load(open(FEATURES_DIR / "feature_names.json"))
df           = pd.read_csv(FEATURES_DIR / "engineered_data.csv")

print(f"X shape: {X.shape}")
print(f"Features: {len(feature_names)}\n")

feature_names = feature_names[:X.shape[1]]

for target in TARGETS:
    if target not in df.columns:
        print(f"  '{target}' not found in columns, skipping")
        continue

    y        = df[target].values
    valid    = ~np.isnan(y)
    X_valid  = X[valid]
    y_valid  = y[valid]

    mi = mutual_info_regression(X_valid, y_valid, random_state=42)

    scores = pd.DataFrame({'feature': feature_names, 'mi': mi})
    scores = scores.sort_values('mi', ascending=False)

    fp_scores     = scores[scores['feature'].str.startswith('fp_bit_')]
    scalar_scores = scores[~scores['feature'].str.startswith('fp_bit_')]

    zero_fp = (fp_scores['mi'] == 0).sum()

    print(f"── {target} (n={valid.sum()}) ──")
    print(f"  Top 5 scalar:  {scalar_scores.head(5)['feature'].tolist()}")
    print(f"  Top 5 fp bits: {fp_scores.head(5)['feature'].tolist()}")
    print(f"  fp bits with zero MI: {zero_fp}/{len(fp_scores)}")
    print(f"  Max fp MI: {fp_scores['mi'].max():.4f}  |  Scalar max MI: {scalar_scores['mi'].max():.4f}\n")