from sklearn.model_selection import train_test_split

from src.data_loader import DataLoader
from src.model_pipeline import MLPipeline
from src.config import TARGETS, MODEL_CONFIG
from src.feature_engineering import (
    build_full_feature_matrix,
    build_preprocessing_pipeline,
    ALL_SCALAR_FEATURES,
)

# 1. Train / test split 
def split_data(df):
    print("\n" + "=" * 70)
    print("STEP 1: TRAIN/TEST SPLIT")
    print("=" * 70)

    available_targets = [t for t in TARGETS if t in df.columns]
    if not available_targets:
        raise ValueError(f"None of the expected targets {TARGETS} found in data.")

    train_df, test_df = train_test_split(
        df,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state'],
    )

    print(f"  Training rows : {len(train_df)}")
    print(f"  Test rows     : {len(test_df)}")
    print(f"  Targets       : {available_targets}")

    return train_df, test_df, available_targets


# 2. Build feature matrices 

def build_feature_matrices(train_df, test_df):
    print("\n" + "=" * 70)
    print("STEP 2: BUILDING FEATURE MATRICES")
    print("=" * 70)

    X_train, _, feature_names = build_full_feature_matrix(
        train_df, scalar_features=ALL_SCALAR_FEATURES, fp_col='morgan_fingerprint'
    )
    X_test, _, _ = build_full_feature_matrix(
        test_df, scalar_features=ALL_SCALAR_FEATURES, fp_col='morgan_fingerprint'
    )

    n_scalar = len(ALL_SCALAR_FEATURES)
    print(f"  Train : {X_train.shape}  ({n_scalar} scalar + {X_train.shape[1] - n_scalar} fp bits)")
    print(f"  Test  : {X_test.shape}")

    return X_train, X_test, feature_names, n_scalar


# 3. Preprocess

def preprocess(X_train, X_test, n_scalar, fp_variance_threshold=0.1):
    print("\n" + "=" * 70)
    print("STEP 3: PREPROCESSING")
    print("=" * 70)

    preprocessor = build_preprocessing_pipeline(
        n_scalar_features=n_scalar,
        fp_variance_threshold=fp_variance_threshold,
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    print(f"  After preprocessing: {X_train_proc.shape[1]} features")
    print(f"  Train : {X_train_proc.shape}  |  Test : {X_test_proc.shape}")

    try:
        final_feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        final_feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    return X_train_proc, X_test_proc, preprocessor, final_feature_names


#  4. Train and evaluate 

def train_and_evaluate(X_train, X_test, train_df, test_df, feature_names, available_targets):
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING AND EVALUATION")
    print("=" * 70)

    y_train_dict = {t: train_df[t].values for t in available_targets}
    y_test_dict  = {t: test_df[t].values  for t in available_targets}

    pipeline = MLPipeline()
    pipeline.train_models(X_train, y_train_dict, feature_names=feature_names)
    pipeline.cross_validate_models(X_train, y_train_dict, cv=MODEL_CONFIG.get('cv_folds', 5))
    pipeline.evaluate(X_test, y_test_dict)

    return pipeline

