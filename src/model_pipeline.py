import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
import joblib
import os

from .config import MODEL_CONFIG, TARGETS


class MLPipeline:
    """
    A machine learning pipeline for training and evaluating models on the PLGA dataset.
    """

    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG
        self.models = {}
        self.results = {}
        self.feature_names = None

    # Training 

    def train_models(self, X_train, y_train_dict, feature_names=None):
        """Train one RandomForestRegressor per target."""
        print(f"\n{'='*50}")
        print("TRAINING MODELS")
        print('=' * 50)
        
        # Store feature names and dimensions
        self.feature_names = feature_names
        n_features = X_train.shape[1]
        
        print(f"  X_train shape: {X_train.shape}")
        print(f"  Number of features: {n_features}")
        
        if feature_names is not None:
            print(f"  Feature names provided: {len(feature_names)}")
        else:
            print(f"  ⚠ No feature names provided - will use indices")
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        for target in TARGETS:
            if target not in y_train_dict:
                print(f"  ⚠ Skipping {target} — not in training data")
                continue

            y = y_train_dict[target]
            valid_mask = ~np.isnan(y)
            
            if valid_mask.sum() < len(y):
                print(f"  ⚠ {target}: dropping {(~valid_mask).sum()} rows with NaN targets")
            
            X_fit = X_train[valid_mask]
            y_fit = y[valid_mask]

            print(f"\n  Training {target} on {len(y_fit)} samples...")

            model = RandomForestRegressor(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', None),
                min_samples_split=self.config.get('min_samples_split', 2),
                min_samples_leaf=self.config.get('min_samples_leaf', 1),
                max_features=self.config.get('max_features', 'sqrt'),
                random_state=self.config.get('random_state', 42),
                n_jobs=-1,
            )
            model.fit(X_fit, y_fit)

            self.models[target] = model
            self.results[target] = {
                'model': model,
                'feature_names': self.feature_names,
                'n_train': len(y_fit),
                'n_features': n_features,
            }
            print(f"  ✓ {target} trained (expects {n_features} features)")

        return self.models

    # Cross-validation

    def cross_validate_models(self, X_train, y_train_dict, cv=5):
        """Run k-fold CV to get stable performance estimates."""
        print(f"\n{'='*50}")
        print(f"CROSS-VALIDATION ({cv}-fold, training set only)")
        print('=' * 50)

        for target in TARGETS:
            if target not in y_train_dict:
                continue

            y = y_train_dict[target]
            valid_mask = ~np.isnan(y)
            X_cv = X_train[valid_mask]
            y_cv = y[valid_mask]

            model_proto = RandomForestRegressor(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', None),
                min_samples_split=self.config.get('min_samples_split', 2),
                min_samples_leaf=self.config.get('min_samples_leaf', 1),
                max_features=self.config.get('max_features', 'sqrt'),
                random_state=self.config.get('random_state', 42),
                n_jobs=-1,
            )

            cv_results = cross_validate(
                model_proto, X_cv, y_cv,
                cv=cv,
                scoring=['neg_mean_absolute_error', 'r2'],
                return_train_score=True,
            )

            mae_mean = -cv_results['test_neg_mean_absolute_error'].mean()
            mae_std = cv_results['test_neg_mean_absolute_error'].std()
            r2_mean = cv_results['test_r2'].mean()
            r2_std = cv_results['test_r2'].std()

            print(f"\n  {target}:")
            print(f"    CV MAE : {mae_mean:.2f} ± {mae_std:.2f}")
            print(f"    CV R²  : {r2_mean:.3f} ± {r2_std:.3f}")

            if target not in self.results:
                self.results[target] = {}
            
            self.results[target].update({
                'cv_mae_mean': mae_mean,
                'cv_mae_std': mae_std,
                'cv_r2_mean': r2_mean,
                'cv_r2_std': r2_std,
            })

    # Evaluation on held-out test set

    def evaluate(self, X_test, y_test_dict):
        """Evaluate trained models on held-out test set."""
        print(f"\n{'='*50}")
        print("HELD-OUT TEST SET RESULTS")
        print('=' * 50)
        
        # Diagnostic information
        print(f"\n  DIAGNOSTICS:")
        print(f"    X_test shape: {X_test.shape}")
        print(f"    Expected features from training: {self.results[TARGETS[0]]['n_features'] if TARGETS and TARGETS[0] in self.results else 'Unknown'}")
        
        # Check if X_test is numpy array or DataFrame
        if hasattr(X_test, 'shape'):
            n_features_test = X_test.shape[1]
            n_features_train = self.results[TARGETS[0]]['n_features'] if TARGETS and TARGETS[0] in self.results else None
            
            if n_features_train and n_features_test != n_features_train:
                print(f"\n  FEATURE MISMATCH DETECTED!")
                print(f"     Training had {n_features_train} features")
                print(f"     Test has {n_features_test} features")
                print(f"\n  SOLUTION: Apply the SAME preprocessing pipeline to test data")
                print(f"  Example:")
                print(f"    X_test_preprocessed = preprocessing_pipeline.transform(X_test_raw)")
                print(f"    results = pipeline.evaluate(X_test_preprocessed, y_test_dict)")
                return self.results
        
        # Try to align features if possible
        X_test_aligned = self._align_features(X_test)
        
        for target in TARGETS:
            if target not in self.models or target not in y_test_dict:
                continue

            y_true = y_test_dict[target]
            valid_mask = ~np.isnan(y_true)
            y_true = y_true[valid_mask]
            y_pred = self.models[target].predict(X_test_aligned[valid_mask])

            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            self.results[target].update({
                'test_mae': mae,
                'test_r2': r2,
                'predictions': y_pred,
                'true_values': y_true,
            })

            print(f"\n  {target}:")
            print(f"    MAE : {mae:.2f}")
            print(f"    R²  : {r2:.3f}")

            cv_r2 = self.results[target].get('cv_r2_mean')
            if cv_r2 is not None and (r2 - cv_r2) > 0.1:
                print(f"    ⚠ Test R² ({r2:.3f}) > CV R² ({cv_r2:.3f})")

        return self.results

    def _align_features(self, X):
        """Align input features with training features."""
        n_features_train = self.results[TARGETS[0]]['n_features'] if TARGETS and TARGETS[0] in self.results else None
        
        if n_features_train is None:
            raise ValueError("No trained models found. Train or load models first.")
        
        # If X is a DataFrame, try to select matching columns
        if hasattr(X, 'columns') and self.feature_names is not None:
            print(f"\n  Attempting to align DataFrame columns...")
            print(f"    Test columns: {len(X.columns)}")
            print(f"    Expected columns: {len(self.feature_names)}")
            
            # Find common columns
            common_cols = [col for col in self.feature_names if col in X.columns]
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            extra_cols = [col for col in X.columns if col not in self.feature_names]
            
            if missing_cols:
                print(f"  Missing {len(missing_cols)} columns, filling with 0")
                for col in missing_cols[:5]:  # Show first 5 missing
                    print(f"      - {col}")
            
            if extra_cols:
                print(f"    ⚠ Extra {len(extra_cols)} columns, will be ignored")
            
            # Reindex to match training features
            X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
            return X_aligned.values
        
        # If X is numpy array, check shape
        elif isinstance(X, np.ndarray):
            if X.shape[1] != n_features_train:
                raise ValueError(
                    f"\n Feature dimension mismatch!\n"
                    f"   Training data had {n_features_train} features\n"
                    f"   Test data has {X.shape[1]} features\n"
                    f"\n   This usually means:\n"
                    f"   1. You forgot to apply preprocessing to test data\n"
                    f"   2. You're using raw data instead of preprocessed data\n"
                    f"   3. Training and test data were preprocessed differently\n"
                    f"\n   Fix: Apply the SAME preprocessing pipeline to test data:\n"
                    f"   X_test_processed = pipeline.transform(X_test_raw)"
                )
            return X
        
        else:
            raise TypeError(f"Unsupported X type: {type(X)}")

    # Feature importance 

    def get_feature_importance(self, top_n_scalar=5, top_n_fp=10):
        """Return top scalar features and fingerprint bits by importance."""
        first_target = next(
            (t for t in TARGETS if t in self.results and 
             self.results[t].get('feature_names') is not None),
            None
        )
        
        if first_target is None:
            print("No feature names available — train models first.")
            return None, None

        feature_names = self.results[first_target]['feature_names']
        
        # Collect all feature importances
        rows = []
        for target in TARGETS:
            if target in self.models:
                imp = self.models[target].feature_importances_
                for name, val in zip(feature_names, imp):
                    rows.append({'feature': name, 'target': target, 'importance': val})

        df = pd.DataFrame(rows)
        
        # Calculate mean importance across targets
        mean_imp = df.groupby('feature')['importance'].mean().reset_index()
        mean_imp.columns = ['feature', 'mean_importance']
        df = df.merge(mean_imp, on='feature')

        # Split into scalar and fingerprint features
        fp_mask = df['feature'].str.startswith('fp_bit_')
        
        scalar_df = (df[~fp_mask]
                    .drop_duplicates('feature')
                    .sort_values('mean_importance', ascending=False)
                    .head(top_n_scalar)
                    .reset_index(drop=True))

        fp_df = (df[fp_mask]
                .drop_duplicates('feature')
                .sort_values('mean_importance', ascending=False)
                .head(top_n_fp)
                .reset_index(drop=True))

        total_fp_imp = df[fp_mask].drop_duplicates('feature')['mean_importance'].sum()
        print(f"\n  Top {top_n_scalar} scalar features and top {top_n_fp} fp bits shown.")
        print(f"  Total fingerprint importance: {total_fp_imp:.3f}")

        return scalar_df, fp_df

    # Saving and Loading

    def save_models(self, preprocess_pipe=None, path='models/'):
        """Save models, results, and preprocessing pipeline."""
        os.makedirs(path, exist_ok=True)

        # Save feature names
        feature_names_path = os.path.join(path, 'feature_names.pkl')
        joblib.dump(self.feature_names, feature_names_path)
        print(f"  ✓ Saved feature names to {feature_names_path}")

        # Save individual models
        for target, model in self.models.items():
            out = os.path.join(path, f'{target}_model.pkl')
            joblib.dump(model, out)
            print(f"  ✓ Saved {out}")

        # Save results
        results_out = os.path.join(path, 'results.pkl')
        joblib.dump(self.results, results_out)
        print(f"  ✓ Saved {results_out}")

        # Save preprocessing pipeline if provided
        if preprocess_pipe is not None:
            pipe_out = os.path.join(path, 'preprocessing_pipeline.pkl')
            joblib.dump(preprocess_pipe, pipe_out)
            print(f"  ✓ Saved {pipe_out}")
        else:
            print("  ⚠ No preprocessing pipeline provided")

        print(f"\n✓ All models saved to '{path}'")
    
    def load_models(self, path='models/'):
        """Load previously saved models and feature names."""
        # Load feature names
        feature_names_path = os.path.join(path, 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
            print(f"  ✓ Loaded feature names from {feature_names_path}")
        
        # Load models
        for target in TARGETS:
            model_path = os.path.join(path, f'{target}_model.pkl')
            if os.path.exists(model_path):
                self.models[target] = joblib.load(model_path)
                print(f"  ✓ Loaded {target} model from {model_path}")
        
        # Load results
        results_path = os.path.join(path, 'results.pkl')
        if os.path.exists(results_path):
            self.results = joblib.load(results_path)
            print(f"  ✓ Loaded results from {results_path}")
        
        return self.models