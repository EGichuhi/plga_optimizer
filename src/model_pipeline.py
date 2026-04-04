# src/model_pipeline.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

from src.config import MODEL_CONFIG, TARGETS


class MLPipeline:
    """Machine learning pipeline for PLGA formulation prediction"""
    
    def __init__(self, config=None):
        self.config = config or MODEL_CONFIG
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train_dict = None
        self.y_test_dict = None
    
    def split_data(self, X, y_dict):
        """Train-test split - single source of truth"""
        X_train, X_test = train_test_split(
            X,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        y_train_dict = {}
        y_test_dict = {}
        
        for target in TARGETS:
            if target in y_dict:
                y_train, y_test = train_test_split(
                    y_dict[target],
                    test_size=self.config['test_size'],
                    random_state=self.config['random_state']
                )
                y_train_dict[target] = y_train
                y_test_dict[target] = y_test
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train_dict = y_train_dict
        self.y_test_dict = y_test_dict
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train_dict, y_test_dict
    
    def train_models(self, X_train, y_train_dict, feature_names=None):
        """Train models - uses config for all parameters"""
        print(f"\n{'='*50}")
        print("TRAINING MODELS")
        print('='*50)
        
        for target in TARGETS:
            if target not in y_train_dict:
                print(f"⚠️ Skipping {target} - not in training data")
                continue
                
            print(f"\nTraining model for {target}...")
            
            model = RandomForestRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', None),
                min_samples_split=self.config.get('min_samples_split', 2),
                min_samples_leaf=self.config.get('min_samples_leaf', 1),
                random_state=self.config.get('random_state', 42),
                n_jobs=-1
            )
            
            model.fit(X_train, y_train_dict[target])
            self.models[target] = model
            self.results[target] = {'model': model, 'feature_names': feature_names}
            
            print(f"  ✓ {target} model trained")
        
        return self.models
    
    def evaluate(self, X_test, y_test_dict):
        """Evaluate all models"""
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print('='*50)
        
        for target in TARGETS:
            if target not in self.models or target not in y_test_dict:
                continue
                
            y_pred = self.models[target].predict(X_test)
            mae = mean_absolute_error(y_test_dict[target], y_pred)
            r2 = r2_score(y_test_dict[target], y_pred)
            
            self.results[target]['test_mae'] = mae
            self.results[target]['test_r2'] = r2
            self.results[target]['predictions'] = y_pred
            self.results[target]['true_values'] = y_test_dict[target]
            
            print(f"\n{target}:")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²:  {r2:.3f}")
        
        return self.results
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance DataFrame"""
        if feature_names is None and TARGETS[0] in self.results:
            feature_names = self.results[TARGETS[0]].get('feature_names')
        
        if feature_names is None:
            return None
        
        importance_df = pd.DataFrame({'feature': feature_names})
        
        for target in TARGETS:
            if target in self.models:
                importance_df[f'importance_{target}'] = self.models[target].feature_importances_
        
        return importance_df.sort_values(f'importance_{TARGETS[0]}', ascending=False)
    
    def save_models(self, path='models/'):
        """Save models and results"""
        os.makedirs(path, exist_ok=True)
        
        for target, model in self.models.items():
            joblib.dump(model, f'{path}/{target}_model.pkl')
        
        joblib.dump(self.results, f'{path}/results.pkl')
        print(f"\n✓ Models saved to {path}")