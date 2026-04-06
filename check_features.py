import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import (
    add_smiles_from_lookup,
    generate_morgan_fingerprints,
    filter_valid_fingerprints,
    create_engineered_features,
    build_full_feature_matrix,
    build_preprocessing_pipeline,
    ALL_SCALAR_FEATURES
)

logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler('feature_engineering.log'),
                               logging.StreamHandler()])
logger = logging.getLogger(__name__)

SEP = "-" * 40


class FeatureEngineeringPipeline:
    def __init__(self, data_path: str = "data/processed", output_dir: str = "data/features"):
        self.data_path  = Path(data_path)
        self.output_dir = Path(output_dir)
        self.dfs        = {}
        self.results    = {}

    def setup_directories(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def load_data(self, file_config: dict) -> bool:
        for name, filename in file_config.get('required', []):
            try:
                self.dfs[name] = pd.read_csv(self.data_path / filename)
                logger.info(f"Loaded {name}: {self.dfs[name].shape}")
            except FileNotFoundError:
                logger.error(f"Required file not found: {filename}")
                return False
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                return False

        for name, filename in file_config.get('optional', []):
            path = self.data_path / filename
            try:
                self.dfs[name] = pd.read_csv(path) if path.exists() else None
                logger.info(f"Loaded {name}: {self.dfs[name].shape}" if self.dfs[name] is not None
                            else f"Optional file not found: {filename}")
            except Exception as e:
                logger.warning(f"Could not load {filename}: {e}")
                self.dfs[name] = None

        print(SEP)
        return True

    def add_smiles(self) -> Optional[pd.DataFrame]:
        logger.info("Step 1: Adding SMILES")
        out = add_smiles_from_lookup(self.dfs['df13'], self.dfs.get('df2', pd.DataFrame()), self.dfs.get('df4'))
        if out['canonical_SMILES'].isna().all():
            logger.error("No SMILES found — cannot generate fingerprints.")
            return None
        self.results['smiles_added'] = out
        print(SEP)
        return out

    def generate_fingerprints(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Step 2: Generating Morgan fingerprints")
        out = generate_morgan_fingerprints(df)
        self.results['fingerprints'] = out
        print(SEP)
        return out

    def filter_fingerprints(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        logger.info("Step 3: Filtering valid fingerprints")
        out = filter_valid_fingerprints(df)
        if len(out) == 0:
            logger.error("No valid fingerprints generated.")
            return None
        self.results['filtered'] = out
        print(SEP)
        return out

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Step 4: Creating engineered features")
        out = create_engineered_features(df)
        self.results['engineered'] = out
        print(SEP)
        return out

    def build_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Index, list]:
        logger.info("Step 5: Building feature matrix")
        X, indices, feature_names = build_full_feature_matrix(df)
        self.results['feature_matrix'] = {'X': X, 'indices': indices, 'names': feature_names}
        print(SEP)
        return X, indices, feature_names

    def preprocess_features(self, X: np.ndarray) -> np.ndarray:
        logger.info("Step 6: Preprocessing")
        pipeline  = build_preprocessing_pipeline(len(ALL_SCALAR_FEATURES), fp_variance_threshold=0.01)
        X_out     = pipeline.fit_transform(X)
        joblib.dump(pipeline, self.output_dir / 'preprocessing_pipeline.joblib')
        logger.info(f"Final shape: {X_out.shape}")
        print(SEP)
        return X_out

    def save_results(self, X: np.ndarray, feature_names: list, df: pd.DataFrame):
        logger.info("Step 7: Saving results")
        np.save(self.output_dir / "X_processed.npy", X)
        df.to_csv(self.output_dir / "engineered_data.csv", index=False)

        with open(self.output_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)

        metadata = {
            'input_shape':        len(self.dfs.get('df13', [])),
            'output_shape':       X.shape,
            'n_scalar_features':  len(ALL_SCALAR_FEATURES),
            'n_fingerprint_bits': X.shape[1] - len(ALL_SCALAR_FEATURES),
        }
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved to {self.output_dir}/")

    def print_summary(self, input_count: int, after_smiles: int, after_filtering: int, final_shape: Tuple):
        logger.info("\n" + "=" * 60)
        logger.info(f"Input samples:              {input_count}")
        logger.info(f"After SMILES lookup:        {after_smiles}")
        logger.info(f"After fingerprint filter:   {after_filtering}")
        logger.info(f"Final features:             {final_shape[1]}")
        logger.info(f"  scalar: {len(ALL_SCALAR_FEATURES)}  |  fp bits: {final_shape[1] - len(ALL_SCALAR_FEATURES)}")

    def run(self, file_config: dict) -> bool:
        try:
            self.setup_directories()
            if not self.load_data(file_config):
                return False

            df = self.add_smiles();             assert df is not None
            df = self.generate_fingerprints(df)
            df = self.filter_fingerprints(df);  assert df is not None
            df = self.create_features(df)

            X, feature_names = self.build_matrix(df)
            X_processed = self.preprocess_features(X)

            self.save_results(X_processed, feature_names, df)
            self.print_summary(len(self.dfs.get('df13', [])), len(self.results['smiles_added']),
                               len(self.results['filtered']), X_processed.shape)
            logger.info("Pipeline completed successfully.")
            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return False


def main():
    file_config = {
        'required': [('df13', 'df13.csv'), ('df2', 'dataset2.csv')],
        'optional': [('df4', 'dataset4.csv')],
    }
    pipeline = FeatureEngineeringPipeline(data_path="data/processed", output_dir="data/features")
    if not pipeline.run(file_config):
        sys.exit(1)


if __name__ == "__main__":
    main()