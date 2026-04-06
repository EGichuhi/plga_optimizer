import joblib
import os
import logging
from datetime import datetime

from src.config import MODEL_CONFIG
from src.data_loader import DataLoader
from src.train_test import split_data, build_feature_matrices, preprocess, train_and_evaluate
from src.feature_engineering import ALL_SCALAR_FEATURES, run_feature_engineering


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_dir='logs/'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(log_dir, f'run_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),           # also prints to console
        ]
    )
    logging.info(f"Log file: {log_path}")
    return log_path


# ── Save ──────────────────────────────────────────────────────────────────────

def save_pipeline(pipeline, preprocessor, available_targets, model_path='models/'):
    logging.info("=" * 70)
    logging.info("STEP 5: SAVING")
    logging.info("=" * 70)

    pipeline.save_models(preprocess_pipe=preprocessor, path=model_path)

    metadata = {
        'scalar_features': ALL_SCALAR_FEATURES,
        'targets':         available_targets,
        'config':          MODEL_CONFIG,
    }
    meta_out = os.path.join(model_path, 'feature_metadata.pkl')
    joblib.dump(metadata, meta_out)
    logging.info(f"Saved metadata → {meta_out}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(pipeline, X_train, X_test, train_df, test_df, available_targets):
    logging.info("=" * 70)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 70)

    logging.info(f"Samples  — train: {X_train.shape[0]}  test: {X_test.shape[0]}")
    logging.info(f"Features after preprocessing: {X_train.shape[1]}")

    logging.info("Target statistics:")
    for t in available_targets:
        logging.info(f"  {t.upper()}")
        logging.info(f"    Train  mean={train_df[t].mean():.2f}  std={train_df[t].std():.2f}")
        logging.info(f"    Test   mean={test_df[t].mean():.2f}  std={test_df[t].std():.2f}")

    logging.info("Model performance:")
    for t in available_targets:
        res = pipeline.results.get(t, {})
        logging.info(f"  {t.upper()}")
        if 'cv_r2_mean' in res:
            logging.info(f"    CV  R²  : {res['cv_r2_mean']:.3f} ± {res.get('cv_r2_std', 0):.3f}")
        if 'cv_mae_mean' in res:
            logging.info(f"    CV  MAE : {res['cv_mae_mean']:.2f} ± {res.get('cv_mae_std', 0):.2f}")
        if 'test_r2' in res:
            logging.info(f"    Test R² : {res['test_r2']:.3f}")
        if 'test_mae' in res:
            logging.info(f"    Test MAE: {res['test_mae']:.2f}")

    logging.info("=" * 70)
    logging.info("PIPELINE RUN COMPLETE")
    logging.info("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_path = setup_logging()

    logging.info("=" * 70)
    logging.info("MLPIPELINE — PLGA FORMULATION PREDICTION")
    logging.info("=" * 70)
    logging.info(f"Config: {MODEL_CONFIG}")

    try:
        logging.info("Step 1: Loading data")
        df = DataLoader().prepare_full_dataset()
        logging.info(f"  Dataset shape: {df.shape}")

        logging.info("Step 2: Feature engineering")
        df = run_feature_engineering(df)
        logging.info(f"  Dataset shape after engineering: {df.shape}")

        logging.info("Step 3: Train/test split")
        train_df, test_df, available_targets = split_data(df)
        logging.info(f"  Targets: {available_targets}")

        logging.info("Step 4: Building feature matrices")
        X_train, X_test, feat_names, n_scalar = build_feature_matrices(train_df, test_df)

        logging.info("Step 5: Preprocessing")
        X_train_p, X_test_p, preproc, final_fn = preprocess(X_train, X_test, n_scalar)

        logging.info("Step 6: Training and evaluation")
        pipeline = train_and_evaluate(
                        X_train_p, X_test_p,
                        train_df, test_df,
                        final_fn, available_targets)

        save_pipeline(pipeline, preproc, available_targets)
        print_summary(pipeline, X_train_p, X_test_p, train_df, test_df, available_targets)

        logging.info(f"Run complete. Log saved to {log_path}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        logging.error("Check that data/processed/ contains dataset1.csv–dataset4.csv")
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")