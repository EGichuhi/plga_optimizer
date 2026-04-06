import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

RDLogger.DisableLog('rdApp.warning')

POLYMER_FEATURES    = ['polymer_MW', 'LA/GA']
DRUG_FEATURES       = ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point',
                        'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms']
FORMULATION_FEATURES = ['drug/polymer', 'surfactant_concentration', 'surfactant_HLB']
PROCESS_FEATURES    = ['aqueous/organic', 'pH', 'solvent_polarity_index']
ENGINEERED_FEATURES = ['LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa',
                        'polymer_drug_MW_ratio', 'la_ga_drug_logP']

ALL_SCALAR_FEATURES = (POLYMER_FEATURES + DRUG_FEATURES + FORMULATION_FEATURES
                       + PROCESS_FEATURES + ENGINEERED_FEATURES)


# SMILES 

def add_smiles_from_lookup(df13, df2, df4=None):
    lookup = {}
    for _, row in df2.iterrows():
        if pd.notna(row.get('small_molecule_name')) and pd.notna(row.get('canonical_SMILES')):
            lookup[row['small_molecule_name']] = row['canonical_SMILES']

    if df4 is not None and 'canonical_SMILES' in df4.columns:
        for _, row in df4.iterrows():
            name, smiles = row.get('small_molecule_name'), row.get('canonical_SMILES')
            if pd.notna(name) and pd.notna(smiles) and name not in lookup:
                lookup[name] = smiles

    out = df13.copy()
    out['canonical_SMILES'] = out['small_molecule_name'].map(lookup)
    matched = out['canonical_SMILES'].notna().sum()
    print(f"SMILES matched: {matched}/{len(out)} ({matched/len(out)*100:.1f}%)")
    return out


# Morgan fingerprints 

def _safe_get_fingerprint(smiles, radius=2, nBits=2048):
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return list(rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits).GetFingerprint(mol))
    except Exception:
        print(f"  WARNING: fingerprint failed for '{smiles}'")
        return None


def generate_morgan_fingerprints(df, smiles_col='canonical_SMILES', radius=2, nBits=2048):
    out = df.copy()
    out['morgan_fingerprint'] = out[smiles_col].apply(lambda x: _safe_get_fingerprint(x, radius, nBits))
    valid = out['morgan_fingerprint'].notna().sum()
    if valid:
        sample = out['morgan_fingerprint'].dropna().iloc[0]
    return out


def filter_valid_fingerprints(df, fp_col='morgan_fingerprint'):
    out = df[df[fp_col].notna()].copy().reset_index(drop=True)
    dropped = len(df) - len(out)
    if dropped:
        print(f"Dropped {dropped} rows with missing fingerprints.")
    return out


# Feature engineering 

def create_engineered_features(df):
    out = df.copy()
    out['LA_mol_fraction']      = out['LA/GA'] / (1 + out['LA/GA'])
    out['log_polymer_MW']       = np.log10(out['polymer_MW'])
    out['polymer_MW_kDa']       = out['polymer_MW'] / 1000
    out['polymer_drug_MW_ratio'] = out['polymer_MW'] / out['mol_MW'].replace(0, np.nan)
    out['la_ga_drug_logP']      = out['LA_mol_fraction'] * out['mol_logP']
    return out


# Feature matrix 

def build_full_feature_matrix(df, scalar_features=ALL_SCALAR_FEATURES, fp_col='morgan_fingerprint'):
    fp_matrix     = np.array(df[fp_col].tolist(), dtype=np.float32)
    scalar_matrix = df[scalar_features].values.astype(np.float64)
    X             = np.hstack([scalar_matrix, fp_matrix])
    feature_names = list(scalar_features) + [f'fp_bit_{i}' for i in range(fp_matrix.shape[1])]
    return X, df.index, feature_names


# sklearn transformers 

class ScalarImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_scalar_features):
        self.n_scalar_features = n_scalar_features

    def fit(self, X, y=None):
        self._imputer = SimpleImputer(strategy='median')
        self._imputer.fit(X[:, :self.n_scalar_features])
        return self

    def transform(self, X):
        X = X.copy()
        X[:, :self.n_scalar_features] = self._imputer.transform(X[:, :self.n_scalar_features])
        return X


class ScalarScaler(BaseEstimator, TransformerMixin):
    def __init__(self, n_scalar_features):
        self.n_scalar_features = n_scalar_features

    def fit(self, X, y=None):
        self._scaler = StandardScaler()
        self._scaler.fit(X[:, :self.n_scalar_features])
        return self

    def transform(self, X):
        X = X.copy()
        X[:, :self.n_scalar_features] = self._scaler.transform(X[:, :self.n_scalar_features])
        return X


class FingerprintVarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, n_scalar_features, threshold=0.01):
        self.n_scalar_features = n_scalar_features
        self.threshold         = threshold

    def fit(self, X, y=None):
        self._vt = VarianceThreshold(threshold=self.threshold)
        self._vt.fit(X[:, self.n_scalar_features:])
        print(f"VarianceThreshold: {self._vt.get_support().sum()}/{X.shape[1] - self.n_scalar_features} fp bits kept")
        return self

    def transform(self, X):
        return np.hstack([X[:, :self.n_scalar_features],
                          self._vt.transform(X[:, self.n_scalar_features:])])


# Pipeline 

def build_preprocessing_pipeline(n_scalar_features: int, fp_variance_threshold: float = 0.1):
    return ColumnTransformer(
        transformers=[
            ('scalar',      Pipeline([('imputer', SimpleImputer(strategy='median')),
                                       ('scaler',  StandardScaler())]),
                            slice(0, n_scalar_features)),
            ('fingerprint', VarianceThreshold(threshold=fp_variance_threshold),
                            slice(n_scalar_features, None))
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

def run_feature_engineering(df):
    df = generate_morgan_fingerprints(df, smiles_col='canonical_SMILES', radius=2, nBits=2048)
    df = filter_valid_fingerprints(df, fp_col='morgan_fingerprint')
    df = create_engineered_features(df)
    return df

