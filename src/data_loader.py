from pathlib import Path
import pandas as pd
from src.feature_engineering import add_smiles_from_lookup


class DataLoader:
    def __init__(self, data_path='data/processed/'):
        self.data_dir = Path(data_path)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_path} does not exist")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, filename):
        return self.data_dir / filename

    def _load_csv(self, filename):
        path = self._path(filename)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def _save_csv(self, df, filename):
        df.to_csv(self._path(filename), index=False)
        print(f"   ✓ Saved {filename} ({len(df)} rows × {len(df.columns)} cols)")

    def load_dataset(self, index):
        return self._load_csv(f'dataset{index}.csv')

    def build_df13(self):
        df13 = pd.concat([self.load_dataset(1), self.load_dataset(3)], ignore_index=True)
        self._save_csv(df13, 'df13.csv')
        return df13

    def build_df13_with_smiles(self):
        df13 = self.load_df13()
        df13_with_smiles = add_smiles_from_lookup(
            df13,
            self.load_dataset(2),
            self.load_dataset(4)
        )
        self._save_csv(df13_with_smiles, 'df13_with_smiles.csv')
        return df13_with_smiles

    def load_df13(self):
        try:
            return self._load_csv('df13.csv')
        except FileNotFoundError:
            return self.build_df13()

    def load_df13_with_smiles(self):
        try:
            return self._load_csv('df13_with_smiles.csv')
        except FileNotFoundError:
            return self.build_df13_with_smiles()

    def load_df13_with_features(self):
        return self._load_csv('df13_with_features.csv')

    def prepare_full_dataset(self):
        return self.load_df13_with_smiles()