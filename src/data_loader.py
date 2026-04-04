# src/data_loader.py

import pandas as pd
import os

class DataLoader:
    """Single source for all data loading and preprocessing"""
    
    def __init__(self, data_path='data/processed/'):
        self.data_path = data_path
        # Ensure directory exists
        os.makedirs(data_path, exist_ok=True)
    
    def _load_csv(self, filename):
        """Internal method to load CSV files"""
        path = os.path.join(self.data_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path)
    
    def _save_csv(self, df, filename):
        """Internal method to save CSV files"""
        path = os.path.join(self.data_path, filename)
        df.to_csv(path, index=False)
        print(f"   ✓ Saved {filename}")
    
    def _file_exists(self, filename):
        """Check if a file exists"""
        return os.path.exists(os.path.join(self.data_path, filename))
    
    def load_dataset1(self):
        return self._load_csv('dataset1.csv')
    
    def load_dataset2(self):
        return self._load_csv('dataset2.csv')
    
    def load_dataset3(self):
        return self._load_csv('dataset3.csv')
    
    def load_dataset4(self):
        return self._load_csv('dataset4.csv')
    
    def build_df13(self):
        """Step 1: Concatenate D1 and D3, save as df13.csv"""
        print("   Building df13.csv (D1 + D3 concat)...")
        
        df1 = self.load_dataset1()
        df3 = self.load_dataset3()
        
        if len(df1) != len(df3):
            print(f"   Warning: D1 has {len(df1)} rows, D3 has {len(df3)} rows")
        
        df13 = pd.concat([df1.reset_index(drop=True), 
                          df3.reset_index(drop=True)], 
                         axis=1)
        df13 = df13.loc[:, ~df13.columns.duplicated(keep='first')]
        
        self._save_csv(df13, 'df13.csv')
        return df13
    
    def add_smiles(self, df13):
        """Add SMILES to df13 using df2 as primary, df4 as fallback"""
        df2 = self.load_dataset2()
        df4 = self.load_dataset4()
        
        smiles_lookup = {}
        
        # Primary: df2
        for _, row in df2.iterrows():
            if pd.notna(row.get('small_molecule_name')) and pd.notna(row.get('canonical_SMILES')):
                smiles_lookup[row['small_molecule_name']] = row['canonical_SMILES']
        
        # Fallback: df4
        if 'canonical_SMILES' in df4.columns:
            for _, row in df4.iterrows():
                name = row.get('small_molecule_name')
                smiles = row.get('canonical_SMILES')
                if pd.notna(name) and pd.notna(smiles) and name not in smiles_lookup:
                    smiles_lookup[name] = smiles
        
        df13_with_smiles = df13.copy()
        df13_with_smiles['canonical_SMILES'] = df13_with_smiles['small_molecule_name'].map(smiles_lookup)
        
        matched = df13_with_smiles['canonical_SMILES'].notna().sum()
        print(f"   Added SMILES to {matched}/{len(df13_with_smiles)} rows ({matched/len(df13_with_smiles)*100:.1f}%)")
        
        return df13_with_smiles
    
    def build_df13_with_smiles(self):
        """Step 2: Add SMILES to df13, save as df13_with_smiles.csv"""
        print("   Building df13_with_smiles.csv...")
        
        df13 = self.load_df13()
        df13_with_smiles = self.add_smiles(df13)
        self._save_csv(df13_with_smiles, 'df13_with_smiles.csv')
        
        return df13_with_smiles
    
    def load_df13(self):
        """Load the base df13 (D1 + D3 concat)"""
        if not self._file_exists('df13.csv'):
            return self.build_df13()
        return self._load_csv('df13.csv')
    
    def load_df13_with_smiles(self):
        """Load df13 with SMILES added"""
        if not self._file_exists('df13_with_smiles.csv'):
            return self.build_df13_with_smiles()
        return self._load_csv('df13_with_smiles.csv')
    
    def load_df13_with_features(self):
        """Load the final fully-processed dataset"""
        if not self._file_exists('df13_with_features.csv'):
            return None
        return self._load_csv('df13_with_features.csv')
    
    def prepare_full_dataset(self):
        """Complete pipeline: ensure df13_with_smiles exists"""
        print("   Checking/preparing data...")
        
        # This will automatically build if missing
        df13 = self.load_df13_with_smiles()
        print(f"   Dataset ready: {df13.shape}")
        
        return df13