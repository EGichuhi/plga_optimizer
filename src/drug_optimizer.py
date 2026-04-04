import pandas as pd
import numpy as np
from itertools import product
import joblib
import os

class PLGAOptimizer:
    """Optimize PLGA formulations for drug delivery"""
    
    def __init__(self, models_path='models/', data_path='data/processed/'):
        self.models_path = models_path
        self.data_path = data_path
        
        # Load models
        self.models = {}
        for target in ['particle_size', 'EE', 'LC']:
            model_path = f'{models_path}/{target}_model.pkl'
            if os.path.exists(model_path):
                self.models[target] = joblib.load(model_path)
                print(f" Loaded {target} model")
            else:
                print(f" Model not found: {model_path}")
        
        # Load reference data
        self.df13 = pd.read_csv(f'{data_path}/df13_with_features.csv')
        self.df2 = pd.read_csv(f'{data_path}/dataset2.csv')
        
        # Define numeric features (exclude string columns)
        self.numeric_features = [
            'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 
            'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 
            'mol_heteroatoms', 'drug/polymer', 'surfactant_concentration', 
            'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index',
            'LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa', 
            'drug_logP_normalized', 'polymer_drug_affinity', 'la_ga_drug_logP'
        ]
        
        # Keep only features that exist in df13
        self.feature_names = [f for f in self.numeric_features if f in self.df13.columns]
        
        print(f"\n Optimizer ready with {len(self.feature_names)} features")
    
    def get_drug_properties(self, drug_name):
        """Retrieve or estimate drug properties from numeric columns only"""
        # Try to find in df13
        drug_data = self.df13[self.df13['small_molecule_name'] == drug_name]
        
        if len(drug_data) > 0:
            # Use actual properties from dataset (numeric only)
            props = {}
            for col in self.numeric_features:
                if col in drug_data.columns:
                    props[col] = drug_data[col].mean()
            print(f" Found {drug_name} in database")
            return props
        else:
            # Use average values from dataset
            print(f" Drug '{drug_name}' not in database. Using average properties.")
            props = {}
            for col in self.numeric_features:
                if col in self.df13.columns:
                    props[col] = self.df13[col].mean()
            return props
    
    def recommend_plga(self, drug_name, min_ee=60, priority='balanced'):
        """Recommend optimal PLGA formulation for a drug"""
        
        print(f"\n{'='*70}")
        print(f" OPTIMIZING PLGA FOR: {drug_name.upper()}")
        print('='*70)
        
        # Get drug properties
        drug_props = self.get_drug_properties(drug_name)
        
        # Define search space
        search_space = {
            'polymer_MW': [10000, 20000, 30000, 50000, 75000, 100000],
            'LA/GA': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            'drug/polymer': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        }
        
        total_combos = len(search_space['polymer_MW']) * len(search_space['LA/GA']) * len(search_space['drug/polymer'])
        print(f"\n Searching {total_combos} combinations...")
        
        results = []
        
        # Grid search
        for mw, la_ga, dp in product(
            search_space['polymer_MW'],
            search_space['LA/GA'],
            search_space['drug/polymer']
        ):
            # Build feature vector
            features = self._build_features(mw, la_ga, dp, drug_props)
            
            # Predict
            size = self.models['particle_size'].predict(features)[0]
            ee = self.models['EE'].predict(features)[0]
            lc = self.models['LC'].predict(features)[0]
            
            # Filter by constraints
            if ee >= min_ee:
                # Calculate score
                if priority == 'size':
                    score = 1000 / (size + 1)
                elif priority == 'ee':
                    score = ee
                elif priority == 'lc':
                    score = lc
                else:  # balanced
                    size_score = (1000/(size+1)) / 10
                    ee_score = ee / 100
                    lc_score = lc / 30
                    score = (size_score * 0.4 + ee_score * 0.3 + lc_score * 0.3) * 100
                
                results.append({
                    'polymer_MW (Da)': mw,
                    'LA/GA_ratio': la_ga,
                    'drug/polymer_ratio': round(dp, 3),
                    'pred_size_nm': round(size, 1),
                    'pred_EE_%': round(ee, 1),
                    'pred_LC_%': round(lc, 1),
                    'score': round(score, 2)
                })
        
        if len(results) == 0:
            print(f"\n No formulations found with EE >= {min_ee}%")
            return pd.DataFrame()
        
        # Sort and display
        results_df = pd.DataFrame(results).sort_values('score', ascending=False)
        
        print(f"\n{'='*70}")
        print(f" TOP 5 RECOMMENDATIONS")
        print('='*70)
        print(results_df.head(5).to_string(index=False))
        
        best = results_df.iloc[0]
        print(f"\n{'='*70}")
        print(f" RECOMMENDED FORMULATION")
        print('='*70)
        print(f"   Polymer MW:      {best['polymer_MW (Da)']:,} Da")
        print(f"   LA/GA ratio:     {best['LA/GA_ratio']}")
        print(f"   Drug loading:    {best['drug/polymer_ratio']}")
        print(f"\n Expected performance:")
        print(f"   • Particle size: {best['pred_size_nm']} nm")
        print(f"   • Encapsulation: {best['pred_EE_%']}%")
        print(f"   • Loading capacity: {best['pred_LC_%']}%")
        
        return results_df
    
    def _build_features(self, polymer_MW, LA_GA, drug_polymer, drug_props):
        """Build feature vector matching training data"""
        
        # Calculate engineered features
        LA_mol_fraction = LA_GA / (1 + LA_GA)
        log_polymer_MW = np.log10(polymer_MW)
        polymer_MW_kDa = polymer_MW / 1000
        
        # Get drug_logP_normalized
        mol_logP = drug_props.get('mol_logP', 2.0)
        drug_logP_normalized = (mol_logP - self.df13['mol_logP'].mean()) / self.df13['mol_logP'].std()
        
        polymer_drug_affinity = log_polymer_MW * drug_logP_normalized
        la_ga_drug_logP = LA_mol_fraction * drug_logP_normalized
        
        # Build feature vector in correct order
        feature_values = [
            float(polymer_MW),  # polymer_MW
            float(LA_GA),  # LA/GA
            float(drug_props.get('mol_MW', 300)),  # mol_MW
            float(drug_props.get('mol_logP', 2.0)),  # mol_logP
            float(drug_props.get('mol_TPSA', 60)),  # mol_TPSA
            float(drug_props.get('mol_melting_point', 150)),  # mol_melting_point
            float(drug_props.get('mol_Hacceptors', 4)),  # mol_Hacceptors
            float(drug_props.get('mol_Hdonors', 2)),  # mol_Hdonors
            float(drug_props.get('mol_heteroatoms', 6)),  # mol_heteroatoms
            float(drug_polymer),  # drug/polymer
            1.5,  # surfactant_concentration
            15.0,  # surfactant_HLB
            1.0,  # aqueous/organic
            7.0,  # pH
            0.8,  # solvent_polarity_index
            LA_mol_fraction,  # LA_mol_fraction
            log_polymer_MW,  # log_polymer_MW
            polymer_MW_kDa,  # polymer_MW_kDa
            drug_logP_normalized,  # drug_logP_normalized
            polymer_drug_affinity,  # polymer_drug_affinity
            la_ga_drug_logP  # la_ga_drug_logP
        ]
        
        return np.array(feature_values).reshape(1, -1)


# Quick test
if __name__ == "__main__":
    optimizer = PLGAOptimizer()
    results = optimizer.recommend_plga('sparfloxacin', min_ee=60)