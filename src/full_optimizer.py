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
                print(f"✓ Loaded {target} model")
        
        # Load reference data
        self.df13 = pd.read_csv(f'{data_path}/df13_with_features.csv')
        
        # Define numeric features
        self.feature_names = [
            'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 
            'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 
            'mol_heteroatoms', 'drug/polymer', 'surfactant_concentration', 
            'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index',
            'LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa', 
            'drug_logP_normalized', 'polymer_drug_affinity', 'la_ga_drug_logP'
        ]
        
        print(f"✓ Optimizer ready with {len(self.feature_names)} features")
    
    def get_drug_properties(self, drug_name):
        """Retrieve drug properties from database or use averages"""
        drug_data = self.df13[self.df13['small_molecule_name'] == drug_name]
        
        if len(drug_data) > 0:
            props = {
                'mol_MW': drug_data['mol_MW'].mean(),
                'mol_logP': drug_data['mol_logP'].mean(),
                'mol_TPSA': drug_data['mol_TPSA'].mean(),
                'mol_melting_point': drug_data['mol_melting_point'].mean(),
                'mol_Hacceptors': drug_data['mol_Hacceptors'].mean(),
                'mol_Hdonors': drug_data['mol_Hdonors'].mean(),
                'mol_heteroatoms': drug_data['mol_heteroatoms'].mean(),
            }
            print(f"✓ Found {drug_name} in database")
            print(f"   Properties: logP={props['mol_logP']:.2f}, TPSA={props['mol_TPSA']:.1f}")
            return props
        else:
            print(f"⚠️ Drug '{drug_name}' not found. Using average properties.")
            return {
                'mol_MW': self.df13['mol_MW'].mean(),
                'mol_logP': self.df13['mol_logP'].mean(),
                'mol_TPSA': self.df13['mol_TPSA'].mean(),
                'mol_melting_point': self.df13['mol_melting_point'].mean(),
                'mol_Hacceptors': self.df13['mol_Hacceptors'].mean(),
                'mol_Hdonors': self.df13['mol_Hdonors'].mean(),
                'mol_heteroatoms': self.df13['mol_heteroatoms'].mean(),
            }
    
    def _build_features(self, polymer_MW, LA_GA, drug_polymer, drug_props):
        """Build feature vector for prediction"""
        # Engineered features
        LA_mol_fraction = LA_GA / (1 + LA_GA)
        log_polymer_MW = np.log10(polymer_MW)
        polymer_MW_kDa = polymer_MW / 1000
        drug_logP_normalized = (drug_props['mol_logP'] - self.df13['mol_logP'].mean()) / self.df13['mol_logP'].std()
        polymer_drug_affinity = log_polymer_MW * drug_logP_normalized
        la_ga_drug_logP = LA_mol_fraction * drug_logP_normalized
        
        # Create feature vector
        features = np.array([[
            float(polymer_MW),
            float(LA_GA),
            float(drug_props['mol_MW']),
            float(drug_props['mol_logP']),
            float(drug_props['mol_TPSA']),
            float(drug_props['mol_melting_point']),
            float(drug_props['mol_Hacceptors']),
            float(drug_props['mol_Hdonors']),
            float(drug_props['mol_heteroatoms']),
            float(drug_polymer),
            1.5,   # surfactant_concentration
            15.0,  # surfactant_HLB
            1.0,   # aqueous/organic
            7.0,   # pH
            0.8,   # solvent_polarity_index
            LA_mol_fraction,
            log_polymer_MW,
            polymer_MW_kDa,
            drug_logP_normalized,
            polymer_drug_affinity,
            la_ga_drug_logP
        ]])
        
        return features
    
    def recommend(self, drug_name, min_ee=30, priority='balanced', verbose=True):
        """Recommend optimal PLGA formulation"""
        
        print(f"\n{'='*60}")
        print(f"Optimizing PLGA for: {drug_name.upper()}")
        print('='*60)
        
        # Get drug properties
        drug_props = self.get_drug_properties(drug_name)
        
        # Expand search space for better exploration
        search_space = {
            'polymer_MW': [10000, 20000, 30000, 50000, 75000, 100000, 150000],
            'LA/GA': [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            'drug/polymer': [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
        }
        
        total = len(search_space['polymer_MW']) * len(search_space['LA/GA']) * len(search_space['drug/polymer'])
        print(f"Searching {total} combinations (min_ee={min_ee}%)...")
        
        results = []
        
        for mw, la_ga, dp in product(
            search_space['polymer_MW'],
            search_space['LA/GA'],
            search_space['drug/polymer']
        ):
            features = self._build_features(mw, la_ga, dp, drug_props)
            
            size = self.models['particle_size'].predict(features)[0]
            ee = self.models['EE'].predict(features)[0]
            lc = self.models['LC'].predict(features)[0]
            
            if ee >= min_ee:
                # Calculate score (normalized)
                size_score = max(0, min(1, (500 - size) / 500))  # Smaller better, cap at 500nm
                ee_score = ee / 100
                lc_score = lc / 30  # LC typically 0-30%
                
                if priority == 'size':
                    score = size_score * 100
                elif priority == 'ee':
                    score = ee_score * 100
                elif priority == 'lc':
                    score = lc_score * 100
                else:  # balanced
                    score = (size_score * 0.4 + ee_score * 0.3 + lc_score * 0.3) * 100
                
                results.append({
                    'polymer_MW': mw,
                    'LA/GA': la_ga,
                    'drug/polymer': round(dp, 3),
                    'size_nm': round(size, 1),
                    'EE_%': round(ee, 1),
                    'LC_%': round(lc, 1),
                    'score': round(score, 2)
                })
        
        if not results:
            print(f"❌ No formulations found with EE ≥ {min_ee}%")
            print("   Try lowering min_ee further (e.g., min_ee=20)")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results).sort_values('score', ascending=False)
        
        # Display results
        print(f"\n{'='*60}")
        print(f"TOP 10 RECOMMENDATIONS (found {len(results)} total)")
        print('='*60)
        print(results_df.head(10).to_string(index=False))
        
        best = results_df.iloc[0]
        print(f"\n{'='*60}")
        print("🏆 RECOMMENDED FORMULATION")
        print('='*60)
        print(f"  Polymer MW:      {best['polymer_MW']:,} Da")
        print(f"  LA/GA ratio:     {best['LA/GA']}")
        print(f"  Drug loading:    {best['drug/polymer']}")
        print(f"\n  Expected performance:")
        print(f"    • Particle size: {best['size_nm']} nm")
        print(f"    • Encapsulation: {best['EE_%']}%")
        print(f"    • Loading cap:   {best['LC_%']}%")
        
        # Add interpretation
        la_ga = best['LA/GA']
        if la_ga < 1:
            print(f"\n  💡 LA/GA={la_ga} → Fast degrading, good for short-term release")
        elif la_ga < 2:
            print(f"\n  💡 LA/GA={la_ga} → Balanced degradation")
        else:
            print(f"\n  💡 LA/GA={la_ga} → Slow degrading, ideal for long-term release")
        
        return results_df


# Test with different drugs
if __name__ == "__main__":
    print("="*70)
    print("🚀 PLGA OPTIMIZER TEST")
    print("="*70)
    
    optimizer = PLGAOptimizer()
    
    # Test 1: Sparfloxacin with lower threshold
    print("\n" + "="*70)
    print("TEST 1: Sparfloxacin (min_ee=25%)")
    print("="*70)
    results1 = optimizer.recommend('sparfloxacin', min_ee=25, priority='balanced')
    
    # Test 2: New drug with default settings
    print("\n" + "="*70)
    print("TEST 2: New Drug (using averages)")
    print("="*70)
    results2 = optimizer.recommend('new_drug', min_ee=30, priority='balanced')
    
    # Test 3: Size-prioritized optimization
    print("\n" + "="*70)
    print("TEST 3: Size-prioritized (smallest particles)")
    print("="*70)
    results3 = optimizer.recommend('new_drug', min_ee=30, priority='size')
    
    # Save results
    if len(results2) > 0:
        results2.to_csv('optimization_results.csv', index=False)
        print("\n✓ Results saved to optimization_results.csv")