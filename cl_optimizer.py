# cl_optimizer.py - Updated with results folder

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from itertools import product
import joblib
import os
from datetime import datetime

class PLGAOptimizer:
    """Optimize PLGA formulations using Random Forest models"""
    
    def __init__(self, models_path='models/', data_path='data/processed/'):
        self.models_path = models_path
        self.data_path = data_path
        
        # Create results folder if it doesn't exist
        self.results_path = 'results/'
        os.makedirs(self.results_path, exist_ok=True)
        
        # Load Random Forest models
        self.models = {}
        for target in ['particle_size', 'EE', 'LC']:
            model_path = f'{models_path}/{target}_model.pkl'
            if os.path.exists(model_path):
                self.models[target] = joblib.load(model_path)
                print(f"✓ Loaded {target} model (Random Forest)")
            else:
                print(f"✗ Model not found: {model_path}")
                raise FileNotFoundError(f"Train models first with python main.py")
        
        # Load reference data
        self.df13 = pd.read_csv(f'{data_path}/df13_with_features.csv')
        
        # Define numeric features (must match training)
        self.feature_names = [
            'polymer_MW', 'LA/GA', 'mol_MW', 'mol_logP', 'mol_TPSA', 
            'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 
            'mol_heteroatoms', 'drug/polymer', 'surfactant_concentration', 
            'surfactant_HLB', 'aqueous/organic', 'pH', 'solvent_polarity_index',
            'LA_mol_fraction', 'log_polymer_MW', 'polymer_MW_kDa', 
            'drug_logP_normalized', 'polymer_drug_affinity', 'la_ga_drug_logP'
        ]
        
        print(f"✓ Optimizer ready with {len(self.feature_names)} features")
        print(f"✓ Database: {len(self.df13)} formulations, {self.df13['small_molecule_name'].nunique()} drugs")
        print(f"✓ Results will be saved to: {self.results_path}")
    
    def get_drug_properties(self, drug_name):
        """Retrieve drug properties from database or use averages"""
        # Case-insensitive search
        drug_data = self.df13[self.df13['small_molecule_name'].str.lower() == drug_name.lower()]
        
        if len(drug_data) > 0:
            actual_name = drug_data['small_molecule_name'].iloc[0]
            props = {
                'mol_MW': drug_data['mol_MW'].mean(),
                'mol_logP': drug_data['mol_logP'].mean(),
                'mol_TPSA': drug_data['mol_TPSA'].mean(),
                'mol_melting_point': drug_data['mol_melting_point'].mean(),
                'mol_Hacceptors': drug_data['mol_Hacceptors'].mean(),
                'mol_Hdonors': drug_data['mol_Hdonors'].mean(),
                'mol_heteroatoms': drug_data['mol_heteroatoms'].mean(),
            }
            print(f"✓ Found '{actual_name}' in database")
            print(f"   Properties: LogP={props['mol_logP']:.2f}, TPSA={props['mol_TPSA']:.1f} Å², MW={props['mol_MW']:.0f} Da")
            return props
        else:
            print(f" Drug '{drug_name}' not found in database.")
            print(f"   Using average properties from {len(self.df13)} formulations.")
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
        """Build feature vector matching training data"""
        # Engineered features
        LA_mol_fraction = LA_GA / (1 + LA_GA)
        log_polymer_MW = np.log10(polymer_MW)
        polymer_MW_kDa = polymer_MW / 1000
        drug_logP_normalized = (drug_props['mol_logP'] - self.df13['mol_logP'].mean()) / self.df13['mol_logP'].std()
        polymer_drug_affinity = log_polymer_MW * drug_logP_normalized
        la_ga_drug_logP = LA_mol_fraction * drug_logP_normalized
        
        # Create feature vector (order must match training)
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
            1.5,   # surfactant_concentration (default)
            15.0,  # surfactant_HLB (default for Tween 80)
            1.0,   # aqueous/organic (default 1:1)
            7.0,   # pH (default neutral)
            0.8,   # solvent_polarity_index (default water)
            LA_mol_fraction,
            log_polymer_MW,
            polymer_MW_kDa,
            drug_logP_normalized,
            polymer_drug_affinity,
            la_ga_drug_logP
        ]])
        
        return features
    
    def save_results(self, results_df, drug_name, priority, min_ee, max_size):
        """Save results to CSV with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{drug_name.replace(' ', '_')}_{priority}_{timestamp}.csv"
        filepath = os.path.join(self.results_path, filename)
        
        # Add metadata as comments in the file
        with open(filepath, 'w') as f:
            f.write(f"# PLGA Optimization Results\n")
            f.write(f"# Drug: {drug_name}\n")
            f.write(f"# Priority: {priority}\n")
            f.write(f"# Min EE: {min_ee}%\n")
            f.write(f"# Max Size: {max_size} nm\n")
            f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total formulations evaluated: {len(results_df)}\n")
            f.write("#\n")
        
        # Append the dataframe
        results_df.to_csv(filepath, mode='a', index=False)
        
        print(f"\n Results saved to: {filepath}")
        return filepath
    
    def recommend(self, drug_name, min_ee=30, max_size=500, priority='balanced', show_top=10, auto_save=True):
        """Recommend optimal PLGA formulation"""
        
        print(f"\n{'='*60}")
        print(f" Optimizing PLGA for: {drug_name.upper()}")
        print(f"{'='*60}")
        
        # Get drug properties
        drug_props = self.get_drug_properties(drug_name)
        
        # Define search space
        # OPTIMAL for 488 training samples

        search_space = {
            'polymer_MW': [10000, 20000, 30000, 50000, 75000, 100000, 150000],
            'LA/GA': [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            'drug/polymer': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        }
        
        total = len(search_space['polymer_MW']) * len(search_space['LA/GA']) * len(search_space['drug/polymer'])
        print(f" Searching {total} formulations...")
        print(f"   Constraints: EE ≥ {min_ee}%, Size ≤ {max_size} nm")
        print(f"   Priority: {priority.upper()}")
        
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
            
            if ee >= min_ee and size <= max_size:
                # Calculate score based on priority
                size_score = max(0, min(1, (500 - size) / 500))
                ee_score = ee / 100
                lc_score = lc / 30
                
                if priority == 'size':
                    score = size_score * 100
                elif priority == 'ee':
                    score = ee_score * 100
                elif priority == 'lc':
                    score = lc_score * 100
                else:  # balanced
                    score = (size_score * 0.4 + ee_score * 0.3 + lc_score * 0.3) * 100
                
                results.append({
                    'polymer_MW (Da)': mw,
                    'LA/GA ratio': la_ga,
                    'drug/polymer': round(dp, 3),
                    'Size (nm)': round(size, 1),
                    'EE (%)': round(ee, 1),
                    'LC (%)': round(lc, 1),
                    'Score': round(score, 2)
                })
        
        if not results:
            print(f"\n❌ No formulations found matching your criteria.")
            print(f"   Try lowering min_EE (currently {min_ee}%)")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results).sort_values('Score', ascending=False)
        
        # Display results
        print(f"\n{'='*60}")
        print(f" TOP {min(show_top, len(results_df))} RECOMMENDATIONS")
        print(f"{'='*60}")
        print(results_df.head(show_top).to_string(index=False))
        
        # Best formulation
        best = results_df.iloc[0]
        print(f"\n{'='*60}")
        print(f"RECOMMENDED FORMULATION")
        print(f"{'='*60}")
        print(f"   Polymer MW:      {best['polymer_MW (Da)']:,} Da")
        print(f"   LA/GA ratio:     {best['LA/GA ratio']}")
        print(f"   Drug loading:    {best['drug/polymer']}")
        print(f"\n   Expected Performance:")
        print(f"   • Particle size:  {best['Size (nm)']} nm")
        print(f"   • Encapsulation:  {best['EE (%)']}%")
        print(f"   • Loading capacity: {best['LC (%)']}%")
        
        # Interpretation
        la_ga = best['LA/GA ratio']
        if la_ga < 1:
            print(f"\n    LA/GA={la_ga} → Fast degrading (weeks), good for short-term release")
        elif la_ga < 2:
            print(f"\n    LA/GA={la_ga} → Medium degradation (months), balanced release")
        else:
            print(f"\n    LA/GA={la_ga} → Slow degrading (months-years), ideal for long-term release")
        
        # Auto-save results
        if auto_save:
            saved_path = self.save_results(results_df, drug_name, priority, min_ee, max_size)
        
        return results_df


def list_available_drugs():
    """Show all drugs in the database"""
    df13 = pd.read_csv('data/processed/df13_with_features.csv')
    drugs = sorted(df13['small_molecule_name'].unique())
    print("\n Available drugs in database:")
    
    # Display in columns for better readability
    col_width = 25
    cols = 3
    for i in range(0, len(drugs), cols):
        row = drugs[i:i+cols]
        print("   " + "".join([f"{drug:<{col_width}}" for drug in row]))
    
    return drugs


def print_banner():
    print("="*70)
    print(" PLGA DRUG DELIVERY OPTIMIZER")
    print("="*70)
    print("Random Forest Model | Optimize polymer characteristics")
    print("="*70)


def get_user_preferences():
    """Get optimization preferences from user"""
    print("\n Optimization Priority:")
    print("   1. Balanced (default) - Good size, EE, and LC")
    print("   2. Minimize particle size - For targeted delivery")
    print("   3. Maximize encapsulation - For drug loading efficiency")
    print("   4. Maximize loading capacity - For high drug dose")
    
    choice = input("\nChoose (1-4) [default: 1]: ").strip()
    
    priorities = {
        '1': 'balanced',
        '2': 'size',
        '3': 'ee',
        '4': 'lc'
    }
    
    return priorities.get(choice, 'balanced')


def main():
    print_banner()
    
    # Initialize optimizer
    print("\n Loading Random Forest models...")
    try:
        optimizer = PLGAOptimizer()
    except FileNotFoundError:
        print("\n Models not found! Please run 'python main.py' first to train models.")
        return
    
    while True:
        print("\n" + "="*70)
        drug_name = input(" Enter drug name (or 'list' to see available, 'quit' to exit): ").strip()
        
        if drug_name.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye!")
            break
        
        if drug_name.lower() == 'list':
            list_available_drugs()
            continue
        
        if not drug_name:
            print(" Please enter a drug name")
            continue
        
        # Get preferences
        priority = get_user_preferences()
        
        # Get constraints
        print("\n Constraints (press Enter for defaults):")
        min_ee_input = input("  Minimum EE (%) [default: 30]: ").strip()
        min_ee = float(min_ee_input) if min_ee_input else 30
        
        max_size_input = input("  Maximum size (nm) [default: 500]: ").strip()
        max_size = float(max_size_input) if max_size_input else 500
        
        # Run optimization (auto-save is True by default)
        results = optimizer.recommend(
            drug_name=drug_name,
            min_ee=min_ee,
            max_size=max_size,
            priority=priority,
            show_top=10,
            auto_save=True  # Automatically saves to results/ folder
        )
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()