import os
import pandas as pd

print("="*50)
print("Checking Data Files")
print("="*50)

data_path = 'data/processed/'
files = ['dataset1.csv', 'dataset2.csv', 'dataset3.csv', 'dataset4.csv']

print(f"\nLooking for files in: {data_path}")
print("-" * 40)

for file in files:
    full_path = os.path.join(data_path, file)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        print(f" {file} ({size:,} bytes)")
    else:
        print(f" {file} - NOT FOUND")
        print(f"  Tried: {os.path.abspath(full_path)}")

print("-" * 40)

# Try loading one file to verify it's readable
try:
    test_df = pd.read_csv(os.path.join(data_path, 'dataset1.csv'))
    print(f"\n Successfully loaded dataset1.csv")
    print(f"  Shape: {test_df.shape}")
    print(f"  Columns: {test_df.columns.tolist()[:5]}...")
except Exception as e:
    print(f"\n Error loading dataset1.csv: {e}")

print("\n" + "="*50)
