import pandas as pd
import sys

print("Attempting to read Excel files...")
print("="*80)

try:
    spending_df = pd.read_excel('STD_spending.xlsx')
    print("\nSTD_SPENDING.XLSX loaded successfully!")
    print(f"Shape: {spending_df.shape}")
    print(f"\nColumns ({len(spending_df.columns)}):")
    for i, col in enumerate(spending_df.columns, 1):
        print(f"  {i}. {col}")
    print(f"\nFirst 3 rows:")
    print(spending_df.head(3))
    print(f"\nData types:")
    print(spending_df.dtypes)
except Exception as e:
    print(f"\nERROR reading STD_spending.xlsx: {str(e)}")
    sys.exit(1)

print("\n" + "="*80)

try:
    features_df = pd.read_excel('raw_data_info.xlsx')
    print("\nRAW_DATA_INFO.XLSX loaded successfully!")
    print(f"Shape: {features_df.shape}")
    print(f"\nColumns ({len(features_df.columns)}):")
    for i, col in enumerate(features_df.columns, 1):
        print(f"  {i}. {col}")
    print(f"\nFirst 3 rows:")
    print(features_df.head(3))
    print(f"\nData types:")
    print(features_df.dtypes)
except Exception as e:
    print(f"\nERROR reading raw_data_info.xlsx: {str(e)}")
    sys.exit(1)

print("\n" + "="*80)
print("\nSearching for vendor columns...")
vendors = ['amazon', 'google', 'mongodb']
for vendor in vendors:
    cols = [col for col in spending_df.columns if vendor.lower() in col.lower()]
    print(f"{vendor.upper()}: {cols if cols else 'NOT FOUND'}")
