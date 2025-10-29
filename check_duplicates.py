import pandas as pd

spending_df = pd.read_excel('STD_spending.xlsx')

spending_df['vendor'] = spending_df['vendor'].replace('Amazon.com. lnc', 'Amazon (ლუქსემბურგი)')

vendor_mapping = {
    'Amazon (ლუქსემბურგი)': 'Amazon',
    'Google Cloud EMEA Limited': 'Google',
    'MongoDB Limited': 'MongoDB'
}

spending_filtered = spending_df[spending_df['vendor'].isin(vendor_mapping.keys())].copy()
spending_filtered['vendor'] = spending_filtered['vendor'].map(vendor_mapping)

print("Checking for duplicates:")
duplicates = spending_filtered[spending_filtered.duplicated(subset=['month', 'vendor'], keep=False)]
print(f"\nFound {len(duplicates)} duplicate records:")
print(duplicates.sort_values(['vendor', 'month']))

print("\n" + "="*80)
print("Duplicate counts per vendor-month:")
dup_counts = spending_filtered.groupby(['vendor', 'month']).size()
dup_counts = dup_counts[dup_counts > 1]
print(dup_counts)
