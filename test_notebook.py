import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

print("Testing Cell 3: Load datasets")
try:
    spending_df = pd.read_excel('STD_spending.xlsx')
    features_df = pd.read_excel('raw_data_info.xlsx')
    print(f"SUCCESS: Spending data shape: {spending_df.shape}")
    print(f"SUCCESS: Features data shape: {features_df.shape}")
    print(f"Spending date range: {spending_df['month'].min()} to {spending_df['month'].max()}")
    print(f"Features date range: {features_df['date'].min()} to {features_df['date'].max()}")
except Exception as e:
    print(f"ERROR in Cell 3: {e}")
    exit(1)

print("\n" + "="*80)
print("Testing Cell 5: Combine Amazon vendors and filter")
try:
    spending_df['vendor'] = spending_df['vendor'].replace('Amazon.com. lnc', 'Amazon (ლუქსემბურგი)')

    vendor_mapping = {
        'Amazon (ლუქსემბურგი)': 'Amazon',
        'Google Cloud EMEA Limited': 'Google',
        'MongoDB Limited': 'MongoDB'
    }

    spending_filtered = spending_df[spending_df['vendor'].isin(vendor_mapping.keys())].copy()
    spending_filtered['vendor'] = spending_filtered['vendor'].map(vendor_mapping)

    print(f"SUCCESS: Filtered spending data shape: {spending_filtered.shape}")
    print(f"Records per vendor:")
    print(spending_filtered['vendor'].value_counts())
except Exception as e:
    print(f"ERROR in Cell 5: {e}")
    exit(1)

print("\n" + "="*80)
print("Testing Cell 6: Pivot spending data")
try:
    # Aggregate duplicate vendor-month entries by summing costs
    spending_agg = spending_filtered.groupby(['month', 'vendor'], as_index=False)['cost'].sum()
    spending_pivot = spending_agg.pivot(index='month', columns='vendor', values='cost').reset_index()
    spending_pivot.columns.name = None
    spending_pivot = spending_pivot.rename(columns={'month': 'date'})

    print(f"SUCCESS: Pivoted spending shape: {spending_pivot.shape}")
    print(f"Columns: {spending_pivot.columns.tolist()}")
    print(f"Missing values per vendor:")
    print(spending_pivot[['Amazon', 'Google', 'MongoDB']].isna().sum())
except Exception as e:
    print(f"ERROR in Cell 6: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("Testing Cell 7: Merge spending with features")
try:
    merged_df = pd.merge(spending_pivot, features_df, on='date', how='inner')

    print(f"SUCCESS: Merged data shape: {merged_df.shape}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Missing values by column:")
    print(merged_df.isna().sum())
except Exception as e:
    print(f"ERROR in Cell 7: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("All critical cells tested successfully!")
