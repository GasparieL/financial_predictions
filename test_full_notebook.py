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

print("="*80)
print("TESTING FULL NOTEBOOK WORKFLOW")
print("="*80)

# Load datasets
spending_df = pd.read_excel('STD_spending.xlsx')
features_df = pd.read_excel('raw_data_info.xlsx')
print(f"\n✓ Data loaded successfully")

# Filter and combine vendors
spending_df['vendor'] = spending_df['vendor'].replace('Amazon.com. lnc', 'Amazon (ლუქსემბურგი)')
vendor_mapping = {
    'Amazon (ლუქსემბურგი)': 'Amazon',
    'Google Cloud EMEA Limited': 'Google',
    'MongoDB Limited': 'MongoDB'
}
spending_filtered = spending_df[spending_df['vendor'].isin(vendor_mapping.keys())].copy()
spending_filtered['vendor'] = spending_filtered['vendor'].map(vendor_mapping)
print(f"✓ Vendors filtered and mapped")

# Aggregate and pivot
spending_agg = spending_filtered.groupby(['month', 'vendor'], as_index=False)['cost'].sum()
spending_pivot = spending_agg.pivot(index='month', columns='vendor', values='cost').reset_index()
spending_pivot.columns.name = None
spending_pivot = spending_pivot.rename(columns={'month': 'date'})
print(f"✓ Data pivoted successfully: {spending_pivot.shape}")

# Merge
merged_df = pd.merge(spending_pivot, features_df, on='date', how='inner')
print(f"✓ Data merged: {merged_df.shape}")

# Time features
def create_time_features(df):
    df = df.copy()
    df['month_num'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['days_in_month'] = df['date'].dt.days_in_month
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    return df

merged_df = create_time_features(merged_df)
print(f"✓ Time features created: {merged_df.shape}")

# Lag features
def create_lag_features(df, target_cols, lags=[1, 2, 3]):
    df = df.copy()
    df = df.sort_values('date')

    for col in target_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        df[f'{col}_rolling_mean_3'] = df[col].shift(1).rolling(window=3, min_periods=1).mean()
        df[f'{col}_rolling_mean_6'] = df[col].shift(1).rolling(window=6, min_periods=1).mean()
        df[f'{col}_rolling_std_3'] = df[col].shift(1).rolling(window=3, min_periods=1).std()

    return df

target_vendors = ['Amazon', 'Google', 'MongoDB']
merged_df = create_lag_features(merged_df, target_vendors, lags=[1, 2, 3])
print(f"✓ Lag features created: {merged_df.shape}")

# Fill missing values
feature_cols = [col for col in merged_df.columns if col not in target_vendors + ['date']]
for col in feature_cols:
    if merged_df[col].isna().sum() > 0:
        merged_df[col] = merged_df[col].ffill().bfill()

merged_df = merged_df.dropna(subset=target_vendors)
print(f"✓ Missing values handled: {merged_df.shape}")

# Feature selection
def select_features_for_vendor(df, vendor, top_n=15):
    feature_cols = [col for col in df.columns if col not in target_vendors + ['date']]
    X = df[feature_cols]
    y = df[vendor]

    correlations = X.corrwith(y).abs().sort_values(ascending=False)

    selected_features = []
    for feature in correlations.index:
        if len(selected_features) >= top_n:
            break
        if len(selected_features) == 0:
            selected_features.append(feature)
        else:
            max_corr = X[selected_features + [feature]].corr()[feature][selected_features].abs().max()
            if max_corr < 0.9:
                selected_features.append(feature)

    return selected_features, correlations.head(top_n)

selected_features_dict = {}
for vendor in target_vendors:
    features, importances = select_features_for_vendor(merged_df, vendor, top_n=15)
    selected_features_dict[vendor] = features

print(f"✓ Features selected for all vendors")

# Data splits
merged_df_sorted = merged_df.sort_values('date').reset_index(drop=True)
n = len(merged_df_sorted)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

train_df = merged_df_sorted.iloc[:train_size]
val_df = merged_df_sorted.iloc[train_size:train_size + val_size]
test_df = merged_df_sorted.iloc[train_size + val_size:]

print(f"✓ Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Test training on one vendor
print(f"\n✓ Testing model training on Amazon...")
vendor = 'Amazon'
features = selected_features_dict[vendor]

X_train = train_df[features]
y_train = train_df[vendor]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Ridge(alpha=10.0, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_train_scaled)
mae = mean_absolute_error(y_train, y_pred)
print(f"✓ Model trained successfully - Train MAE: ${mae:.2f}")

print("\n" + "="*80)
print("ALL TESTS PASSED - NOTEBOOK IS READY TO RUN")
print("="*80)
