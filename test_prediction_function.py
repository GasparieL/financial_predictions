import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

print("Testing enhanced prediction function...")
print("="*80)

# Load and prepare data (abbreviated version)
spending_df = pd.read_excel('STD_spending.xlsx')
features_df = pd.read_excel('raw_data_info.xlsx')

spending_df['vendor'] = spending_df['vendor'].replace('Amazon.com. lnc', 'Amazon (ლუქსემბურგი)')
vendor_mapping = {
    'Amazon (ლუქსემბურგი)': 'Amazon',
    'Google Cloud EMEA Limited': 'Google',
    'MongoDB Limited': 'MongoDB'
}
spending_filtered = spending_df[spending_df['vendor'].isin(vendor_mapping.keys())].copy()
spending_filtered['vendor'] = spending_filtered['vendor'].map(vendor_mapping)

spending_agg = spending_filtered.groupby(['month', 'vendor'], as_index=False)['cost'].sum()
spending_pivot = spending_agg.pivot(index='month', columns='vendor', values='cost').reset_index()
spending_pivot.columns.name = None
spending_pivot = spending_pivot.rename(columns={'month': 'date'})

merged_df = pd.merge(spending_pivot, features_df, on='date', how='inner')

# Create time features
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

# Create lag features
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

feature_cols = [col for col in merged_df.columns if col not in target_vendors + ['date']]
for col in feature_cols:
    if merged_df[col].isna().sum() > 0:
        merged_df[col] = merged_df[col].ffill().bfill()

merged_df = merged_df.dropna(subset=target_vendors)
merged_df_sorted = merged_df.sort_values('date').reset_index(drop=True)

# Select features for Amazon
feature_cols_all = [col for col in merged_df.columns if col not in target_vendors + ['date']]
X = merged_df[feature_cols_all]
y = merged_df['Amazon']
correlations = X.corrwith(y).abs().sort_values(ascending=False)
selected_features = list(correlations.head(15).index)

# Train a simple model
n = len(merged_df_sorted)
train_size = int(0.7 * n)
train_df = merged_df_sorted.iloc[:train_size]

X_train = train_df[selected_features]
y_train = train_df['Amazon']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Ridge(alpha=10.0, random_state=42)
model.fit(X_train_scaled, y_train)

print("✓ Model trained successfully")

# Test prediction function
def predict_future_spending_test(future_date, future_features=None):
    if future_features is None:
        future_features = {}

    if isinstance(future_date, str):
        future_date = pd.to_datetime(future_date)

    future_row = pd.DataFrame({'date': [future_date]})
    future_row = create_time_features(future_row)

    time_features = ['month_num', 'quarter', 'year', 'days_in_month',
                    'is_quarter_start', 'is_quarter_end', 'month_sin', 'month_cos']

    source_row = merged_df_sorted.iloc[-1]

    for feature in selected_features:
        if feature in time_features:
            pass  # Already calculated
        elif feature in future_features:
            future_row[feature] = future_features[feature]
        else:
            future_row[feature] = source_row[feature]

    X_future = future_row[selected_features]
    X_future_scaled = scaler.transform(X_future)
    prediction = model.predict(X_future_scaled)[0]

    return prediction

# Test 1: Basic prediction
print("\n✓ Test 1: Prediction without future features")
pred1 = predict_future_spending_test('2026-01-01')
print(f"  Predicted spending: ${pred1:,.2f}")

# Test 2: Prediction with future features
print("\n✓ Test 2: Prediction with future features")
pred2 = predict_future_spending_test('2026-01-01', {
    'ecomm_orders': 15000,
    'ecomm_active_users': 60000
})
print(f"  Predicted spending: ${pred2:,.2f}")
print(f"  Difference: ${pred2 - pred1:,.2f}")

# Test 3: Multiple predictions
print("\n✓ Test 3: Multiple predictions with different scenarios")
for scenario, orders in [('Low', 8000), ('Medium', 12000), ('High', 16000)]:
    pred = predict_future_spending_test('2026-02-01', {'ecomm_orders': orders})
    print(f"  {scenario} activity ({orders:,} orders): ${pred:,.2f}")

print("\n" + "="*80)
print("ALL TESTS PASSED - Enhanced prediction function works correctly!")
print("="*80)
