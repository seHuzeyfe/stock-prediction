import pandas as pd
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller

# Load the dataset
df = pd.read_csv("data/processed/AAPL_model_ready.csv", parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# =================================================================
# 1. Fix Multicollinearity (Remove Redundant Features)
# =================================================================
# Drop highly correlated raw price features (keep only 'close')
df = df.drop(columns=['open', 'high', 'low'])

# Drop redundant SMA features (keep only SMA_20 and distance to SMA_20)
df = df.drop(columns=['sma_5', 'dist_to_sma_5', 'sma_10', 'dist_to_sma_10'])

# =================================================================
# 2. Handle Non-Stationarity
# =================================================================
# Create differenced closing price (1st order differencing)
df['close_diff2'] = df['close'].diff().diff()

# Drop rows with NaN from differencing
df = df.dropna(subset=['close_diff2'])

# Verify stationarity
adf_result = adfuller(df['close_diff2'].dropna())
print(f"ADF p-value for second-order differenced series: {adf_result[1]:.4f}")

# =================================================================
# 3. Handle Outliers
# =================================================================
# List of columns to winsorize (cap extreme values)
winsorize_columns = [
    'volume', 'daily_return', 'intraday_return', 
    'bb_width', 'target_return', 'volume_ratio'
]

for col in winsorize_columns:
    df[col] = winsorize(df[col], limits=[0.01, 0.01])  # 1% trimming on both ends

# =================================================================
# 4. Feature Engineering
# =================================================================
# Create lagged features for returns
for lag in [1, 2, 3]:
    df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)

# Create volatility measure (rolling 5-day std of returns)
df['volatility_5'] = df['daily_return'].rolling(5).std()
df['volume_rolling_mean'] = df['volume'].rolling(5).mean()
df['volume_rolling_std'] = df['volume'].rolling(5).std()
df['return_rolling_mean'] = df['daily_return'].rolling(5).mean()
df['return_rolling_std'] = df['daily_return'].rolling(5).std()

# Create momentum indicator (close vs SMA_20 ratio)
df['momentum_sma_20'] = df['close'] / df['sma_20']

# Drop rows with NaN from rolling operations
df = df.dropna()

# =================================================================
# 5. Final Feature Selection
# =================================================================
# Select relevant features
final_features = [
    'date', 'close', 'volume', 'daily_return', 'intraday_return',
    'volatility_5', 'momentum_sma_20', 'rsi', 'bb_width', 'macd',
    'volume_ratio', 'return_lag_1', 'return_lag_2', 'return_lag_3',
    'volume_rolling_mean', 'volume_rolling_std', 'return_rolling_mean',
    'return_rolling_std', 'target_direction'
]

df = df[final_features]


