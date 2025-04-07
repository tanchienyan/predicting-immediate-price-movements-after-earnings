import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load data
df = pd.read_csv("data.csv")

# Define target
target = 'Target'
X = df.select_dtypes(include='number').drop(columns=[target])
y = df[target]
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

# --- Notebook-style MI function ---
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# --- Run MI Analysis ---
mi_scores = make_mi_scores(X, y)

# Create a table with ranking for MI scores
mi_table = pd.DataFrame({'MI Score': mi_scores})
mi_table['Rank'] = mi_table['MI Score'].rank(ascending=False, method='dense').astype(int)
mi_table = mi_table.sort_values(by='MI Score', ascending=False)
# Reset the index to include the feature names as a column called 'Feature'
mi_table = mi_table.reset_index().rename(columns={'index': 'Feature'})

mi_table.to_csv('top_features.csv', index=False)

# def prepare_data(df):
#     # Select features based on those actually in df. These technical indicators have predictive value.
#     features = [
#         'Close', 'Price_Change',
#         'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
#         'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
#         'RSI_6', 'RSI_14', 'RSI_21',
#         'MACD_12_26', 'MACD_5_35', 'MACD_19_39',
#         'DIF_12_26', 'DEA_12_26',
#         'BB_upper', 'BB_lower',
#         'bb_z_score', 'bb_signal',
#         'OBV', 'CMF_10', 'CMF_20', 'VROC_10', 'VROC_14', 'VROC_30',
#         'Volume_SMA_5', 'Volume_SMA_20', 'volume_trend',
#         'volume_zscore_10', 'volume_zscore_20',
#         'ATR_7', 'ATR_14', 'ATR_30',
#         'momentum_3', 'momentum_5', 'momentum_10',
#         'momentum_signal_3', 'momentum_signal_5', 'momentum_signal_10',
#         'Stoch_K_14', 'Stoch_D_14', 'Stoch_K_21', 'Stoch_D_21',
#         'Williams_%R_14', 'Williams_%R_21',
#         'distance_from_SMA10', 'distance_from_SMA20', 'distance_from_SMA50',
#         'abs_distance_SMA10', 'abs_distance_SMA20', 'abs_distance_SMA50',
#         'above_EMA_10', 'above_EMA_50',
#         # Ratio features
#         'SMA_5_to_SMA_10', 'SMA_10_to_SMA_20', 'SMA_20_to_SMA_50',
#         'EMA_5_to_EMA_10', 'EMA_10_to_EMA_20', 'EMA_20_to_EMA_50',
#         'RSI_6_to_RSI_14', 'RSI_14_to_RSI_21',
#         'ATR_7_to_ATR_14', 'ATR_14_to_ATR_30',
#         'Vol_SMA_5_to_SMA_20',
#         'momentum_3_to_momentum_5', 'momentum_5_to_momentum_10',
#         'MACD_12_26_to_MACD_5_35', 'MACD_5_35_to_MACD_19_39',
#         'BB_width_to_SMA_20',
#         # Candlestick pattern features as defined in pattern_recognition.py
#         'Signal','HT_DCPERIOD','HT_DCPHASE','CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
#         'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
#         'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
#         'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
#         'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
#         'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
#         'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE',
#         'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK',
#         'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM',
#         'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW',
#         'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
#         'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR',
#         'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH',
#         'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER',
#         'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS',
#         # Technical indicators
#         'TRANGE', 'ADX', 'ADXR', 'APO', 'AROON_UP', 'AROON_DOWN', 'AROONOSC',
#         'BOP', 'CCI', 'CMO', 'DX', 'MACDEXT', 'MACDEXT_signal', 'MACDEXT_hist',
#         'MACDFIX', 'MACDFIX_hist', 'MFI', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI',
#         'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'STOCHF_fastk',
#         'STOCHF_fastd', 'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX', 'ULTOSC', 'WILLR',
#         # Chart pattern signal features
#         'double_top_signal', 'double_top_drop_pct', 'double_top_peak_diff',
#         'double_bottom_signal', 'double_bottom_rise_pct', 'double_bottom_peak_diff',
#         'triple_top_signal', 'triple_top_avg_drop_pct', 'triple_top_avg_peak_diff',
#         'triple_bottom_signal', 'triple_bottom_avg_rise_pct', 'triple_bottom_avg_trough_diff',
#         'head_shoulders_signal', 'head_shoulders_ratio','rising_wedge', 'falling_wedge','bullish_fractal', 'bearish_fractal',
#         'SMA', 'STD', 'UpperBand', 'LowerBand', 'Bandwidth','MinBandwidth', 'Squeeze', 'RSI', 'Vol_MA','BreakoutUp', 'BreakoutDown', 
#         'ConfirmedBreakoutUp', 'ConfirmedBreakoutDown',
#         # Macro indicators
#         'sp500', 'vix', 'treasury_yield', 'dollar_index', 'oil', 'gold','sp500_return', 'vix_return', 'treasury_yield_return',
#         'dollar_index_return', 'oil_return', 'gold_return','sp500_volatility', 'vix_volatility', 'treasury_yield_volatility',
#         'dollar_index_volatility', 'oil_volatility', 'gold_volatility','sp500_volatility_10_days', 'vix_volatility_10_days', 
#         'treasury_yield_volatility_10_days','dollar_index_volatility_10_days', 'oil_volatility_10_days', 'gold_volatility_10_days',
#         'sp500_volatility_20_days', 'vix_volatility_20_days', 'treasury_yield_volatility_20_days','dollar_index_volatility_20_days', 
#         'oil_volatility_20_days', 'gold_volatility_20_days', 'sp500_return_ratio', 'vix_return_ratio', 'treasury_yield_return_ratio',
#         'dollar_index_return_ratio', 'oil_return_ratio', 'gold_return_ratio','sp500_volatility_ratio', 'vix_volatility_ratio', 
#         'treasury_yield_volatility_ratio','dollar_index_volatility_ratio', 'oil_volatility_ratio', 'gold_volatility_ratio',
#         'sp500_volatility_10_ratio', 'vix_volatility_10_ratio', 'treasury_yield_volatility_10_ratio','dollar_index_volatility_10_ratio', 
#         'oil_volatility_10_ratio', 'gold_volatility_10_ratio','sp500_volatility_20_ratio', 'vix_volatility_20_ratio', 
#         'treasury_yield_volatility_20_ratio','dollar_index_volatility_20_ratio', 'oil_volatility_20_ratio', 'gold_volatility_20_ratio'
#     ]
    
#     # Ensure only features that exist in the DataFrame are used
#     features = [f for f in features if f in df.columns]
#     df_clean = df.dropna(subset=features + ['Target'])
#     X, y = [], []
#     for i in range(look_back, len(df_clean)):
#         X.append(df_clean[features].iloc[i - look_back:i].values)
#         y.append(df_clean['Target'].iloc[i])
#     return np.array(X), np.array(y), features

# prepare_data(df, 60)

# # Drop NaNs and define features + target
# df_clean = df.dropna(subset=['Target'])
# X = df_clean[features]
# y = df_clean['Target']

# # Identify discrete features (integers)
# discrete_features = X.dtypes == int

# # Calculate MI scores
# mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
# mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# # Display top features
# print(mi_series.head(30))  # You can change this number to see more


# # --- Prepare Data ---
# look_back = 60  # you can adjust the look-back window
# X, y, selected_features = prepare_data(df, look_back)

# # For feature selection and evaluation, we use the last time step (most recent values) from each sample.
# X_last = X[:, -1, :]  # shape: (n_samples, n_features)

# # --- Scale Features for PCA ---
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_last)

# # ============================================================
# # 1. Mutual Information for Feature Selection
# # ============================================================
# mi_scores = mutual_info_regression(X_last, y)
# mi_scores_series = pd.Series(mi_scores, index=selected_features)
# mi_scores_series = mi_scores_series.sort_values(ascending=False)
# print("Mutual Information Scores:")
# print(mi_scores_series)

# # Plot MI scores
# plt.figure(figsize=(10, 6))
# mi_scores_series.plot(kind='bar')
# plt.title("Mutual Information Scores")
# plt.ylabel("MI Score")
# plt.show()

# # ============================================================
# # 2. Principal Component Analysis (PCA)
# # ============================================================

# pca = PCA()
# X_pca = pca.fit_transform(X_scaled)
# explained_variance = np.cumsum(pca.explained_variance_ratio_)
# print("Cumulative Explained Variance by PCA Components:")
# for i, variance in enumerate(explained_variance, 1):
#     print(f"PC{i}: {variance:.2f}")

# # Plot cumulative explained variance
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
# plt.title("Cumulative Explained Variance by PCA Components")
# plt.xlabel("Number of Components")
# plt.ylabel("Cumulative Explained Variance")
# plt.grid(True)
# plt.show()

# # ============================================================
# # 3. Permutation Importance with RandomForestRegressor
# # ============================================================
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_last, y)
# perm_importance = permutation_importance(rf_model, X_last, y, n_repeats=10, random_state=42)
# perm_importance_series = pd.Series(perm_importance.importances_mean, index=selected_features)
# perm_importance_series = perm_importance_series.sort_values(ascending=False)
# print("Permutation Importance Scores (RandomForestRegressor):")
# print(perm_importance_series)

# # Plot permutation importance scores
# plt.figure(figsize=(10, 6))
# perm_importance_series.plot(kind='bar')
# plt.title("Permutation Importance (RandomForestRegressor)")
# plt.ylabel("Importance")
# plt.show()

# # ============================================================
# # 4. Model Evaluation with XGBoost (Optional)
# # ============================================================
# xgb_model = XGBRegressor(n_estimators=100, random_state=42)
# xgb_model.fit(X_last, y)
# y_pred = xgb_model.predict(X_last)
# mae = mean_absolute_error(y, y_pred)
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)
# print(f"XGBoost Performance: MAE = {mae:.4f}, MSE = {mse:.4f}, R2 = {r2:.4f}")