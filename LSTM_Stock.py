import yfinance as yf
import numpy as np
import pandas as pd
from pattern_recognition import get_pattern_signals
from chart_pattern import detect_patterns, detect_fractals, process_wedges, do_bollinger_squeeze

# GET TICKER AND DOWNLOAD DATA
ticker = input("Ticker Symbol: ").upper()
df = yf.download(ticker, period='max')
df = df.reset_index()
df.dropna(inplace=True)
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# BASE PRICE CHANGE AND TARGET
df['Price_Change'] = df['Close'].pct_change()  
df['Target'] = df['Close'].shift(-1)
df_duplicate = df.copy()

# CALCULATE RETURNS
df['Return'] = df['Close'].pct_change()

# CALCULATE VOLATILITY FOR MULTIPLE WINDOWS
for win in [5, 10, 20]:
    col_name = f"Volatility_{win}"
    df[col_name] = df['Return'].rolling(window=win).std() 
    
### ---- TREND INDICATORS: SMA + EMA ----
ema_windows = [5, 10, 20, 50]
sma_windows = [5, 10, 20, 50]
for w in sma_windows:
    df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
for w in ema_windows:
    df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
df['above_EMA_10'] = (df['Close'] > df['EMA_10']).astype(int)
df['above_EMA_50'] = (df['Close'] > df['EMA_50']).astype(int)

### ---- MOMENTUM INDICATORS: LOG RETURNS, MOMENTUM, MACD, RSI ----
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
for window in [3, 5, 10]:
    df[f'momentum_{window}'] = df['log_return'].rolling(window).mean()
    df[f'momentum_signal_{window}'] = np.sign(df[f'momentum_{window}'])
macd_pairs = [(12, 26, 9), (5, 35, 5), (19, 39, 9)]
for fast, slow, signal in macd_pairs:
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = 2 * (dif - dea)
    df[f'DIF_{fast}_{slow}'] = dif
    df[f'DEA_{fast}_{slow}'] = dea
    df[f'MACD_{fast}_{slow}'] = macd
rsi_windows = [6, 14, 21]
for w in rsi_windows:
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=w).mean()
    avg_loss = loss.rolling(window=w).mean()
    rs = avg_gain / avg_loss
    df[f'RSI_{w}'] = 100 - (100 / (1 + rs))

### ---- VOLATILITY/MEAN REVERSION INDICATORS: STOCHASTIC & BOLLINGER BANDS ----
stoch_windows = [14, 21]
for w in stoch_windows:
    low = df['Low'].rolling(window=w).min()
    high = df['High'].rolling(window=w).max()
    df[f'Stoch_K_{w}'] = 100 * ((df['Close'] - low) / (high - low))
    df[f'Stoch_D_{w}'] = df[f'Stoch_K_{w}'].rolling(window=3).mean()
    df[f'Williams_%R_{w}'] = -100 * ((high - df['Close']) / (high - low))
rolling_std = df['Close'].rolling(window=20).std()
df['BB_upper'] = df['SMA_20'] + 2 * rolling_std
df['BB_lower'] = df['SMA_20'] - 2 * rolling_std
df['bb_signal'] = np.where(df['Close'] > df['BB_upper'], -1, np.where(df['Close'] < df['BB_lower'], 1, 0))
df['bb_z_score'] = (df['Close'] - df['SMA_20']) / rolling_std

### ---- VOLATILITY INDICATORS: ATR ----
atr_windows = [7, 14, 30]
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
for w in atr_windows:
    df[f'ATR_{w}'] = df['TR'].rolling(window=w).mean()

### ---- MEAN REVERSION FEATURES ----
for period in [10, 20, 50]:
    df[f'distance_from_SMA{period}'] = df['Close'] - df[f'SMA_{period}']
    df[f'abs_distance_SMA{period}'] = df[f'distance_from_SMA{period}'].abs()

### ---- VOLUME-BASED INDICATORS ----
# OBV
df['OBV'] = 0
df.loc[1:, 'OBV'] = np.where(
    df['Close'].diff()[1:] > 0, df['Volume'][1:], 
    np.where(df['Close'].diff()[1:] < 0, -df['Volume'][1:], 0)
).cumsum()
# CMF
cmf_windows = [10, 20]
for w in cmf_windows:
    mf_mult = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_mult * df['Volume']
    df[f'CMF_{w}'] = mf_volume.rolling(w).sum() / df['Volume'].rolling(w).sum()
# VROC
for w in [10, 14, 30]:
    df[f'VROC_{w}'] = df['Volume'].pct_change(periods=w)
# Volume SMA and Trend
df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
df['volume_trend'] = (df['Volume_SMA_5'] > df['Volume_SMA_20']).astype(int)
# Volume Z-score
for w in [10, 20]:
    volume_mean = df['Volume'].rolling(window=w).mean()
    volume_std = df['Volume'].rolling(window=w).std()
    df[f'volume_zscore_{w}'] = (df['Volume'] - volume_mean) / volume_std

# Ratio Features
epsilon = 1e-6  # small constant to avoid division by zero

# ---- Ratio Features for Moving Averages ----
df['SMA_5_to_SMA_10'] = df['SMA_5'] / (df['SMA_10'] + epsilon)
df['SMA_10_to_SMA_20'] = df['SMA_10'] / (df['SMA_20'] + epsilon)
df['SMA_20_to_SMA_50'] = df['SMA_20'] / (df['SMA_50'] + epsilon)

# ---- Ratio Features for Exponential Moving Averages ----
df['EMA_5_to_EMA_10'] = df['EMA_5'] / (df['EMA_10'] + epsilon)
df['EMA_10_to_EMA_20'] = df['EMA_10'] / (df['EMA_20'] + epsilon)
df['EMA_20_to_EMA_50'] = df['EMA_20'] / (df['EMA_50'] + epsilon)

# ---- Ratio Features for RSI ----
df['RSI_6_to_RSI_14'] = df['RSI_6'] / (df['RSI_14'] + epsilon)
df['RSI_14_to_RSI_21'] = df['RSI_14'] / (df['RSI_21'] + epsilon)

# ---- Ratio Features for ATR (Volatility) ----
df['ATR_7_to_ATR_14'] = df['ATR_7'] / (df['ATR_14'] + epsilon)
df['ATR_14_to_ATR_30'] = df['ATR_14'] / (df['ATR_30'] + epsilon)

# ---- Ratio for Volume-Based Moving Averages ----
df['Vol_SMA_5_to_SMA_20'] = df['Volume_SMA_5'] / (df['Volume_SMA_20'] + epsilon)

# ---- Additional Ratios ----
# Momentum ratios: comparing short vs. medium term momentum
df['momentum_3_to_momentum_5'] = df['momentum_3'] / (df['momentum_5'] + epsilon)
df['momentum_5_to_momentum_10'] = df['momentum_5'] / (df['momentum_10'] + epsilon)

# MACD ratios: comparing MACD values from different parameter settings
df['MACD_12_26_to_MACD_5_35'] = df['MACD_12_26'] / (df['MACD_5_35'] + epsilon)
df['MACD_5_35_to_MACD_19_39'] = df['MACD_5_35'] / (df['MACD_19_39'] + epsilon)

# Bollinger Bands: ratio of the band width to the SMA (a measure of relative volatility)
df['BB_width_to_SMA_20'] = (df['BB_upper'] - df['BB_lower']) / (df['SMA_20'] + epsilon)

## Pattern Recognition Signals
# Get pattern signals using the refactored function
pattern_df = get_pattern_signals(df_duplicate)
# Merge pattern signals with your main DataFrame based on the Date
df = pd.merge(df, pattern_df, on="Date", how="left")
df.dropna(inplace=True)

##########################################################
## Chart Recognitions
##########################################################
# Assume df is your main DataFrame with a 'Close' price column
prices = df['Close'].to_numpy().flatten()
prices = prices[~np.isnan(prices)]

# Call the function to detect patterns
patterns = detect_patterns(prices)

# Process Double Tops
double_top_features = []  # List of tuples: (index, drop_pct, peak_diff, signal)
for first, second, trough in patterns["double_tops"]:
    peak1 = prices[first]
    peak2 = prices[second]
    trough_price = prices[trough]
    drop_pct = (peak1 - trough_price) / peak1
    peak_diff = abs(peak1 - peak2)
    signal = -1  # Sell signal for double tops.
    double_top_features.append((first, drop_pct, peak_diff, signal))

# Process Double Bottoms
double_bottom_features = []  # List of tuples: (index, rise_pct, peak_diff, signal)
for t1, t2, peak in patterns["double_bottoms"]:
    trough1_price = prices[t1]
    trough2_price = prices[t2]
    peak_price = prices[peak]
    rise_pct = (peak_price - trough1_price) / trough1_price
    peak_diff = abs(trough1_price - trough2_price)
    signal = 1  # Buy signal for double bottoms.
    double_bottom_features.append((t1, rise_pct, peak_diff, signal))

# Process Triple Tops
triple_top_features = []  # List of tuples: (index, avg_drop_pct, avg_peak_diff, signal)
for p1, p2, p3, trough1, trough2 in patterns["triple_tops"]:
    peak1 = prices[p1]
    peak2 = prices[p2]
    peak3 = prices[p3]
    trough_price1 = prices[trough1]
    trough_price2 = prices[trough2]
    drop_pct1 = (peak1 - trough_price1) / peak1
    drop_pct2 = (peak2 - trough_price2) / peak2
    avg_drop_pct = (drop_pct1 + drop_pct2) / 2
    peak_diff1 = abs(peak1 - peak2)
    peak_diff2 = abs(peak2 - peak3)
    avg_peak_diff = (peak_diff1 + peak_diff2) / 2
    signal = -1  # Sell signal for triple tops.
    triple_top_features.append((p1, avg_drop_pct, avg_peak_diff, signal))

# Process Triple Bottoms
triple_bottom_features = []  # List of tuples: (index, avg_rise_pct, avg_trough_diff, signal)
for t1, t2, t3, peak1, peak2 in patterns["triple_bottoms"]:
    trough1 = prices[t1]
    trough2 = prices[t2]
    trough3 = prices[t3]
    peak_price1 = prices[peak1]
    peak_price2 = prices[peak2]
    rise_pct1 = (peak_price1 - trough1) / peak_price1
    rise_pct2 = (peak_price2 - trough2) / peak_price2
    avg_rise_pct = (rise_pct1 + rise_pct2) / 2
    trough_diff1 = abs(trough1 - trough2)
    trough_diff2 = abs(trough2 - trough3)
    avg_trough_diff = (trough_diff1 + trough_diff2) / 2
    signal = 1  # Buy signal for triple bottoms.
    triple_bottom_features.append((t1, avg_rise_pct, avg_trough_diff, signal))

# Process Head-and-Shoulders
head_shoulders_features = []  # List of tuples: (index, ratio, signal)
for left, head, right, trough1, trough2 in patterns["head_and_shoulders"]:
    left_price = prices[left]
    head_price = prices[head]
    right_price = prices[right]
    # For example, compute a ratio: head price divided by the average of the shoulder prices.
    avg_shoulder = (left_price + right_price) / 2
    ratio = head_price / avg_shoulder if avg_shoulder != 0 else np.nan
    signal = -1  # Sell signal for head-and-shoulders.
    head_shoulders_features.append((left, ratio, signal))

# Add New Feature Columns to the DataFrame
# Initialize new columns for double tops.
df['double_top_signal'] = 0
df['double_top_drop_pct'] = np.nan
df['double_top_peak_diff'] = np.nan

# Initialize new columns for double bottoms.
df['double_bottom_signal'] = 0
df['double_bottom_rise_pct'] = np.nan
df['double_bottom_peak_diff'] = np.nan

# Initialize new columns for triple tops.
df['triple_top_signal'] = 0
df['triple_top_avg_drop_pct'] = np.nan
df['triple_top_avg_peak_diff'] = np.nan

# Initialize new columns for triple bottoms.
df['triple_bottom_signal'] = 0
df['triple_bottom_avg_rise_pct'] = np.nan
df['triple_bottom_avg_trough_diff'] = np.nan

# Initialize new columns for head-and-shoulders.
df['head_shoulders_signal'] = 0
df['head_shoulders_ratio'] = np.nan

# For double tops.
for index, drop_pct, peak_diff, signal in double_top_features:
    df.loc[index, 'double_top_signal'] = int(signal)
    df.loc[index, 'double_top_drop_pct'] = float(drop_pct) 
    df.loc[index, 'double_top_peak_diff'] = float(peak_diff) 

# For double bottoms.
for index, rise_pct, peak_diff, signal in double_bottom_features:
    df.loc[index, 'double_bottom_signal'] = int(signal)
    df.loc[index, 'double_bottom_rise_pct'] = float(rise_pct) 
    df.loc[index, 'double_bottom_peak_diff'] = float(peak_diff) 

# For triple tops.
for index, avg_drop_pct, avg_peak_diff, signal in triple_top_features:
    df.loc[index, 'triple_top_signal'] = int(signal)
    df.loc[index, 'triple_top_avg_drop_pct'] = float(avg_drop_pct) 
    df.loc[index, 'triple_top_avg_peak_diff'] = float(avg_peak_diff) 

# For triple bottoms.
for index, avg_rise_pct, avg_trough_diff, signal in triple_bottom_features:
    df.loc[index, 'triple_bottom_signal'] = int(signal)
    df.loc[index, 'triple_bottom_avg_rise_pct'] = float(avg_rise_pct) 
    df.loc[index, 'triple_bottom_avg_trough_diff'] = float(avg_trough_diff) 

# For head-and-shoulders.
for index, ratio, signal in head_shoulders_features:
    df.loc[index, 'head_shoulders_signal'] = int(signal)
    df.loc[index, 'head_shoulders_ratio'] = float(ratio) 
    
# Get the resolved wedge intervals.
resolved_rising, resolved_falling = process_wedges(prices, df['Date'])

# Create new DataFrame columns to mark the wedge intervals.
df['rising_wedge'] = 0.0
df['falling_wedge'] = 0.0

# For each resolved rising wedge interval, mark the rows (based on the Date column)
# that fall within the interval with the rising wedge slope (or 1 as a flag, if preferred).
for start_date, end_date, slope in resolved_rising:
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df.loc[mask, 'rising_wedge'] = slope  # Alternatively, you can set this to 1 if you just want a flag.

# For each resolved falling wedge interval, mark the corresponding rows.
for start_date, end_date, slope in resolved_falling:
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df.loc[mask, 'falling_wedge'] = slope  # Alternatively, set to -1 for a flag.

# Call the fractal detection function.
bullish_fractals, bearish_fractals = detect_fractals(df)
# Create binary feature columns for fractals.
df['bullish_fractal'] = 0  # 1 indicates a bullish fractal
df['bearish_fractal'] = 0  # 1 indicates a bearish fractal
# Mark detected bullish fractals.
for idx in bullish_fractals:
    df.loc[idx, 'bullish_fractal'] = 1
# Mark detected bearish fractals.
for idx in bearish_fractals:
    df.loc[idx, 'bearish_fractal'] = 1

# Bollinger
df_bollinger = do_bollinger_squeeze(ticker, period='max')
# Reset index so that 'Date' becomes a column
df_bollinger = df_bollinger.reset_index()

# Ensure the Date column exists and is named correctly.
if 'Date' not in df_bollinger.columns:
    df_bollinger = df_bollinger.rename(columns={'index': 'Date'})
    
# Flatten the columns if they are a MultiIndex
if isinstance(df_bollinger.columns, pd.MultiIndex):
    df_bollinger.columns = df_bollinger.columns.get_level_values(0)

# Convert the Date columns in both DataFrames to datetime for consistency.
df_bollinger['Date'] = pd.to_datetime(df_bollinger['Date'])
df['Date'] = pd.to_datetime(df['Date'])

# Select only the desired columns from the Bollinger DataFrame.
columns_to_merge = [
    'Date','SMA', 'STD', 'UpperBand', 'LowerBand', 'Bandwidth',
    'MinBandwidth', 'Squeeze', 'RSI', 'Vol_MA',
    'BreakoutUp', 'BreakoutDown', 'ConfirmedBreakoutUp', 'ConfirmedBreakoutDown'
]
df_bollinger_subset = df_bollinger[columns_to_merge]
    
# Merge the Bollinger features into your main DataFrame on the Date column.
df = pd.merge(df, df_bollinger_subset, on='Date', how='left')

# Fill numeric columns with 0
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

# Optionally, fill other non-datetime columns (or leave them)
non_datetime_cols = df.columns.difference(df.select_dtypes(include=['datetime64[ns]']).columns)
df[non_datetime_cols] = df[non_datetime_cols].fillna(0)

df.to_csv('data.csv', index=False)
# Note: The above indicators were selected for their historical and empirical predictive value:
# • Moving Averages & MACD: Identify trend shifts and momentum reversals.
# • RSI & Stochastic: Indicate overbought/oversold conditions and potential mean reversion.
# • Bollinger Bands: Signal volatility extremes and possible reversal points.
# • ATR: Measures volatility to contextualize price movements.
# • OBV, CMF & VROC: Incorporate volume dynamics to confirm trends.
# These indicators are commonly combined to form a confluence of signals with improved reliability.



# Define macro indicators and their Yahoo Finance tickers
macro_symbols = {
    'sp500': '^GSPC',
    'vix': '^VIX',
    'treasury_yield': '^TNX',
    'dollar_index': 'DX-Y.NYB',
    'oil': 'CL=F',
    'gold': 'GC=F'
}

# Download and process each macro indicator one by one
macro_dfs = []
for name, symbol in macro_symbols.items():
    temp_df = yf.download(symbol, period='max')
    if temp_df.empty:
        print(f"Skipping {name}: no data found for symbol {symbol}.")
        continue
    temp_df = temp_df.reset_index()[['Date', 'Close']]
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    temp_df.rename(columns={'Close': name}, inplace=True)
    macro_dfs.append(temp_df)
    
if not macro_dfs:
    raise ValueError("No macro indicator data available to merge.")

# Merge all macro dataframes on the Date column using inner join
macro_df = macro_dfs[0]
for df_temp in macro_dfs[1:]:
    macro_df = pd.merge(macro_df, df_temp, on='Date', how='inner')

# Flatten macro_df columns in case they are multi-indexed
if isinstance(macro_df.columns, pd.MultiIndex):
    macro_df.columns = [col[1] if isinstance(col, tuple) and col[1] else col[0] for col in macro_df.columns]

# Ensure Date column is present
if 'Date' not in macro_df.columns:
    macro_df.reset_index(inplace=True)

for asset, ticker in macro_symbols.items():
    if ticker in macro_df.columns:
        macro_df[f'{asset}_return'] = macro_df[ticker].pct_change()
        macro_df[f'{asset}_return_ratio'] = macro_df[ticker].pct_change() / df['Return']
        macro_df[f'{asset}_volatility'] = macro_df[ticker].rolling(5).std()
        macro_df[f'{asset}_volatility_ratio'] = macro_df[ticker].rolling(5).std() / df['Volatility_5']
        macro_df[f'{asset}_volatility_10_days'] = macro_df[ticker].rolling(10).std()
        macro_df[f'{asset}_volatility_10_ratio'] = macro_df[ticker].rolling(10).std() / df['Volatility_10']
        macro_df[f'{asset}_volatility_20_days'] = macro_df[ticker].rolling(20).std() 
        macro_df[f'{asset}_volatility_20_ratio'] = macro_df[ticker].rolling(20).std() / df['Volatility_20']
        
# Merge the macro indicator dataframe with stock data
df = pd.merge(df, macro_df, on='Date', how='inner')
df.fillna(0, inplace=True)
df.to_csv('data.csv', index=False)