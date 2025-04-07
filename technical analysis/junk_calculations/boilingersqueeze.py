import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
pd.set_option('future.no_silent_downcasting', True)

# Ask user for ticker symbol and download data.
ticker = input("Ticker Symbol: ").upper()
# Explicitly set auto_adjust=False (default now is True in yfinance)
df = yf.download(ticker, period='max', progress=False, auto_adjust=False)
df = df.reset_index()

# Extract 'Close', 'Date', and 'Volume'.
prices = df['Close'].values
dates = pd.to_datetime(df['Date'])
volumes = df['Volume'].values
# Ensure 'Date' is set as the index
df.set_index('Date', inplace=True)

# --- Bollinger Bands Calculation ---
window = 20  # 20-day moving average period
df['SMA'] = df['Close'].rolling(window=window).mean()
df['STD'] = df['Close'].rolling(window=window).std()
df['UpperBand'] = df['SMA'] + 2 * df['STD']
df['LowerBand'] = df['SMA'] - 2 * df['STD']

# Compute Bandwidth as a percentage of the SMA.
df['Bandwidth'] = (df['UpperBand'] - df['LowerBand']) / df['SMA']

# --- Identify the Squeeze ---
squeeze_window = 126  # roughly 6 months of trading days
# Use min_periods to avoid NaNs when there are less data points than the window size.
df['MinBandwidth'] = df['Bandwidth'].rolling(window=squeeze_window, min_periods=squeeze_window).min()
df['Squeeze'] = df['Bandwidth'] <= (df['MinBandwidth'] * 1.03)

# RSI calculation function (using a 14-day period)
def rsi(series, period=14):
    delta = series.diff()
    # Positive gains (or 0 if negative)
    gain = delta.clip(lower=0)
    # Negative gains (or 0 if positive) -- take absolute value for losses.
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = rsi(df['Close'], period=14)
# Compute a simple Volume Moving Average (20-day)
df['Vol_MA'] = df['Volume'].rolling(window=20).mean()

# --- Identify Breakout Signals ---
# A bullish breakout occurs when the price crosses above the UpperBand after a squeeze.
# A bearish breakout occurs when the price crosses below the LowerBand after a squeeze.
# Use .shift(1) on the Squeeze flag to require that the previous period was in a squeeze.
# Ensure numeric types for relevant columns and keep .squeeze() calls
close_series = pd.to_numeric(df['Close'].squeeze(), errors='coerce')
upper_series = pd.to_numeric(df['UpperBand'].squeeze(), errors='coerce')
lower_series = pd.to_numeric(df['LowerBand'].squeeze(), errors='coerce')
volma_series = pd.to_numeric(df['Vol_MA'].squeeze(), errors='coerce')
volume_series = df['Volume'].squeeze()

# Prepare the shifted Squeeze flag with inferred object types
squeeze_shift = df['Squeeze'].shift(1).fillna(False).infer_objects(copy=False)

# Define breakout conditions using the temporary variables
df['BreakoutUp'] = (
    (close_series > upper_series) &
    squeeze_shift &
    (df['RSI'] > 50) &
    (volume_series > 1.2* volma_series)
)

df['BreakoutDown'] = (
    (close_series < lower_series) &
    squeeze_shift &
    (df['RSI'] < 50) &
    (volume_series > 1.2* volma_series)
)

# Calculate the close price 5 trading days later
close_after = close_series.shift(-10)

# Confirm breakout events:
# For a breakout up: the breakout day price must be lower than the price 5 days later.
# For a breakout down: the breakout day price must be higher than the price 5 days later.

df['ConfirmedBreakoutUp'] = df['BreakoutUp'] & (close_series < close_after)
df['ConfirmedBreakoutDown'] = df['BreakoutDown'] & (close_series > close_after)

# Replace NaN values (from the shift) with False
df['ConfirmedBreakoutUp'] = df['ConfirmedBreakoutUp'].fillna(False)
df['ConfirmedBreakoutDown'] = df['ConfirmedBreakoutDown'].fillna(False)

# --- Plotting ---
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Close'], label='Close Price', color='lightblue')
# plt.plot(df.index, df['SMA'], label='20-day SMA', color='black', linestyle='--')
# plt.plot(df.index, df['UpperBand'], label='Upper Band', color='lightgreen', linestyle='--')
# plt.plot(df.index, df['LowerBand'], label='Lower Band', color='lightcoral', linestyle='--')

# # Highlight squeeze periods (orange dots)
# squeeze_points = df.loc[df['Squeeze']]
# plt.scatter(squeeze_points.index, squeeze_points['Close'], color='orange', label='Squeeze', s=50)

# Mark confirmed breakout signals with larger markers:
plt.scatter(df.loc[df['ConfirmedBreakoutUp']].index, 
            df.loc[df['ConfirmedBreakoutUp'], 'Close'],
            color='green', marker='^', s=100, label='Breakout Up')
plt.scatter(df.loc[df['ConfirmedBreakoutDown']].index, 
            df.loc[df['ConfirmedBreakoutDown'], 'Close'],
            color='red', marker='v', s=100, label='Breakout Down')

plt.title(f"Bollinger Squeeze for {ticker}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

