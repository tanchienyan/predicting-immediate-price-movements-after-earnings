import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import talib

# GET TICKER AND DOWNLOAD DATA VIA YFINANCE
ticker = input("Ticker Symbol: ").upper()
df = yf.download(ticker, period='max', progress=False)
df = df.reset_index()

# Initialize the 'Fractal' column as an empty string (object dtype)
df['Fractal'] = ''

# Ensure our DataFrame has a datetime column named 'Date'
if 'Date' not in df.columns:
    df['Date'] = pd.to_datetime(df.index)

# Function to detect closing price fractals in the data
def detect_fractals(data):
    """
    Detects bullish and bearish fractals in a DataFrame based on the 'Close' price.
    
    A bullish fractal occurs when the middle close is lower than
    the closes of the two bars before and two bars after.
    
    A bearish fractal occurs when the middle close is higher than
    the closes of the two bars before and two bars after.
    
    Parameters:
        data (DataFrame): DataFrame with a 'Close' column.
        
    Returns:
        bullish (list): Indices where bullish fractals occur.
        bearish (list): Indices where bearish fractals occur.
    """
    bullish = []
    bearish = []
    # We require at least 2 bars before and after the current bar
    for i in range(2, len(data)-2):
        close_window = data['Close'].iloc[i-2:i+3].to_numpy()
        mid_close = close_window[2].item() # get scalar from NumPy array
        next_close = data['Close'].iloc[i + 1].item() # get scalar from single-element Series
        
        if mid_close == np.min(close_window) and np.sum(close_window == mid_close) == 1 and next_close > mid_close: 
            bullish.append(i)
        if mid_close == np.max(close_window) and np.sum(close_window == mid_close) == 1 and next_close < mid_close:
            bearish.append(i)
    return bullish, bearish

# Detect fractals in the DataFrame using closing price
bullish_fractals, bearish_fractals = detect_fractals(df)

# Initialize the 'Fractal' column as NaN with object dtype to allow string assignments
df['Fractal'] = pd.Series(np.nan, index=df.index, dtype=object)
# Assign fractal signals: 'B' for bullish fractal and 'S' for bearish fractal
df.loc[bullish_fractals, 'Fractal'] = 'B'
df.loc[bearish_fractals, 'Fractal'] = 'S'

# Plot the close price and fractal signals
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue', alpha=0.5)

# Plot bullish fractals with green x markers
plt.scatter(df.loc[bullish_fractals, 'Date'], df.loc[bullish_fractals, 'Close'],
            marker='x', color='green', s=50, label='Bullish Fractal')

# Plot bearish fractals with red x markers
plt.scatter(df.loc[bearish_fractals, 'Date'], df.loc[bearish_fractals, 'Close'],
            marker='x', color='red', s=50, label='Bearish Fractal')

plt.legend()
plt.show()

