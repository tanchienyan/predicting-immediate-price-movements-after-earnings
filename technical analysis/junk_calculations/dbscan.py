import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import yfinance as yf

# GET TICKER AND DOWNLOAD DATA VIA YFINANCE
ticker = input("Ticker Symbol: ").upper()
df = yf.download(ticker, period='max')
df = df.reset_index()

# Use the 'Close' price for pattern detection.
prices = np.ravel(df['Close'].values)

# Step 1: Find local peaks (resistance) and troughs (support)
high_peaks, _ = find_peaks(prices, distance=10)
low_peaks, _ = find_peaks(-prices, distance=10)

# Add binary indicators to the DataFrame
df['LocalPeak'] = False
df.loc[high_peaks, 'LocalPeak'] = True

df['LocalTrough'] = False
df.loc[low_peaks, 'LocalTrough'] = True

# Combine and cluster to form zones
levels = np.concatenate((prices[high_peaks], prices[low_peaks]))
levels = levels.reshape(-1, 1)

# Step 2: Cluster support/resistance levels using DBSCAN
db = DBSCAN(eps=3, min_samples=3).fit(levels)
clusters = pd.DataFrame({'Price': levels.flatten(), 'Cluster': db.labels_})

# Step 3: Calculate average price of each cluster (zone center)
zones = clusters[clusters['Cluster'] != -1].groupby('Cluster').mean().reset_index()
zones = zones.sort_values(by='Price').reset_index(drop=True)
zone_centers = zones['Price'].values

# Step 4: Backtest trading strategy and record triggers
balance = 0
position = None
entry_price = 0
trade_log = []   # Records only actual trades
signal_log = []  # Records every evaluation with signal: 1 (buy), -1 (sell), or 0 (no trade)

for i in range(1, len(df)):
    p = df['Close'].iloc[i].item()
    prev_price = df['Close'].iloc[i-1].item()
    signal = 0             
    trigger_reason = None  
    distance = np.nan      
    
    for z in zones['Price'].values.flatten():
        z = float(z)  # Ensure zone value is a scalar
        zone_range = 0.02 * z
        if abs(p - z) <= zone_range:
            distance = abs(p - z)
            if p > prev_price and p < z:
                signal = 1
                trigger_reason = "Bounce at Support"
                if position is None:
                    position = 'long'
                    entry_price = p
                    trade_log.append({
                        'Date': df['Date'].iloc[i],
                        'Action': 'Buy',
                        'Price': p,
                        'Signal': signal,
                        'Distance': distance,
                        'Trigger': trigger_reason
                    })
                break
            elif p < prev_price and p > z:
                signal = -1
                trigger_reason = "Rejection at Resistance"
                if position == 'long':
                    pnl = p - entry_price
                    balance += pnl
                    trade_log.append({
                        'Date': df['Date'].iloc[i],
                        'Action': 'Sell',
                        'Price': p,
                        'PnL': pnl,
                        'Signal': signal,
                        'Distance': distance,
                        'Trigger': trigger_reason
                    })
                    position = None
                break
    signal_log.append({
        'Date': df['Date'].iloc[i],
        'Signal': signal,
        'Distance': distance,
        'Trigger': trigger_reason
    })
    
# Step 5: Show results
trades_df = pd.DataFrame(trade_log)
signals_df = pd.DataFrame(signal_log)

print("\nTrade Log (Actual Trades):")
print(trades_df)
print(f"\nFinal Balance: ${balance:.2f}")

if len(trades_df) >= 2:
    sells = trades_df[trades_df['Action'] == 'Sell']
    wins = sells['PnL'] > 0
    print(f"Total Trades: {len(sells)}")
    print(f"Win Rate: {wins.sum()}/{len(wins)} = {100 * wins.mean():.2f}%")

# Plot the data with zones
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['Close'], label='Close Price')

# Ensure zones is not empty before iterating
if not zones.empty:
    for z in zones['Price'].values:
        plt.axhline(z, color='orange', linestyle='--', alpha=0.5)

plt.title(f"{ticker} - Price with Support/Resistance Zones")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
