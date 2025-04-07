import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import yfinance as yf
import csv

# FILE: cup_handle.py
import matplotlib.pyplot as plt

# GET TICKER AND DOWNLOAD DATA VIA YFINANCE
ticker = input("Ticker Symbol: ").upper()
df = yf.download(ticker, period='max', progress=False)
df = df.reset_index()

# Use the 'Close' price for pattern detection and also retrieve Volume.
prices = np.ravel(df['Close'].values)
dates = pd.to_datetime(df['Date'].values)
volumes = np.ravel(df['Volume'].values)

def detect_cup_handle(prices, dates, volumes,
                      min_peak_distance=20,
                      peak_tolerance=0.05,
                      min_cup_depth=0.10,
                      handle_window=20,
                      max_handle_retrace=0.03,
                      min_duration_days=49,
                      max_duration_days=455,
                      volume_multiplier=1.2):
    """
    Detect potential classic cup and handle patterns with slightly looser parameters.
    """
    patterns = []
    peaks, _ = find_peaks(prices, distance=min_peak_distance)

    for i in range(len(peaks) - 1):
        cup_start = peaks[i]
        for j in range(i+1, len(peaks)):
            cup_end = peaks[j]
            if cup_end - cup_start < min_peak_distance:
                continue

            price_start, price_end = prices[cup_start], prices[cup_end]
            if abs(price_start - price_end) / price_start > peak_tolerance:
                continue

            cup_region = prices[cup_start:cup_end+1]
            local_min_idx = np.argmin(cup_region)
            cup_bottom = cup_region[local_min_idx]
            cup_bottom_idx = cup_start + local_min_idx

            if (price_start - cup_bottom) / price_start < min_cup_depth or (price_end - cup_bottom) / price_end < min_cup_depth:
                continue

            handle_start = cup_end
            handle_region_end = min(cup_end + handle_window, len(prices) - 1)
            handle_region = prices[handle_start:handle_region_end+1]
            if len(handle_region) < 3:
                continue

            handle_peak_rel = np.argmax(handle_region)
            handle_peak = handle_region[handle_peak_rel]
            handle_peak_idx = handle_start + handle_peak_rel

            handle_trough_rel = np.argmin(handle_region)
            handle_trough = handle_region[handle_trough_rel]
            handle_trough_idx = handle_start + handle_trough_rel

            if (price_end - handle_trough) / price_end > max_handle_retrace:
                continue

            avg_volume_cup = np.mean(volumes[cup_start:cup_end+1])
            avg_volume_handle = np.mean(volumes[handle_start:handle_trough_idx+1])
            if avg_volume_handle < avg_volume_cup * volume_multiplier:
                continue

            pattern_start_date = dates[cup_start]
            pattern_end_date = dates[handle_trough_idx]
            pattern_duration = (pattern_end_date - pattern_start_date).days
            if pattern_duration < min_duration_days or pattern_duration > max_duration_days:
                continue

            breakout_idx = None
            breakout_window = handle_window
            for k in range(handle_region_end, min(handle_region_end + breakout_window, len(prices))):
                if prices[k] > handle_peak:
                    breakout_idx = k
                    break

            pattern = {
                'cup_start': cup_start,
                'cup_bottom': cup_bottom_idx,
                'cup_end': cup_end,
                'handle_start': handle_start,
                'handle_peak': handle_peak_idx,
                'handle_end': handle_trough_idx,
                'breakout': breakout_idx,
                'pattern_duration_days': pattern_duration,
                'avg_volume_cup': avg_volume_cup,
                'avg_volume_handle': avg_volume_handle
            }
            patterns.append(pattern)

    return patterns

def simulate_trade(prices, pattern, lookahead=50):
    """
    Simulate a trade based on a detected cup and handle pattern.
    Entry is assumed at the breakout point (stop-buy order slightly above handle peak).
    Profit target = entry price + (cup high - cup bottom), and stop-loss at handle trough.
    """
    if pattern['breakout'] is None:
        return None

    entry_idx = pattern['breakout']
    entry_price = prices[entry_idx]
    measured_move = prices[pattern['cup_end']] - prices[pattern['cup_bottom']]
    target_price = entry_price + measured_move
    stop_price = prices[pattern['handle_end']]

    exit_idx = None
    exit_price = None
    outcome = None

    for i in range(entry_idx+1, min(entry_idx+lookahead, len(prices))):
        if prices[i] >= target_price:
            exit_idx = i
            exit_price = target_price
            outcome = 'win'
            break
        if prices[i] <= stop_price:
            exit_idx = i
            exit_price = stop_price
            outcome = 'loss'
            break
    if exit_idx is None:
        outcome = 'open'

    # Do not execute a trade if the outcome remains "open"
    if outcome == 'open':
        return None

    ret_pct = (exit_price - entry_price) / entry_price
    trade = {
        'entry_idx': entry_idx,
        'entry_price': entry_price,
        'target_price': target_price,
        'stop_price': stop_price,
        'exit_idx': exit_idx,
        'exit_price': exit_price,
        'outcome': outcome,
        'return_pct': ret_pct,
        'pattern': pattern
    }
    return trade

# Run detection with loosened parameters.
patterns = detect_cup_handle(prices, dates, volumes,
                             min_peak_distance=15,
                             peak_tolerance=0.07,
                             min_cup_depth=0.08,
                             handle_window=20,
                             max_handle_retrace=0.05,
                             min_duration_days=49,
                             max_duration_days=455,
                             volume_multiplier=1.1)

trades = []
for pat in patterns:
    if pat['breakout'] is not None:
        trade = simulate_trade(prices, pat)
        if trade:  # Only add if trade is not open.
            trades.append(trade)

# Amend trades to keep only one trade per unique entry index.
unique_trades = {}
for trade in trades:
    if trade['entry_idx'] not in unique_trades:
        unique_trades[trade['entry_idx']] = trade
trades = list(unique_trades.values())

wins = [t for t in trades if t['outcome'] == 'win']
losses = [t for t in trades if t['outcome'] == 'loss']

print("Total trades simulated:", len(trades))
print("Wins:", len(wins))
print("Losses:", len(losses))

if trades:
    trade_log_returns = [np.log(t['exit_price'] / t['entry_price']) for t in trades]
    total_log_return = sum(trade_log_returns)
    avg_log_return = total_log_return / len(trade_log_returns)
    avg_return_percent = np.exp(avg_log_return) - 1

    print("Detailed Calculation:")
    print("Sum of log returns: {:.2%}".format(total_log_return))
    print("Average log return per trade: {:.2%}".format(avg_log_return))
    print("Equivalent average simple return per trade: {:.2%}".format(avg_return_percent))

csv_filename = f"trade_log_{ticker}.csv"
with open(csv_filename, mode='w', newline='') as csvfile:
    fieldnames = ['entry_idx', 'entry_price', 'target_price', 'stop_price',
                  'exit_idx', 'exit_price', 'outcome', 'return_pct', 'cup_start',
                  'cup_bottom', 'cup_end', 'handle_start', 'handle_peak', 'handle_end', 'breakout',
                  'pattern_duration_days', 'avg_volume_cup', 'avg_volume_handle']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    positive_trades = [t for t in trades if t['return_pct'] > 0]
    for t in positive_trades:
        pattern = t['pattern']
        row = {
            'entry_idx': t['entry_idx'],
            'entry_price': t['entry_price'],
            'target_price': t['target_price'],
            'stop_price': t['stop_price'],
            'exit_idx': t['exit_idx'],
            'exit_price': t['exit_price'],
            'outcome': t['outcome'],
            'return_pct': t['return_pct'],
            'cup_start': pattern.get('cup_start'),
            'cup_bottom': pattern.get('cup_bottom'),
            'cup_end': pattern.get('cup_end'),
            'handle_start': pattern.get('handle_start'),
            'handle_peak': pattern.get('handle_peak'),
            'handle_end': pattern.get('handle_end'),
            'breakout': pattern.get('breakout'),
            'pattern_duration_days': pattern.get('pattern_duration_days'),
            'avg_volume_cup': pattern.get('avg_volume_cup'),
            'avg_volume_handle': pattern.get('avg_volume_handle')
        }
        writer.writerow(row)
print(f"Trade log saved to {csv_filename}")

# Plot only executed trades and their associated pattern markers.
plt.figure(figsize=(14,7))
plt.plot(dates, prices, label='Close Price', color='black')

# Loop over executed trades and plot the associated pattern markers.
for t in trades:
    pat = t['pattern']
    plt.scatter(dates[pat['cup_start']], prices[pat['cup_start']], color='green', marker='^', s=100, label='Cup Start')
    plt.scatter(dates[pat['cup_bottom']], prices[pat['cup_bottom']], color='red', marker='v', s=100, label='Cup Bottom')
    plt.scatter(dates[pat['cup_end']], prices[pat['cup_end']], color='green', marker='^', s=100, label='Cup End')
    plt.scatter(dates[pat['handle_end']], prices[pat['handle_end']], color='blue', marker='o', s=100, label='Handle End')
    plt.scatter(dates[pat['handle_peak']], prices[pat['handle_peak']], color='orange', marker='o', s=100, label='Handle Peak')
    plt.plot([dates[pat['cup_start']], dates[pat['cup_bottom']]], [prices[pat['cup_start']], prices[pat['cup_bottom']]], '--', color='gray')
    plt.plot([dates[pat['cup_bottom']], dates[pat['cup_end']]], [prices[pat['cup_bottom']], prices[pat['cup_end']]], '--', color='gray')
    plt.plot([dates[pat['cup_end']], dates[pat['handle_end']]], [prices[pat['cup_end']], prices[pat['handle_end']]], '--', color='gray')
    if pat['breakout'] is not None:
        plt.scatter(dates[pat['breakout']], prices[pat['breakout']], color='purple', marker='*', s=150, label='Breakout')

# Plot trade entry and exit markers.
for t in trades:
    plt.scatter(dates[t['entry_idx']], prices[t['entry_idx']], color='cyan', marker='D', s=100, label='Trade Entry')
    plt.scatter(dates[t['exit_idx']], t['exit_price'], color='magenta', marker='X', s=100, label='Trade Exit')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title(f"Cup and Handle Patterns & Trade Signals for {ticker}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()