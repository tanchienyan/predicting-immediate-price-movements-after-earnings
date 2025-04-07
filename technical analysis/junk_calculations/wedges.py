import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
import yfinance as yf
import matplotlib.pyplot as plt

ticker = input("Ticker Symbol: ").upper()
df = yf.download(ticker, period='max')
df = df.reset_index()

# Use the 'Close' price for pattern detection.
prices = np.ravel(df['Close'].values)

def detect_wedges(prices, dates, window_size=50, wedge_type='rising', peak_prominence=0.01, slope_threshold=0.0001):
    wedges = []
    n = len(prices)
    for start in range(0, n - window_size, window_size // 2):
        end = start + window_size

        # Ensure the time span of the window is at least one day
        if (dates.iloc[end - 1] - dates.iloc[start]).days < 1:
            continue

        window_prices = prices[start:end]
        if wedge_type == 'rising':
            indices, _ = find_peaks(window_prices, prominence=peak_prominence)
        elif wedge_type == 'falling':
            indices, _ = find_peaks(-window_prices, prominence=peak_prominence)
        else:
            continue

        if len(indices) >= 2:
            x = np.array(indices)
            y = window_prices[indices]
            slope, _ = np.polyfit(x, y, 1)
            if wedge_type == 'rising' and slope < -slope_threshold:
                global_indices = indices + start  # convert local indices to global indices
                wedges.append((start, end, slope, global_indices))
            elif wedge_type == 'falling' and slope > slope_threshold:
                global_indices = indices + start
                wedges.append((start, end, slope, global_indices))
    return wedges

# Detect rising and falling wedges.
rising_wedges = detect_wedges(prices, df['Date'], window_size=50, wedge_type='rising', 
                              peak_prominence=0.01, slope_threshold=0.0001)
falling_wedges = detect_wedges(prices, df['Date'], window_size=50, wedge_type='falling', 
                               peak_prominence=0.01, slope_threshold=0.0001)

# Plot individual wedge detections.
plt.figure(figsize=(14,8))
plt.plot(df['Date'], prices, label='Close Price', color='blue')

first_rising = True
rising_wedges_intervals = []
for i, (start, end, slope, indices) in enumerate(rising_wedges, 1):
    start_date = df['Date'].iloc[start]
    end_date   = df['Date'].iloc[end - 1]
    if first_rising:
        plt.axvspan(start_date, end_date, color='orange', alpha=0.3,
                    edgecolor='orange', lw=3, label='Rising Wedge')
        first_rising = False
    else:
        plt.axvspan(start_date, end_date, color='orange', alpha=0.3)
    print(f"Rising Wedge {i}: Start = {start_date.date()}, End = {end_date.date()}, Slope = {slope:.5f}")
    rising_wedges_intervals.append((start_date, end_date, slope))

first_falling = True
falling_wedges_intervals = []
for i, (start, end, slope, indices) in enumerate(falling_wedges, 1):
    start_date = df['Date'].iloc[start]
    end_date   = df['Date'].iloc[end - 1]
    if first_falling:
        plt.axvspan(start_date, end_date, color='purple', alpha=0.3, 
                    edgecolor='purple', lw=3, label='Falling Wedge')
        first_falling = False
    else:
        plt.axvspan(start_date, end_date, color='purple', alpha=0.3)
    print(f"Falling Wedge {i}: Start = {start_date.date()}, End = {end_date.date()}, Slope = {slope:.5f}")
    falling_wedges_intervals.append((start_date, end_date, slope))

plt.title("Detected Rising and Falling Wedges")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

def merge_intervals(intervals):
    if not intervals:
        return []
    # Sort intervals by their start date.
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            new_start = last[0]
            new_end = max(last[1], current[1])
            # Weighted average slope based on duration
            duration_last = (last[1] - last[0]).days
            duration_current = (current[1] - current[0]).days
            total_duration = duration_last + duration_current
            new_slope = (last[2]*duration_last + current[2]*duration_current) / total_duration
            merged[-1] = (new_start, new_end, new_slope)
        else:
            merged.append(current)
    return merged

merged_rising_wedges = merge_intervals(rising_wedges_intervals)
merged_falling_wedges = merge_intervals(falling_wedges_intervals)

print("Merged Rising Wedges:")
for i, (start_date, end_date, slope) in enumerate(merged_rising_wedges, 1):
    print(f"{i}: Start = {start_date.date()}, End = {end_date.date()}, Avg Slope = {slope:.5f}")

print("\nMerged Falling Wedges:")
for i, (start_date, end_date, slope) in enumerate(merged_falling_wedges, 1):
    print(f"{i}: Start = {start_date.date()}, End = {end_date.date()}, Avg Slope = {slope:.5f}")

def resolve_conflicts(rising_intervals, falling_intervals):
    """
    Combine rising and falling wedges into a single list and, if any rising and falling
    wedges overlap, keep only the one with the stronger absolute slope.
    """
    # Add wedge type into each tuple.
    combined = []
    for (start, end, slope) in rising_intervals:
        combined.append((start, end, slope, 'rising'))
    for (start, end, slope) in falling_intervals:
        combined.append((start, end, slope, 'falling'))
    
    # Sort combined list by start date.
    combined.sort(key=lambda x: x[0])
    
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined):
            # If the next wedge starts before the current one ends, there's an overlap.
            if combined[j][0] <= combined[i][1]:
                # Only resolve if they are of opposite types.
                if combined[i][3] != combined[j][3]:
                    # Compare strength (absolute slope).
                    if abs(combined[i][2]) >= abs(combined[j][2]):
                        # Current wedge wins; remove the j-th wedge.
                        combined.pop(j)
                        continue  # re-check current i with next j.
                    else:
                        # Next wedge wins; remove current wedge and break to restart from previous index.
                        combined.pop(i)
                        i = max(i - 1, 0)
                        break
                else:
                    j += 1  # if same type, keep both (or merge if you want)
            else:
                break
        i += 1
    # Separate back into rising and falling lists if needed.
    resolved_rising = [ (s, e, sl) for s,e,sl,t in combined if t=='rising' ]
    resolved_falling = [ (s, e, sl) for s,e,sl,t in combined if t=='falling' ]
    return resolved_rising, resolved_falling

resolved_rising, resolved_falling = resolve_conflicts(merged_rising_wedges, merged_falling_wedges)

print("\nAfter conflict resolution:")
print("Resolved Rising Wedges:")
for i, (start_date, end_date, slope) in enumerate(resolved_rising, 1):
    print(f"{i}: Start = {start_date.date()}, End = {end_date.date()}, Avg Slope = {slope:.5f}")
print("Resolved Falling Wedges:")
for i, (start_date, end_date, slope) in enumerate(resolved_falling, 1):
    print(f"{i}: Start = {start_date.date()}, End = {end_date.date()}, Avg Slope = {slope:.5f}")

# Plot the price data with resolved wedge intervals.
plt.figure(figsize=(14, 8))
plt.plot(df['Date'], prices, label='Close Price', color='blue')

first_label_rising = True
for (start_date, end_date, slope) in resolved_rising:
    label = 'Resolved Rising Wedge' if first_label_rising else None
    plt.axvspan(start_date, end_date, color='orange', alpha=0.3, edgecolor='orange', lw=3, label=label)
    first_label_rising = False

first_label_falling = True
for (start_date, end_date, slope) in resolved_falling:
    label = 'Resolved Falling Wedge' if first_label_falling else None
    plt.axvspan(start_date, end_date, color='purple', alpha=0.3, edgecolor='purple', lw=3, label=label)
    first_label_falling = False

plt.title("Overall Wedge Patterns")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
