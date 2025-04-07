import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import yfinance as yf
import matplotlib.lines as mlines
pd.set_option('future.no_silent_downcasting', True)

def detect_patterns(prices):
    # Detect double tops.
    def detect_double_tops(prices, distance=20, prominence=0.01, tolerance=0.02, min_drop=0.03):
        peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
        double_tops = []
        for i in range(len(peaks) - 1):
            first_peak = peaks[i]
            second_peak = peaks[i + 1]
            if abs(prices[first_peak] - prices[second_peak]) / prices[first_peak] < tolerance:
                trough_index = np.argmin(prices[first_peak:second_peak]) + first_peak
                if (prices[first_peak] - prices[trough_index]) / prices[first_peak] > min_drop:
                    double_tops.append((first_peak, second_peak, trough_index))
        return double_tops

    # Detect double bottoms.
    def detect_double_bottoms(prices, distance=20, prominence=0.01, tolerance=0.02, min_rise=0.03):
        troughs, _ = find_peaks(-prices, distance=distance, prominence=prominence)
        double_bottoms = []
        for i in range(len(troughs) - 1):
            first_trough = troughs[i]
            second_trough = troughs[i + 1]
            if abs(prices[first_trough] - prices[second_trough]) / prices[first_trough] < tolerance:
                peak_index = np.argmax(prices[first_trough:second_trough]) + first_trough
                if (prices[peak_index] - prices[first_trough]) / prices[peak_index] > min_rise:
                    double_bottoms.append((first_trough, second_trough, peak_index))
        return double_bottoms

    # Detect triple tops.
    def detect_triple_tops(prices, distance=20, prominence=0.01, tolerance=0.05, min_drop=0.03):
        peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
        triple_tops = []
        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
            if (abs(prices[p1] - prices[p2]) / prices[p1] < tolerance and 
                abs(prices[p2] - prices[p3]) / prices[p2] < tolerance):
                trough1 = np.argmin(prices[p1:p2]) + p1
                trough2 = np.argmin(prices[p2:p3]) + p2
                if ((prices[p1] - prices[trough1]) / prices[p1] > min_drop and
                    (prices[p2] - prices[trough2]) / prices[p2] > min_drop):
                    triple_tops.append((p1, p2, p3, trough1, trough2))
        return triple_tops

    # Detect triple bottoms.
    def detect_triple_bottoms(prices, distance=20, prominence=0.01, tolerance=0.05, min_rise=0.03):
        troughs, _ = find_peaks(-prices, distance=distance, prominence=prominence)
        triple_bottoms = []
        for i in range(len(troughs) - 2):
            t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
            if (abs(prices[t1] - prices[t2]) / prices[t1] < tolerance and 
                abs(prices[t2] - prices[t3]) / prices[t2] < tolerance):
                peak1 = np.argmax(prices[t1:t2]) + t1
                peak2 = np.argmax(prices[t2:t3]) + t2
                if ((prices[peak1] - prices[t1]) / prices[peak1] > min_rise and
                    (prices[peak2] - prices[t2]) / prices[peak2] > min_rise):
                    triple_bottoms.append((t1, t2, t3, peak1, peak2))
        return triple_bottoms

    # Detect head and shoulders.
    def detect_head_and_shoulders(prices, distance=20, prominence=0.01, shoulder_tolerance=0.05, min_drop=0.03):
        peaks, _ = find_peaks(prices, distance=distance, prominence=prominence)
        patterns = []
        for i in range(len(peaks) - 2):
            left, head, right = peaks[i], peaks[i+1], peaks[i+2]
            if prices[head] > prices[left] and prices[head] > prices[right]:
                if abs(prices[left] - prices[right]) / prices[left] < shoulder_tolerance:
                    trough1 = np.argmin(prices[left:head]) + left
                    trough2 = np.argmin(prices[head:right]) + head
                    if ((prices[left] - prices[trough1]) / prices[left] > min_drop and 
                        (prices[right] - prices[trough2]) / prices[right] > min_drop):
                        patterns.append((left, head, right, trough1, trough2))
        return patterns

    
    patterns = {
        "double_tops": detect_double_tops(prices),
        "double_bottoms": detect_double_bottoms(prices),
        "triple_tops": detect_triple_tops(prices),
        "triple_bottoms": detect_triple_bottoms(prices),
        "head_and_shoulders": detect_head_and_shoulders(prices)
    }
    return patterns
    


def plot_all_patterns(df, prices, double_top_patterns, double_bottoms_patterns, 
                      triple_tops, triple_bottoms, detect_head_and_shoulders):
    # Plot Double Top patterns along with the price series.
    plt.figure(figsize=(14,7))
    plt.plot(df['Date'], prices, label='Close Price', color='blue')
    for first, second, trough in double_top_patterns:
        plt.plot(df['Date'].iloc[first], prices[first], 'ro')   # First peak
        plt.plot(df['Date'].iloc[second], prices[second], 'ro')  # Second peak
        plt.plot(df['Date'].iloc[trough], prices[trough], 'go')    # Trough
    
    # Create custom legend handles.
    peak_handle = mlines.Line2D([], [], color='red', marker='o',
                                linestyle='None', markersize=8, label='Peak')
    trough_handle = mlines.Line2D([], [], color='green', marker='o',
                                  linestyle='None', markersize=8, label='Trough')
    
    plt.legend(handles=[peak_handle, trough_handle])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Detected Double Tops")
    plt.show()
    
    # Plot Double Bottom patterns along with the price series.
    plt.figure(figsize=(14,7))
    plt.plot(df['Date'], prices, color='blue')  # Price series
    
    for first, second, trough in double_bottoms_patterns:
        plt.plot(df['Date'].iloc[first], prices[first], 'ro')   # First peak
        plt.plot(df['Date'].iloc[second], prices[second], 'ro')  # Second peak
        plt.plot(df['Date'].iloc[trough], prices[trough], 'go')   # Trough
    
    close_line = mlines.Line2D([], [], color='blue', label='Close Price')
    trough_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None', 
                                  markersize=8, label='Peaks')
    peak_marker = mlines.Line2D([], [], color='green', marker='o', linestyle='None', 
                                markersize=8, label='Trough')
    
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Detected Double Bottoms")
    plt.legend(handles=[close_line, peak_marker, trough_marker])
    plt.show()
    
    # Plot triple tops.
    plt.figure(figsize=(14,7))
    plt.plot(df['Date'], prices, label='Close Price', color='blue')
    first_tt = True
    for pattern in triple_tops:
        p1, p2, p3, trough1, trough2 = pattern
        if first_tt:
            plt.plot(df['Date'].iloc[p1], prices[p1], 'ro', label='Triple Top Peak')
            plt.plot(df['Date'].iloc[p2], prices[p2], 'ro')
            plt.plot(df['Date'].iloc[p3], prices[p3], 'ro')
            plt.plot(df['Date'].iloc[trough1], prices[trough1], 'kx', label='Triple Top Trough')
            plt.plot(df['Date'].iloc[trough2], prices[trough2], 'kx')
            first_tt = False
        else:
            plt.plot(df['Date'].iloc[p1], prices[p1], 'ro')
            plt.plot(df['Date'].iloc[p2], prices[p2], 'ro')
            plt.plot(df['Date'].iloc[p3], prices[p3], 'ro')
            plt.plot(df['Date'].iloc[trough1], prices[trough1], 'kx')
            plt.plot(df['Date'].iloc[trough2], prices[trough2], 'kx')
    plt.title("Detected Triple Tops")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    # Plot triple bottoms.
    plt.figure(figsize=(14,7))
    plt.plot(df['Date'], prices, label='Close Price', color='blue')
    first_tb = True
    for pattern in triple_bottoms:
        t1, t2, t3, peak1, peak2 = pattern
        if first_tb:
            plt.plot(df['Date'].iloc[t1], prices[t1], 'go', label='Triple Bottom Trough')
            plt.plot(df['Date'].iloc[t2], prices[t2], 'go')
            plt.plot(df['Date'].iloc[t3], prices[t3], 'go')
            plt.plot(df['Date'].iloc[peak1], prices[peak1], 'm^', label='Triple Bottom Peak')
            plt.plot(df['Date'].iloc[peak2], prices[peak2], 'm^')
            first_tb = False
        else:
            plt.plot(df['Date'].iloc[t1], prices[t1], 'go')
            plt.plot(df['Date'].iloc[t2], prices[t2], 'go')
            plt.plot(df['Date'].iloc[t3], prices[t3], 'go')
            plt.plot(df['Date'].iloc[peak1], prices[peak1], 'm^')
            plt.plot(df['Date'].iloc[peak2], prices[peak2], 'm^')
    plt.title("Detected Triple Bottoms")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    # Detect head-and-shoulders patterns using the Close prices.
    hs_patterns = detect_head_and_shoulders(prices)
    
    plt.figure(figsize=(14,7))
    plt.plot(df['Date'], prices, label='Close Price', color='blue')
    
    if hs_patterns:
        # For the first pattern, label each point only once.
        for pattern in hs_patterns:
            left, head, right, trough1, trough2 = pattern
            plt.plot(df['Date'].iloc[left], prices[left], 'ro', 
                     label='Left Shoulder' if pattern == hs_patterns[0] else "")
            plt.plot(df['Date'].iloc[head], prices[head], 'ko',
                     label='Head' if pattern == hs_patterns[0] else "")
            plt.plot(df['Date'].iloc[right], prices[right], 'ro', 
                     label='Right Shoulder' if pattern == hs_patterns[0] else "")
            plt.plot(df['Date'].iloc[trough1], prices[trough1], 'go',
                     label='Trough' if pattern == hs_patterns[0] else "")
            plt.plot(df['Date'].iloc[trough2], prices[trough2], 'go',
                     label='Trough' if pattern == hs_patterns[0] else "")
    else:
        print("No head-and-shoulders pattern detected.")
        
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Detected Head-and-Shoulders Patterns")
    
    # Remove duplicate legend entries.
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.show()

# GET TICKER AND DOWNLOAD DATA VIA YFINANCE
# ticker = input("Ticker Symbol: ").upper()
# df = yf.download(ticker, period='max', auto_adjust=False)
# df.reset_index(inplace=True)
# # Use the 'Close' price for pattern detection.
# # Ensure the 'Close' prices are a one-dimensional NumPy array and drop NaNs.
# prices = df['Close'].to_numpy().flatten()
# prices = prices[~np.isnan(prices)]

# Plot all the detected patterns.
# For the head-and-shoulders plot, we pass a lambda that retrieves the
# patterns from detect_patterns (it could also be passed directly if refactored).
# plot_all_patterns(df, prices,
#                   patterns["double_tops"],
#                   patterns["double_bottoms"],
#                   patterns["triple_tops"],
#                   patterns["triple_bottoms"],
#                   lambda p: detect_patterns(p)["head_and_shoulders"])

# ######################################################
# ## Wedges
# ######################################################
def process_wedges(prices, dates, window_size=50, peak_prominence=0.01, slope_threshold=0.0001, plot=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    # Local function to detect wedges based on type.
    def detect_wedges(prices, dates, window_size, wedge_type, peak_prominence, slope_threshold):
        wedges = []
        n = len(prices)
        for start in range(0, n - window_size, window_size // 2):
            end = start + window_size
            # Check that the window covers at least one day.
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

    # Local function to merge overlapping intervals.
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
                # Weighted average slope based on duration.
                duration_last = (last[1] - last[0]).days
                duration_current = (current[1] - current[0]).days
                total_duration = duration_last + duration_current
                new_slope = (last[2] * duration_last + current[2] * duration_current) / total_duration if total_duration != 0 else last[2]
                merged[-1] = (new_start, new_end, new_slope)
            else:
                merged.append(current)
        return merged

    # Local function to resolve overlapping rising and falling wedge intervals by comparing slope strength.
    def resolve_conflicts(rising_intervals, falling_intervals):
        combined = []
        for (start, end, slope) in rising_intervals:
            combined.append((start, end, slope, 'rising'))
        for (start, end, slope) in falling_intervals:
            combined.append((start, end, slope, 'falling'))
        # Sort by start date.
        combined.sort(key=lambda x: x[0])
        i = 0
        while i < len(combined):
            j = i + 1
            while j < len(combined):
                # If the next interval overlaps.
                if combined[j][0] <= combined[i][1]:
                    # Resolve only if they are opposite types.
                    if combined[i][3] != combined[j][3]:
                        if abs(combined[i][2]) >= abs(combined[j][2]):
                            # Keep the current, remove the overlapping one.
                            combined.pop(j)
                            continue  # re-check at the same i.
                        else:
                            combined.pop(i)
                            i = max(i - 1, 0)
                            break
                    else:
                        j += 1
                else:
                    break
            i += 1
        resolved_rising = [(s, e, sl) for s, e, sl, t in combined if t == 'rising']
        resolved_falling = [(s, e, sl) for s, e, sl, t in combined if t == 'falling']
        return resolved_rising, resolved_falling

    # Detect rising and falling wedges.
    rising_wedges = detect_wedges(prices, dates, window_size, 'rising', peak_prominence, slope_threshold)
    falling_wedges = detect_wedges(prices, dates, window_size, 'falling', peak_prominence, slope_threshold)

    # Build interval lists without plotting.
    rising_wedges_intervals = []
    falling_wedges_intervals = []

    for i, (start, end, slope, indices) in enumerate(rising_wedges, 1):
        start_date = dates.iloc[start]
        end_date = dates.iloc[end - 1]
        rising_wedges_intervals.append((start_date, end_date, slope))

    for i, (start, end, slope, indices) in enumerate(falling_wedges, 1):
        start_date = dates.iloc[start]
        end_date = dates.iloc[end - 1]
        falling_wedges_intervals.append((start_date, end_date, slope))

    # Merge overlapping intervals.
    merged_rising = merge_intervals(rising_wedges_intervals)
    merged_falling = merge_intervals(falling_wedges_intervals)

    # Resolve conflicts between rising and falling wedges.
    resolved_rising, resolved_falling = resolve_conflicts(merged_rising, merged_falling)
    return resolved_rising, resolved_falling

# Function to detect closing price fractals
def detect_fractals(data):
    bullish = []
    bearish = []
    for i in range(2, len(data) - 2):
        close_window = data['Close'].iloc[i - 2:i + 3].to_numpy()
        mid_close = close_window[2].item()
        next_close = data['Close'].iloc[i + 1].item()
        if mid_close == np.min(close_window) and np.sum(close_window == mid_close) == 1 and next_close > mid_close:
            bullish.append(i)
        if mid_close == np.max(close_window) and np.sum(close_window == mid_close) == 1 and next_close < mid_close:
            bearish.append(i)
    return bullish, bearish

def do_bollinger_squeeze(ticker, period='max'):
    # Ask user for ticker symbol and download data.
    ticker = ticker.upper()
    # Explicitly set auto_adjust=False (default now is True in yfinance)
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    df = df.reset_index()

    # Extract 'Close', 'Date', and 'Volume'.
    prices = df['Close'].values
    dates = pd.to_datetime(df['Date'])
    volumes = df['Volume'].values
    # Ensure 'Date' is set as the index.
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
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values

    df['RSI'] = rsi(df['Close'], period=14)
    # Compute a simple Volume Moving Average (20-day)
    df['Vol_MA'] = df['Volume'].rolling(window=20).mean()

    # --- Identify Breakout Signals ---
    # A bullish breakout occurs when the price crosses above the UpperBand after a squeeze.
    # A bearish breakout occurs when the price crosses below the LowerBand after a squeeze.
    close_series = pd.to_numeric(df['Close'].squeeze(), errors='coerce')
    upper_series = pd.to_numeric(df['UpperBand'].squeeze(), errors='coerce')
    lower_series = pd.to_numeric(df['LowerBand'].squeeze(), errors='coerce')
    volma_series = pd.to_numeric(df['Vol_MA'].squeeze(), errors='coerce')
    volume_series = df['Volume'].squeeze()
    
    squeeze_shift = df['Squeeze'].shift(1).fillna(False).infer_objects(copy=False)

    df['BreakoutUp'] = (
        (close_series > upper_series) &
        squeeze_shift &
        (df['RSI'] > 50) &
        (volume_series > 1.2 * volma_series)
    )

    df['BreakoutDown'] = (
        (close_series < lower_series) &
        squeeze_shift &
        (df['RSI'] < 50) &
        (volume_series > 1.2 * volma_series)
    )

    close_after = close_series.shift(-10)

    df['ConfirmedBreakoutUp'] = df['BreakoutUp'] & (close_series < close_after)
    df['ConfirmedBreakoutDown'] = df['BreakoutDown'] & (close_series > close_after)

    df['ConfirmedBreakoutUp'] = df['ConfirmedBreakoutUp'].fillna(False)
    df['ConfirmedBreakoutDown'] = df['ConfirmedBreakoutDown'].fillna(False)
    # Replace boolean values with 1s and 0s.
    df = df.replace({True: 1, False: 0})
    

    # # --- Plotting Bollinger Squeeze and Breakouts ---
    # plt.figure(figsize=(14, 8))
    # plt.plot(df.index, df['Close'], label='Close Price', color='lightblue')
    # plt.scatter(df.loc[df['ConfirmedBreakoutUp']].index, 
    #             df.loc[df['ConfirmedBreakoutUp'], 'Close'],
    #             color='green', marker='^', s=100, label='Breakout Up')
    # plt.scatter(df.loc[df['ConfirmedBreakoutDown']].index, 
    #             df.loc[df['ConfirmedBreakoutDown'], 'Close'],
    #             color='red', marker='v', s=100, label='Breakout Down')
    # plt.title(f"Bollinger Squeeze for {ticker}")
    # plt.xlabel("Date")
    # plt.ylabel("Price")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return df


