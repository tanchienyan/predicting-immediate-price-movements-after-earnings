import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import yfinance as yf

def detect_triangle_patterns(prices, window_size, pattern_type='ascending', 
                               base_prominence=0.01, slope_threshold=0.004, 
                               std_factor=0.2, min_price_range_ratio=0.08, 
                               convergence_ratio=0.01):
    """
    Detect triangle patterns in a price series using a sliding window.
    
    Parameters:
      prices: 1D numpy array of prices.
      window_size: Number of data points in each sliding window.
      pattern_type: 'ascending', 'descending', or 'symmetrical'
      base_prominence: Factor to scale the prominence used for peak detection.
      slope_threshold: Maximum (or minimum) slope to consider a line nearly horizontal.
      std_factor: (Not used explicitly in this simple version)
      min_price_range_ratio: Minimum required range relative to mean price.
      convergence_ratio: Maximum allowed gap (relative to price range) between trendlines at midpoint.
      
    Returns:
      A list of tuples for each detected pattern:
         (start_index, end_index, peak_indices, trough_indices, resistance_slope, support_slope)
    """
    patterns = []
    n = len(prices)
    # Use a step smaller than window_size to have overlapping windows
    step = window_size // 2  
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = prices[start:end]
        
        # Ensure sufficient price range
        price_range = np.max(window) - np.min(window)
        if price_range < min_price_range_ratio * np.mean(window):
            continue
        
        # Detect peaks and troughs with prominence scaled by the price range.
        peak_indices, _ = find_peaks(window, prominence=base_prominence * price_range)
        trough_indices, _ = find_peaks(-window, prominence=base_prominence * price_range)
        
        # Need at least two peaks and two troughs to form trendlines
        if len(peak_indices) < 2 or len(trough_indices) < 2:
            continue
        
        # Fit linear regression for resistance using peaks
        x_peaks = peak_indices
        y_peaks = window[peak_indices]
        res_coeffs = np.polyfit(x_peaks, y_peaks, 1)  # slope, intercept
        res_slope = res_coeffs[0]
        
        # Fit linear regression for support using troughs
        x_troughs = trough_indices
        y_troughs = window[trough_indices]
        sup_coeffs = np.polyfit(x_troughs, y_troughs, 1)
        sup_slope = sup_coeffs[0]
        
        # Check convergence at the midpoint of the window
        mid = window_size // 2
        res_val_mid = res_coeffs[0] * mid + res_coeffs[1]
        sup_val_mid = sup_coeffs[0] * mid + sup_coeffs[1]
        gap = res_val_mid - sup_val_mid
        if gap > convergence_ratio * price_range:
            continue

        # Now check pattern-specific conditions:
        if pattern_type == 'ascending':
            # Ascending: support (troughs) should be rising and resistance nearly flat.
            if sup_slope < slope_threshold or abs(res_slope) > slope_threshold:
                continue
        elif pattern_type == 'descending':
            # Descending: resistance (peaks) should be falling and support nearly flat.
            if res_slope > -slope_threshold or abs(sup_slope) > slope_threshold:
                continue
        elif pattern_type == 'symmetrical':
            # Symmetrical: both lines must be non-horizontal with similar magnitude slopes (ideally opposite)
            if not (abs(res_slope) > slope_threshold and abs(sup_slope) > slope_threshold):
                continue
            if abs(res_slope + sup_slope) > slope_threshold:
                continue
        else:
            # Unknown pattern type
            continue
        
        # If conditions met, record the pattern
        patterns.append((start, end, peak_indices, trough_indices, res_slope, sup_slope))
    
    return patterns

# ------------------ Main Script ------------------

# Download historical data via yfinance
ticker = input("Ticker Symbol: ").upper()
df = yf.download(ticker, period='max')
df.reset_index(inplace=True)
prices = np.ravel(df['Close'].values)

# Smooth the price series to reduce noise.
window_smooth = 5
smoothed_prices = pd.Series(prices).rolling(window=window_smooth, min_periods=1, center=True).mean().values

# Use relaxed settings for easier detection.
desc_patterns = detect_triangle_patterns(
    smoothed_prices,
    window_size=120,
    pattern_type='descending',
    base_prominence=0.003,         # Lowered from 0.005
    slope_threshold=0.002,         # Lowered from 0.003
    std_factor=0.15,               # Lowered from 0.2
    min_price_range_ratio=0.03,    # Lowered from 0.05
    convergence_ratio=0.01         # Lowered from 0.02
)

sym_patterns = detect_triangle_patterns(
    smoothed_prices,
    window_size=120,
    pattern_type='symmetrical',
    base_prominence=0.003,         # Lowered from 0.005
    slope_threshold=0.002,         # Lowered from 0.003
    std_factor=0.15,               # Lowered from 0.2
    min_price_range_ratio=0.03,    # Lowered from 0.05
    convergence_ratio=0.01         # Lowered from 0.02
)

asc_patterns = detect_triangle_patterns(
    smoothed_prices,
    window_size=120,
    pattern_type='ascending',      # Added ascending pattern detection
    base_prominence=0.003,         # Lowered from 0.005
    slope_threshold=0.002,         # Lowered from 0.003
    std_factor=0.15,               # Lowered from 0.2
    min_price_range_ratio=0.03,    # Lowered from 0.05
    convergence_ratio=0.01         # Lowered from 0.02
)

# Plotting all detected patterns on one chart.
plt.figure(figsize=(14, 8))
plt.plot(df['Date'], prices, color='blue', label='Close Price')

# Plot ascending triangle patterns
first_asc = True
for pattern in asc_patterns:
    # [Add plotting logic for ascending triangle pattern]
    # Example: Mark the pattern's key points.
    x_coords = [df['Date'].iloc[idx] for idx in pattern['indices']]
    y_coords = [prices[idx] for idx in pattern['indices']]
    if first_asc:
        plt.plot(x_coords, y_coords, 'g-', lw=2, label='Ascending Triangle')
        first_asc = False
    else:
        plt.plot(x_coords, y_coords, 'g-', lw=2)

# Plot descending triangle patterns.
first_desc = True
for pattern in desc_patterns:
    x_coords = [df['Date'].iloc[idx] for idx in pattern['indices']]
    y_coords = [prices[idx] for idx in pattern['indices']]
    if first_desc:
        plt.plot(x_coords, y_coords, 'r-', lw=2, label='Descending Triangle')
        first_desc = False
    else:
        plt.plot(x_coords, y_coords, 'r-', lw=2)

# Plot symmetrical triangle patterns.
first_sym = True
for pattern in sym_patterns:
    x_coords = [df['Date'].iloc[idx] for idx in pattern['indices']]
    y_coords = [prices[idx] for idx in pattern['indices']]
    if first_sym:
        plt.plot(x_coords, y_coords, 'm-', lw=2, label='Symmetrical Triangle')
        first_sym = False
    else:
        plt.plot(x_coords, y_coords, 'm-', lw=2)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Detected Triangle Patterns")
plt.legend()
plt.show()