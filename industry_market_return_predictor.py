import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define sector tickers (adjust these lists as needed)
sector_tickers = {
    "ENERGY": [
        "XLE",                   # SPDR Energy ETF
        "^SP500-10",             # S&P Energy Index 
    ],
    "MATERIALS": [
        "XLB",                   # SPDR Materials ETF
        "^SP500-15",             # S&P Materials Index (assumed)
    ],
    "INDUSTRIALS": [
        "XLI",                   # SPDR Industrials ETF
        "^SP500-20",             # S&P Industrials Index (assumed)
    ],
    "CONSUMER DISCRETIONARY": [
        "XLY",                   # SPDR Consumer Discretionary ETF
        "^SP500-25",             # S&P Consumer Discretionary Index (assumed)
    ],
    "CONSUMER STAPLES": [
        "XLP",                   # SPDR Consumer Staples ETF
        "^SP500-30",             # S&P Consumer Staples Index (assumed)
    ],
    "HEALTH": [
        "XLV",                   # SPDR Health Care ETF
        "^SP500-35",             # S&P Health Care Index (assumed)
        "^NBI"                   # Nasdaq Biotechnology Index (as a proxy)
    ],
    "FINANCIALS": [
        "XLF",                   # SPDR Financials ETF
        "BKX",                   # KBW Bank Index ETF
        "^SP500-40",             # S&P Financials Index (assumed)
    ],
    "TECH": [
        "XLK",                   # SPDR Information Technology ETF
        "^SP500-45",             # S&P Information Technology Index 
    ],
    "COMMUNICATION SERVICES": [
        "XLC",                   # SPDR Communication Services ETF
        "^SP500-50",             # S&P Communication Services Index 
    ],
    "UTILITIES": [
        "XLU",                   # SPDR Utilities ETF
        "^SP500-55",             # S&P Utilities Index (assumed)
    ],
    "REAL ESTATE": [
        "XLRE",                  # SPDR Real Estate ETF
        "^SP500-60",             # S&P Real Estate Index (assumed)
    ]
}


def get_industry_expected_return(period_days):
    """
    Prompts the user for an industry (GICS sector) and then calculates 
    the average predicted expected return for that sector using multiple regression models,
    using the provided period_days as the return period.
    
    For return periods ≤ 63 days, five models are used with weights:
        - Window model: 12.5%
        - Overall model: 12.5%
        - 2-year model: 15%
        - 1-year model: 40%
        - Specific period model: 20%
    
    For return periods > 63 days, the 1‑year and 2‑year models are skipped and new weights are applied:
        - Window model: 40%
        - Overall model: 40%
        - Specific period model: 20%
    
    Args:
        period_days (int): The number of days to use as the return period.
    
    Returns:
        float: Average predicted expected return (as a percentage). 
               (Convert to a decimal in your main script if desired, e.g. divide by 100.)
        Returns None if no predictions could be computed.
    """
    # Get industry input from the user
    gics_sector_input = input("Enter Industry: ").upper()

    # Normalize synonyms to a standard key
    if gics_sector_input in ["TECH", "INFORMATION TECHNOLOGY", "TECHNOLOGY"]:
        group_key = "TECH"
    elif gics_sector_input in ["HEALTH", "HEALTHCARE", "HEALTH CARE"]:
        group_key = "HEALTH"
    else:
        group_key = gics_sector_input

    if group_key not in sector_tickers:
        print("Invalid GICS sector entered.")
        return None

    tickers = sector_tickers[group_key]
    predicted = []
    
    print(f"\nCalculating expected returns for the {group_key} sector using a period of {period_days} days:")
    
    for ticker in tickers:
        print(f"\nProcessing ticker: {ticker}")
        data = yf.download(ticker, period="max", interval="1d")
        data_window = yf.download(ticker, period="max", interval="1d")
        data_specific = data.tail(period_days)
        
        if data.empty:
            print(f"No data found for ticker {ticker}. Skipping...")
            continue
        
        # Prepare overall data
        data["Log Return"] = np.log(data["Close"] / data["Close"].shift(1))
        data["Simple Return (Y)"] = np.exp(data["Log Return"]) - 1
        data = data.dropna().reset_index(drop=True)
        if data.empty:
            print(f"Not enough data for ticker {ticker} to compute returns. Skipping...")
            continue

        # Part 1: Window data
        data_window["Log Return"] = np.log(data_window["Close"] / data_window["Close"].shift(1))
        data_window["Cumulative Log Return"] = data_window["Log Return"].rolling(window=period_days).sum()
        data_window["Simple Return (Y)"] = np.exp(data_window["Cumulative Log Return"]) - 1
        data_window = data_window.dropna().reset_index(drop=True)
    
        # Prepare specific period data
        data_specific["Log Return"] = np.log(data_specific["Close"] / data_specific["Close"].shift(1))
        data_specific = data_specific.dropna().reset_index(drop=True)
        
        # For period_days ≤ 63, download and process 1-year and 2-year data
        if period_days <= 63:
            data_1_year = yf.download(ticker, period="1y", interval="1d")
            data_2_year = yf.download(ticker, period="2y", interval="1d")
            if data_1_year.empty or data_2_year.empty:
                print(f"Not enough 1-year or 2-year data for ticker {ticker}. Skipping ticker.")
                continue

            data_1_year["Log Return"] = np.log(data_1_year["Close"] / data_1_year["Close"].shift(1))
            data_1_year["Cumulative Log Return"] = data_1_year["Log Return"].rolling(window=period_days).sum()
            data_1_year["Simple Return (Y)"] = np.exp(data_1_year["Cumulative Log Return"]) - 1
            data_1_year = data_1_year.dropna().reset_index(drop=True)

            data_2_year["Log Return"] = np.log(data_2_year["Close"] / data_2_year["Close"].shift(1))
            data_2_year["Cumulative Log Return"] = data_2_year["Log Return"].rolling(window=period_days).sum()
            data_2_year["Simple Return (Y)"] = np.exp(data_2_year["Cumulative Log Return"]) - 1
            data_2_year = data_2_year.dropna().reset_index(drop=True)
        
        # Create time indices for regression
        data_window["Time Index"] = np.arange(len(data_window)) / period_days
        data_specific["Time Index"] = np.arange(len(data_specific))
        data["Time Index"] = np.arange(len(data))
        
        # Prepare regression data
        X_window = data_window["Time Index"].values.reshape(-1, 1)
        y_window = data_window["Simple Return (Y)"].values * 100  # expressed as percentage

        X_specific = data_specific["Time Index"].values.reshape(-1, 1)
        y_specific = data_specific["Log Return"].values 

        X = data["Time Index"].values.reshape(-1, 1)
        y = data["Log Return"].values 
        
        # Train and predict using window model
        model_window = LinearRegression()
        model_window.fit(X_window, y_window)
        predicted_returns_window = model_window.predict(np.array([[X_window[-1][0] + 1]]))[0]
        
        # Train and predict using specific period model
        model_specific = LinearRegression()
        model_specific.fit(X_specific, y_specific)
        predicted_returns_specific = model_specific.predict(np.array([[X_specific[-1][0] + 1]]))[0]
        predicted_returns_specific = (np.exp(predicted_returns_specific * period_days) - 1) * 100
        
        # Train and predict using overall model
        model_overall = LinearRegression()
        model_overall.fit(X, y)
        predicted_returns = model_overall.predict(np.array([[X[-1][0] + 1]]))[0]
        predicted_returns = (np.exp(predicted_returns * period_days) - 1) * 100
        
        if period_days <= 63:
            # Create time indices for 1-year and 2-year data
            data_1_year["Time Index"] = np.arange(len(data_1_year)) / period_days
            data_2_year["Time Index"] = np.arange(len(data_2_year)) / period_days

            X_1_year = data_1_year["Time Index"].values.reshape(-1, 1)
            y_1_year = data_1_year["Simple Return (Y)"].values * 100

            X_2_year = data_2_year["Time Index"].values.reshape(-1, 1)
            y_2_year = data_2_year["Simple Return (Y)"].values * 100
            
            # Train 1-year model
            model_1_year = LinearRegression()
            model_1_year.fit(X_1_year, y_1_year)
            predicted_returns_1_year = model_1_year.predict(np.array([[X_1_year[-1][0] + 1]]))[0]
            
            # Train 2-year model
            model_2_year = LinearRegression()
            model_2_year.fit(X_2_year, y_2_year)
            predicted_returns_2_year = model_2_year.predict(np.array([[X_2_year[-1][0] + 1]]))[0]
            
            # Compute final predicted return using all 5 models with weights:
            # Window: 12.5%, Overall: 12.5%, 2-year: 15%, 1-year: 40%, Specific: 20%
            final_pred = (0.125 * predicted_returns_window) + (0.125 * predicted_returns) + \
                         (0.15 * predicted_returns_2_year) + (0.4 * predicted_returns_1_year) + \
                         (0.2 * predicted_returns_specific)
        else:
            # For period_days > 63, skip the 1-year and 2-year models.
            # Use new weights: Window: 40%, Overall: 40%, Specific: 20%
            final_pred = (0.4 * predicted_returns_window) + (0.4 * predicted_returns) + \
                         (0.2 * predicted_returns_specific)
            
        predicted.append(final_pred)
    
    if predicted:
        avg_return = np.mean(predicted)
        return avg_return  # Return as a percentage (convert to decimal later if needed)
    else:
        return None
    
