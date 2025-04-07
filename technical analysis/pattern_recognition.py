import pandas as pd
import yfinance as yf
import numpy as np
import talib
from talib import MA_Type

# Updated function signature to accept the dataframe as a parameter
def get_pattern_signals(df):
    # Convert columns to numpy arrays with double precision
    open_prices = df['Open'].values.astype(np.double)
    high_prices = df['High'].values.astype(np.double)
    low_prices = df['Low'].values.astype(np.double)
    close_prices = df['Close'].values.astype(np.double)
    volume = df['Volume'].values.astype(np.double)

    # Calculate Hilbert Transform sine components and trend mode.
    sine, leadsine = talib.HT_SINE(close_prices)
    trend_mode = talib.HT_TRENDMODE(close_prices)
    dc_period = talib.HT_DCPERIOD(close_prices)
    dc_phase = talib.HT_DCPHASE(close_prices)

    # Initialize signal array with zeros.
    signal = np.zeros_like(close_prices)
    for i in range(1, len(close_prices)):
        if trend_mode[i] == 0:
            if sine[i - 1] < leadsine[i - 1] and sine[i] > leadsine[i]:
                signal[i] = 1  # Buy signal
            elif sine[i - 1] > leadsine[i - 1] and sine[i] < leadsine[i]:
                signal[i] = -1  # Sell signal

    df['Signal'] = signal
    df['HT_DCPERIOD'] = dc_period
    df['HT_DCPHASE'] = dc_phase

    # Define TA-Lib candlestick pattern functions
    pattern_functions = {
        'CDL2CROWS': talib.CDL2CROWS,
        'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
        'CDL3INSIDE': talib.CDL3INSIDE,
        'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
        'CDL3OUTSIDE': talib.CDL3OUTSIDE,
        'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH,
        'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
        'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
        'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK,
        'CDLBELTHOLD': talib.CDLBELTHOLD,
        'CDLBREAKAWAY': talib.CDLBREAKAWAY,
        'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU,
        'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL,
        'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
        'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
        'CDLDOJI': talib.CDLDOJI,
        'CDLDOJISTAR': talib.CDLDOJISTAR,
        'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
        'CDLENGULFING': talib.CDLENGULFING,
        'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
        'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
        'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE,
        'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
        'CDLHAMMER': talib.CDLHAMMER,
        'CDLHANGINGMAN': talib.CDLHANGINGMAN,
        'CDLHARAMI': talib.CDLHARAMI,
        'CDLHARAMICROSS': talib.CDLHARAMICROSS,
        'CDLHIGHWAVE': talib.CDLHIGHWAVE,
        'CDLHIKKAKE': talib.CDLHIKKAKE,
        'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
        'CDLHOMINGPIGEON': talib.CDLHOMINGPIGEON,
        'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS,
        'CDLINNECK': talib.CDLINNECK,
        'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
        'CDLKICKING': talib.CDLKICKING,
        'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH,
        'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM,
        'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI,
        'CDLLONGLINE': talib.CDLLONGLINE,
        'CDLMARUBOZU': talib.CDLMARUBOZU,
        'CDLMATCHINGLOW': talib.CDLMATCHINGLOW,
        'CDLMATHOLD': talib.CDLMATHOLD,
        'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
        'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
        'CDLONNECK': talib.CDLONNECK,
        'CDLPIERCING': talib.CDLPIERCING,
        'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN,
        'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
        'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
        'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
        'CDLSHORTLINE': talib.CDLSHORTLINE,
        'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
        'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN,
        'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH,
        'CDLTAKURI': talib.CDLTAKURI,
        'CDLTASUKIGAP': talib.CDLTASUKIGAP,
        'CDLTHRUSTING': talib.CDLTHRUSTING,
        'CDLTRISTAR': talib.CDLTRISTAR,
        'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER,
        'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS,
        'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS
    }

    # Apply each candlestick pattern function and normalize the output
    for name, func in pattern_functions.items():
        df[name] = func(open_prices, high_prices, low_prices, close_prices)

    def normalize_signal(val):
        if val == 200:
            return 1.0
        elif val == 100:
            return 0.8
        elif val == 80:
            return 0.5
        elif val == -80:
            return -0.5
        elif val == -100:
            return -0.8
        elif val == -200:
            return -1.0
        else:
            return 0.0

    for col in pattern_functions.keys():
        df[col] = df[col].apply(normalize_signal)
        
    # Collect all technical indicators into a dictionary
    tech_indicators = {
        'TRANGE': talib.TRANGE(high_prices, low_prices, close_prices),
        'ADX': talib.ADX(high_prices, low_prices, close_prices, timeperiod=14),
        'ADXR': talib.ADXR(high_prices, low_prices, close_prices, timeperiod=14),
        'APO': talib.APO(close_prices, fastperiod=12, slowperiod=26, matype=MA_Type.SMA),
        'AROON_UP': talib.AROON(high_prices, low_prices, timeperiod=14)[0],
        'AROON_DOWN': talib.AROON(high_prices, low_prices, timeperiod=14)[1],
        'AROONOSC': talib.AROONOSC(high_prices, low_prices, timeperiod=14),
        'BOP': talib.BOP(open_prices, high_prices, low_prices, close_prices),
        'CCI': talib.CCI(high_prices, low_prices, close_prices, timeperiod=14),
        'CMO': talib.CMO(close_prices, timeperiod=14),
        'DX': talib.DX(high_prices, low_prices, close_prices, timeperiod=14),
        'MACDEXT': talib.MACDEXT(
                        close_prices,
                        fastperiod=12, fastmatype=MA_Type.SMA,
                        slowperiod=26, slowmatype=MA_Type.SMA,
                        signalperiod=9, signalmatype=MA_Type.SMA)[0],
        'MACDEXT_signal': talib.MACDEXT(
                        close_prices,
                        fastperiod=12, fastmatype=MA_Type.SMA,
                        slowperiod=26, slowmatype=MA_Type.SMA,
                        signalperiod=9, signalmatype=MA_Type.SMA)[1],
        'MACDEXT_hist': talib.MACDEXT(
                        close_prices,
                        fastperiod=12, fastmatype=MA_Type.SMA,
                        slowperiod=26, slowmatype=MA_Type.SMA,
                        signalperiod=9, signalmatype=MA_Type.SMA)[2],
        'MACDFIX': talib.MACDFIX(close_prices, signalperiod=9)[0],
        'MACDFIX_hist': talib.MACDFIX(close_prices, signalperiod=9)[2],
        'MFI': talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14),
        'MINUS_DI': talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14),
        'MINUS_DM': talib.MINUS_DM(high_prices, low_prices, timeperiod=14),
        'PLUS_DI': talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14),
        'PLUS_DM': talib.PLUS_DM(high_prices, low_prices, timeperiod=14),
        'PPO': talib.PPO(close_prices, fastperiod=12, slowperiod=26, matype=MA_Type.SMA),
        'ROC': talib.ROC(close_prices, timeperiod=10),
        'ROCP': talib.ROCP(close_prices, timeperiod=10),
        'ROCR': talib.ROCR(close_prices, timeperiod=10),
        'ROCR100': talib.ROCR100(close_prices, timeperiod=10),
        'STOCHF_fastk': talib.STOCHF(
                            high_prices, low_prices, close_prices,
                            fastk_period=14, fastd_period=3, fastd_matype=MA_Type.SMA)[0],
        'STOCHF_fastd': talib.STOCHF(
                            high_prices, low_prices, close_prices,
                            fastk_period=14, fastd_period=3, fastd_matype=MA_Type.SMA)[1],
        'STOCHRSI_fastk': talib.STOCHRSI(
                            close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=MA_Type.SMA)[0],
        'STOCHRSI_fastd': talib.STOCHRSI(
                            close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=MA_Type.SMA)[1],
        'TRIX': talib.TRIX(close_prices, timeperiod=30),
        'ULTOSC': talib.ULTOSC(high_prices, low_prices, close_prices,
                               timeperiod1=7, timeperiod2=14, timeperiod3=28),
        'WILLR': talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
    }
    tech_df = pd.DataFrame(tech_indicators, index=df.index)
    
    df['Overall Pattern Signal'] = df[pattern_functions.keys()].sum(axis=1)
    df['Overall Pattern Signal'] += df['Signal']
    max_signal = df['Overall Pattern Signal'].max()
    min_signal = df['Overall Pattern Signal'].min()
    df['Overall Pattern Signal'] = 2 * (df['Overall Pattern Signal'] - min_signal) / (max_signal - min_signal) - 1
    
    df = pd.concat([df, tech_df], axis=1)
    df.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume', 'Price_Change', 'Target'], inplace=True)
    df.to_csv('_data.csv', index=False)
    return df