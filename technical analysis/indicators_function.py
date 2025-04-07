import talib
from talib import MA_Type
import numpy as np

def calculate_indicators(df):
    open_prices = df['Open'].values.flatten()
    high_prices = df['High'].values.flatten()
    low_prices = df['Low'].values.flatten()
    close_prices = df['Close'].values.flatten()
    volume = df['Volume'].values.flatten()
    
    # Removed unused variables
    df['TRANGE'] = talib.TRANGE(high_prices, low_prices, close_prices)
    
    # ADX and ADXR
    df['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
    df['ADXR'] = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=14)

    # Absolute Price Oscillator (APO)
    df['APO'] = talib.APO(close_prices, fastperiod=12, slowperiod=26, matype=MA_Type.SMA)

    # Aroon and Aroon Oscillator
    aroon_up, aroon_down = talib.AROON(high_prices, low_prices, timeperiod=14)
    df['AROON_UP'] = aroon_up
    df['AROON_DOWN'] = aroon_down
    df['AROONOSC'] = talib.AROONOSC(high_prices, low_prices, timeperiod=14)

    # Balance Of Power (BOP)
    df['BOP'] = talib.BOP(open_prices, high_prices, low_prices, close_prices)

    # Commodity Channel Index (CCI)
    df['CCI'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)

    # Chande Momentum Oscillator (CMO)
    df['CMO'] = talib.CMO(close_prices, timeperiod=14)

    # Directional Movement Index (DX)
    df['DX'] = talib.DX(high_prices, low_prices, close_prices, timeperiod=14)

    # MACDEXT (MACD with controllable MA type)
    macd_ext, macd_ext_signal, macd_ext_hist = talib.MACDEXT(
        close_prices,
        fastperiod=12, fastmatype=MA_Type.SMA,
        slowperiod=26, slowmatype=MA_Type.SMA,
        signalperiod=9, signalmatype=MA_Type.SMA
    )
    df['MACDEXT'] = macd_ext
    df['MACDEXT_signal'] = macd_ext_signal
    df['MACDEXT_hist'] = macd_ext_hist

    # MACDFIX (Moving Average Convergence/Divergence Fix 12/26)
    macd_fix, macd_fix_signal, macd_fix_hist = talib.MACDFIX(close_prices, signalperiod=9)
    df['MACDFIX'] = macd_fix
    df['MACDFIX_hist'] = macd_fix_hist

    # Convert arrays to double precision if needed
    high_prices = np.asarray(high_prices, dtype=np.double)
    low_prices = np.asarray(low_prices, dtype=np.double)
    close_prices = np.asarray(close_prices, dtype=np.double)
    volume = np.asarray(volume, dtype=np.double)

     # Money Flow Index (MFI)
    df['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)

    # Minus Directional Indicator (MINUS_DI) and Minus Directional Movement (MINUS_DM)
    df['MINUS_DI'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
    df['MINUS_DM'] = talib.MINUS_DM(high_prices, low_prices, timeperiod=14)

    # Plus Directional Indicator (PLUS_DI) and Plus Directional Movement (PLUS_DM)
    df['PLUS_DI'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
    df['PLUS_DM'] = talib.PLUS_DM(high_prices, low_prices, timeperiod=14)

    # Percentage Price Oscillator (PPO)
    df['PPO'] = talib.PPO(close_prices, fastperiod=12, slowperiod=26, matype=MA_Type.SMA)

    # Rate of Change Indicators
    df['ROC'] = talib.ROC(close_prices, timeperiod=10)
    df['ROCP'] = talib.ROCP(close_prices, timeperiod=10)
    df['ROCR'] = talib.ROCR(close_prices, timeperiod=10)
    df['ROCR100'] = talib.ROCR100(close_prices, timeperiod=10)

    # Stochastic Fast (STOCHF)
    stochf_fastk, stochf_fastd = talib.STOCHF(
        high_prices, low_prices, close_prices,
        fastk_period=14, fastd_period=3, fastd_matype=MA_Type.SMA
    )
    df['STOCHF_fastk'] = stochf_fastk
    df['STOCHF_fastd'] = stochf_fastd

    # Stochastic RSI (STOCHRSI)
    stochrsi_fastk, stochrsi_fastd = talib.STOCHRSI(
        close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=MA_Type.SMA
    )
    df['STOCHRSI_fastk'] = stochrsi_fastk
    df['STOCHRSI_fastd'] = stochrsi_fastd

    # TRIX (Triple Exponential Average Oscillator)
    df['TRIX'] = talib.TRIX(close_prices, timeperiod=30)

    # Ultimate Oscillator (ULTOSC)
    df['ULTOSC'] = talib.ULTOSC(
        high_prices, low_prices, close_prices,
        timeperiod1=7, timeperiod2=14, timeperiod3=28
    )

    # Williams' %R (WILLR)
    df['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)

    return df