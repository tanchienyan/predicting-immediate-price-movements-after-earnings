# predicting-immediate-price-movements-after-earnings
In financial markets, earnings announcements often act as catalysts for significant stock price movements. Predicting how the market will react to these announcements is a complex task that requires an integration of multiple layers of analysis. This project aims to build a robust predictive model based on my own investment styles and what I look at. It combines fundamental analysis (via financial earnings data), technical analysis (through indicators like RSI, MACD, and Bollinger Bands), chart pattern recognition (e.g., head and shoulders, double tops), and sentiment analysis (from news and social media). By incorporating all these diverse signals, the model captures both quantitative and behavioral factors that influence stock prices post-earnings, creating a more holistic and accurate forecasting tool.

The core of this project is a predictive stock price model built using a hybrid approach that leverages technical indicators, chart pattern recognition, post-earnings behavior, and time-series deep learning models such as LSTM. The model is designed to capture immediate to short-term reactions of stocks to quarterly earnings announcements.

To enhance model accuracy and robustness, I engineered a diverse and rich set of features—ranging from micro-level technical indicators to macro-level market signals.

Feature Engineering and Technical Indicators
A key component of this model lies in the feature_engineering.py script, which consolidates multiple layers of market signals into a clean, model-ready dataset. The features engineered include:

Traditional Technical Indicators:
Indicators such as RSI, MACD, Bollinger Bands, moving averages (SMA/EMA), and volume-based signals are computed to provide insight into price momentum, volatility, and overbought/oversold conditions.
Chart Pattern Recognition:
Using the pattern_recognition.py and pattern_recognition_demo.py modules, the model integrates classic chart patterns (e.g., head and shoulders, double tops, flags) as features. These patterns are widely used by traders to signal potential breakout or reversal zones.
Drawdown and Recovery Analysis:
A custom notebook, drawdown.ipynb, was developed to analyze major drawdowns in a stock's historical price data. The magnitude and duration of these drawdowns—as well as how long the stock takes to recover—are included as features. These metrics help measure a stock’s resilience and historical volatility.
Industry Market Return Forecasting:
Through industry_market_return_predictor.py, the expected return of each stock's broader industry is calculated using sector-level trends and historical averages. This macroeconomic factor helps contextualize individual stock performance within its industry group.


LSTM Model for Time-Series Forecasting:
Implemented in LSTM_Stock.py, the long short-term memory (LSTM) neural network is designed to capture sequential dependencies in time-series data. It is particularly well-suited for forecasting short-term price movements post-earnings due to its memory capabilities and pattern sensitivity.
Unsupervised Learning via K-Means Clustering:
In K-mean algorithm SP500.ipynb, K-means clustering is used to segment S&P 500 stocks based on their historical post-earnings behavior and volatility. This clustering enables the model to learn group-level behavioral patterns and tailor predictions based on which cluster a stock belongs to.

Mini Project: Sentiment and Volatility Analysis
In parallel with the main model, I conducted a mini project titled “The Impact of Elon Musk’s Tweets on DOGE Volatility” (tweet_sentimental_musk_doge.ipynb). This exploration investigates how social media sentiment—specifically Elon Musk’s tweets—affects the volatility of Dogecoin. By performing sentiment analysis on Musk’s tweets and comparing them to DOGE price movements, I demonstrated a clear correlation between online sentiment and real-time market volatility. This study highlights the potential impact of non-fundamental information on asset prices and provided valuable insights into the role of sentiment in predictive modeling—insights that could later be extended to equities.
