<h1 align="center">Data Folder</h1>

## Data Description

This project uses financial time series data for LSTM model training and prediction. The specific datasets include:

- **Stock Price Data**: Daily historical stock prices (Open, High, Low, Close, Volume) for selected companies from Yahoo Finance. Typically covering a 5-year period (2018-2023).
- **Forex Data**: Currency exchange rate time series with hourly intervals for major currency pairs.
- **Economic Indicators**: Monthly macroeconomic indicators that might influence market performance (interest rates, GDP growth, unemployment rates, etc.).

### Feature Description

| Feature | Description | Type | Range/Units |
|---------|-------------|------|-------------|
| **Price Data** |
| Open | Opening price for the period | Numeric | Currency units |
| High | Highest price during the period | Numeric | Currency units |
| Low | Lowest price during the period | Numeric | Currency units |
| Close | Closing price for the period | Numeric | Currency units |
| Adj Close | Adjusted closing price (accounts for dividends and splits) | Numeric | Currency units |
| Volume | Number of shares/contracts traded | Numeric | Count |
| **Technical Indicators** |
| SMA(n) | Simple Moving Average over n periods | Numeric | Currency units |
| EMA(n) | Exponential Moving Average over n periods | Numeric | Currency units |
| RSI | Relative Strength Index | Numeric | 0-100 |
| MACD | Moving Average Convergence Divergence | Numeric | Currency units |
| MACD Signal | Signal line for MACD | Numeric | Currency units |
| MACD Histogram | Difference between MACD and signal line | Numeric | Currency units |
| Bollinger Upper | Upper Bollinger Band | Numeric | Currency units |
| Bollinger Lower | Lower Bollinger Band | Numeric | Currency units |
| ATR | Average True Range (volatility) | Numeric | Currency units |
| **Derived Features** |
| Returns | Daily/periodic return | Numeric | Percentage |
| Log Returns | Natural logarithm of returns | Numeric | Dimensionless |
| Volatility(n) | Standard deviation of returns over n periods | Numeric | Percentage |
| **Economic Indicators** |
| Interest Rate | Central bank interest rate | Numeric | Percentage |
| CPI | Consumer Price Index (inflation measure) | Numeric | Index value |
| Unemployment | Unemployment rate | Numeric | Percentage |
| GDP Growth | Gross Domestic Product growth rate | Numeric | Percentage |

### Sample Data Structure

#### Stock Price Data (CSV format)
```
Date,Open,High,Low,Close,Adj Close,Volume
2023-01-03,130.28,130.90,124.17,125.07,125.07,87155800
2023-01-04,126.89,128.66,125.08,126.36,126.36,70790800
2023-01-05,127.13,127.77,124.76,125.02,125.02,54873400
```

#### Forex Data (CSV format)
```
Timestamp,Open,High,Low,Close,Volume
2023-01-03 00:00:00,1.0678,1.0694,1.0671,1.0684,12754
2023-01-03 01:00:00,1.0684,1.0697,1.0680,1.0691,10542
2023-01-03 02:00:00,1.0691,1.0702,1.0688,1.0699,9324
```

#### Processed Dataset Dimensions
- Training data shape: (X_train: [samples, time_steps, features], y_train: [samples, 1])
- Testing data shape: (X_test: [samples, time_steps, features], y_test: [samples, 1])
- Typical sequence length (time_steps): 60 days
- Feature count: 5-20 features depending on technical indicators included

### Visualization

The processed data generates several visualization types stored in the `figures/` directory:

1. **Time Series Plots**: Raw and predicted prices over time
2. **Correlation Heatmaps**: Feature correlations
3. **Distribution Plots**: Histograms of returns and feature distributions
4. **Performance Metrics**: RMSE, MAE comparisons between models
5. **Training History**: Loss and validation metrics during training

Data preprocessing includes:
- Normalizing/scaling values (using MinMaxScaler)
- Creating sliding windows for sequence learning
- Train/test splitting (80/20 ratio)
- Feature engineering including technical indicators (MACD, RSI, moving averages)

## Contents

This directory contains sample datasets for demonstration and testing purposes:

- `stock_price_sample.csv`: Sample stock price data with daily OHLCV values
- `forex_sample.csv`: Sample forex data with hourly prices
- `economic_indicators_sample.csv`: Sample monthly economic indicators
- `processed_features_sample.csv`: Sample processed data with technical indicators

These samples are small enough to be included in the repository but represent the structure of the larger datasets that would be used in production.

For full datasets, this directory serves as a placeholder for:

- Input datasets
- Processed/transformed data
- Output datasets

## Usage

Datasets in this folder can be accessed from within Docker containers as they are mounted as volumes. This allows you to:

1. Perform exploratory data analysis
2. Train machine learning models
3. Test data processing pipelines

## Best Practices

- Keep raw data separate from processed data
- Large datasets should be placed here but excluded from version control
- This directory is already included in `.gitignore` and `.dockerignore`
- Document the source and structure of each dataset you add
- Consider using data version control tools for large datasets 