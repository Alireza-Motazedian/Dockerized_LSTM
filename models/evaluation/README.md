<h1 align="center">Model Evaluation Utilities</h1>

This directory contains utilities for evaluating and analyzing LSTM and other time series forecasting models.

## 1. Contents

### 1.1. Files

- **lstm_state_analysis.py**: Utilities for analyzing LSTM internal states, including cell states, hidden states, and gate activations.
- **forecast_analyzer.py**: Comprehensive tools for analyzing forecast performance, including error analysis, trend/seasonality decomposition, and robustness testing.

### 1.2. Main Features

#### 1.2.1. LSTM State Analysis
- Extract and visualize LSTM internal states
- Analyze LSTM gate activations
- Study feature importance through perturbation analysis
- Evaluate LSTM memory capabilities

#### 1.2.2. Forecast Analysis
- Error distribution and statistics
- Trend and seasonality component analysis
- Autocorrelation analysis
- Prediction intervals evaluation
- Extreme error analysis
- Forecast robustness testing

## 2. Usage

The utilities in this directory can be used to gain deeper insights into model performance and behavior:

```python
# Example: Analyzing LSTM states
from models.evaluation.lstm_state_analysis import extract_lstm_states

states = extract_lstm_states(model, X_sample)
print(f"Cell state shape: {states['cell_state'].shape}")
print(f"Hidden state shape: {states['hidden_state'].shape}")

# Example: Analyzing forecast errors
from models.evaluation.forecast_analyzer import analyze_forecast_errors

error_analysis = analyze_forecast_errors(y_true, y_pred, scaler)
print(f"RMSE: {error_analysis['rmse']}")
print(f"MAPE: {error_analysis['mape']}")
```

## 3. Dependencies

These utilities rely on:
- NumPy
- Pandas
- TensorFlow/Keras
- Scikit-learn
- Statsmodels
- Matplotlib (for visualization) 