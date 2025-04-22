<h1 align="center">Utilities</h1>

## Table of Contents

<details>
  <summary><a href="#overview"><i><b>1. Overview</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#purpose">1.1. Purpose</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#directory-structure">1.2. Directory Structure</a><br>
  </div>
</details>
&nbsp;

<details>
  <summary><a href="#usage"><i><b>2. Usage</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#metrics">2.1. Metrics</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#visualization">2.2. Visualization</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#preprocessing">2.3. Preprocessing</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#file-descriptions"><i><b>3. File Descriptions</b></i></a>
</div>
&nbsp;

## Overview

### Purpose
This directory contains utility functions and helper modules used throughout the LSTM time series forecasting project. These utilities provide common functionality for data preprocessing, visualization, and evaluation metrics.

### Directory Structure
The utilities are organized into specific Python modules:
- `metrics.py`: Custom evaluation metrics for time series forecasting
- `visualization.py`: Functions for creating plots and visualizations
- `preprocessing.py`: Data preprocessing utilities

## Usage

### Metrics
To use the custom metrics for evaluating time series forecasts:

```python
from utils.metrics import mean_absolute_percentage_error, directional_accuracy

# Calculate MAPE between actual and predicted values
mape = mean_absolute_percentage_error(y_true, y_pred)

# Calculate directional accuracy (% of correct direction predictions)
da = directional_accuracy(y_true, y_pred)
```

### Visualization
To create standard visualizations for time series data and model results:

```python
from utils.visualization import plot_forecast, plot_training_history, plot_lstm_states

# Plot the forecast against actual values
plot_forecast(y_true, y_pred, title='LSTM Forecast', save_path='figures/forecast.png')

# Plot the training history
plot_training_history(history, save_path='figures/training_history.png')

# Visualize LSTM internal states
plot_lstm_states(model, X_sample, save_path='figures/lstm_states.png')
```

### Preprocessing
To preprocess time series data for LSTM models:

```python
from utils.preprocessing import create_sequences, normalize_data, train_val_test_split

# Normalize the data
normalized_data, scaler = normalize_data(time_series_data)

# Create sequences for LSTM input
X, y = create_sequences(normalized_data, seq_length=10, horizon=1)

# Split the data into training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, val_size=0.2, test_size=0.1)
```

## File Descriptions
- `metrics.py`: Contains custom evaluation metrics specific to time series forecasting
- `visualization.py`: Contains functions for creating standard plots and visualizations
- `preprocessing.py`: Contains functions for preparing time series data for modeling 