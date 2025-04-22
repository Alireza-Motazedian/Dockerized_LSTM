<h1 align="center">Model Configurations</h1>

## Table of Contents

<details>
  <summary><a href="#overview"><i><b>1. Overview</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#purpose">1.1. Purpose</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#configuration-structure">1.2. Configuration Structure</a><br>
  </div>
</details>
&nbsp;

<details>
  <summary><a href="#usage"><i><b>2. Usage</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#loading-configurations">2.1. Loading Configurations</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#creating-custom-configurations">2.2. Creating Custom Configurations</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#file-descriptions"><i><b>3. File Descriptions</b></i></a>
</div>
&nbsp;

## Overview

### Purpose
This directory contains configuration files for the different LSTM model architectures. These configuration files specify the hyperparameters and structure of the models for different time series forecasting tasks and datasets.

### Configuration Structure
Each configuration file is a JSON file specifying:
- Model architecture type
- Input sequence length
- LSTM layer configurations (units, activation, etc.)
- Dropout rates
- Dense layer configurations
- Compilation settings (optimizer, loss, learning rate)
- Training parameters (batch size, epochs, etc.)

## Usage

### Loading Configurations
To load and use a configuration:

```python
import json
from models.model_factory import create_model_from_config

# Load the configuration file
with open('models/configs/stock_price_lstm.json', 'r') as f:
    config = json.load(f)

# Create a model using the configuration
model = create_model_from_config(config)
```

### Creating Custom Configurations
You can create custom configurations for new datasets or tasks. The basic structure is:

```json
{
  "architecture": "vanilla_lstm",
  "input_shape": [10, 1],
  "lstm_layers": [
    {"units": 50, "return_sequences": false, "activation": "tanh"}
  ],
  "dropout_rate": 0.2,
  "dense_layers": [
    {"units": 1, "activation": "linear"}
  ],
  "compile": {
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "mse",
    "metrics": ["mae"]
  },
  "training": {
    "batch_size": 32,
    "epochs": 100,
    "early_stopping": true,
    "patience": 10
  }
}
```

## File Descriptions
- `__init__.py`: Package initialization
- `stock_price_lstm.json`: Configuration for stock price forecasting
- `weather_lstm.json`: Configuration for weather data forecasting
- `energy_consumption_lstm.json`: Configuration for energy consumption forecasting
- `multivariate_lstm.json`: Configuration for multivariate time series forecasting
- `config_utils.py`: Utility functions for working with configuration files 