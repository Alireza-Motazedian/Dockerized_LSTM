<h1 align="center">LSTM Models</h1>

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
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#creating-models">2.1. Creating Models</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#training-models">2.2. Training Models</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#evaluating-models">2.3. Evaluating Models</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#file-descriptions"><i><b>3. File Descriptions</b></i></a>
</div>
&nbsp;

## Overview

### Purpose
This directory contains all LSTM (Long Short-Term Memory) model implementations, configurations, training utilities, and evaluation scripts for time series forecasting.

### Directory Structure
- `architectures/`: Contains different LSTM architecture implementations
- `configs/`: Model configuration files for different datasets and scenarios
- `evaluation/`: Utilities for evaluating model performance
- `training/`: Utilities for model training, including callbacks and optimization strategies
- Root files: Saved models and factory/registry utility files

## Usage

### Creating Models
To create a new LSTM model:
```python
from models.model_factory import create_model

# Create a univariate LSTM model with default parameters
model = create_model('univariate_lstm')

# Create a multivariate LSTM model with custom parameters
model = create_model('multivariate_lstm', 
                    input_shape=(10, 5),  # 10 time steps, 5 features
                    lstm_units=[64, 32],  # Two LSTM layers with 64 and 32 units
                    dropout_rate=0.2)
```

### Training Models
To train a model:
```python
from models.training.trainer import train_model

history = train_model(
    model,
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    early_stopping=True,
    patience=10
)
```

### Evaluating Models
To evaluate a trained model:
```python
from models.evaluation.evaluator import evaluate_model

metrics = evaluate_model(model, X_test, y_test)
print(f"Test MSE: {metrics['mse']}")
print(f"Test MAE: {metrics['mae']}")
```

## File Descriptions
- `__init__.py`: Package initialization
- `lstm_model_best.h5`: Best model checkpoint based on validation loss
- `lstm_model_final.h5`: Final trained model after complete training
- `model_factory.py`: Factory for creating model instances with different architectures
- `model_registry.py`: Registry of available model architectures 