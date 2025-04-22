<h1 align="center">Model Training Utilities</h1>

## Table of Contents

<details>
  <summary><a href="#overview"><i><b>1. Overview</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#purpose">1.1. Purpose</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#training-features">1.2. Training Features</a><br>
  </div>
</details>
&nbsp;

<details>
  <summary><a href="#usage"><i><b>2. Usage</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#basic-training">2.1. Basic Training</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#advanced-training">2.2. Advanced Training</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#callbacks">2.3. Callbacks</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#file-descriptions"><i><b>3. File Descriptions</b></i></a>
</div>
&nbsp;

## Overview

### Purpose
This directory contains utilities for training LSTM models for time series forecasting. It includes training functions, custom callbacks, and learning rate management utilities.

### Training Features
- Standard model training with appropriate callbacks
- Early stopping to prevent overfitting
- Model checkpointing to save the best model
- Learning rate scheduling
- Training history visualization

## Usage

### Basic Training
For basic model training:

```python
from models.training.trainer import train_model

history = train_model(
    model,
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32
)
```

### Advanced Training
For advanced training with custom callbacks and learning rate scheduling:

```python
from models.training.trainer import train_model
from models.training.callbacks import create_callbacks
from models.training.lr_scheduler import CyclicLR

# Create custom callbacks
callbacks = create_callbacks(
    checkpoint_path='models/checkpoints/model_{epoch}.h5',
    early_stopping=True,
    patience=15,
    reduce_lr=True
)

# Add a custom learning rate scheduler
lr_scheduler = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=2000)
callbacks.append(lr_scheduler)

# Train with custom settings
history = train_model(
    model,
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

### Callbacks
You can create custom callbacks for training:

```python
from models.training.callbacks import create_callbacks

callbacks = create_callbacks(
    checkpoint_path='models/checkpoint.h5',  # Path to save model checkpoints
    early_stopping=True,      # Enable early stopping
    patience=10,              # Patience for early stopping
    reduce_lr=True,           # Enable learning rate reduction
    lr_patience=5,            # Patience for learning rate reduction
    tensorboard=True,         # Enable TensorBoard logging
    log_dir='logs/training'   # Directory for TensorBoard logs
)
```

## File Descriptions
- `__init__.py`: Package initialization
- `trainer.py`: Main training functions
- `callbacks.py`: Custom callback creation utilities
- `lr_scheduler.py`: Learning rate scheduling utilities
- `data_generator.py`: Data generators for efficient training with large datasets 