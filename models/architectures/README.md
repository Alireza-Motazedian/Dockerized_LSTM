<h1 align="center">LSTM Architectures</h1>

## Table of Contents

<details>
  <summary><a href="#overview"><i><b>1. Overview</b></i></a></summary>
  <div>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#purpose">1.1. Purpose</a><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#available-architectures">1.2. Available Architectures</a><br>
  </div>
</details>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#architecture-descriptions"><i><b>2. Architecture Descriptions</b></i></a>
</div>
&nbsp;

<div>
  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#file-descriptions"><i><b>3. File Descriptions</b></i></a>
</div>
&nbsp;

## Overview

### Purpose
This directory contains different LSTM (Long Short-Term Memory) architecture implementations for time series forecasting. Each architecture is designed for different types of time series forecasting tasks.

### Available Architectures
- Vanilla LSTM: Basic LSTM implementation for univariate time series forecasting
- Stacked LSTM: Multiple LSTM layers for learning more complex patterns
- Bidirectional LSTM: For capturing both past and future context
- Seq2Seq LSTM: Encoder-decoder architecture for variable length predictions
- Multivariate LSTM: For forecasting with multiple input features

## Architecture Descriptions

### Vanilla LSTM
A simple LSTM architecture for univariate time series forecasting with a single LSTM layer followed by a dense layer for prediction. This architecture is suitable for simpler time series with clear patterns.

### Stacked LSTM
Multiple LSTM layers stacked on top of each other, allowing the model to learn more complex temporal relationships. This architecture is suitable for time series with complex non-linear patterns.

### Bidirectional LSTM
Processes the input sequence in both forward and backward directions, allowing the model to learn from both past and future contexts. This is particularly useful for time series where future values depend on both past and future contexts.

### Seq2Seq LSTM
An encoder-decoder architecture where one LSTM encodes the input sequence into a fixed-length context vector, and another LSTM decodes this vector into a variable-length output sequence. This is suitable for multi-step forecasting.

### Multivariate LSTM
Designed to handle multiple input features, making it suitable for time series forecasting where multiple variables affect the target variable.

## File Descriptions
- `__init__.py`: Package initialization
- `vanilla_lstm.py`: Implementation of a basic LSTM model
- `stacked_lstm.py`: Implementation of a multi-layer LSTM model
- `bidirectional_lstm.py`: Implementation of a bidirectional LSTM model
- `seq2seq_lstm.py`: Implementation of an encoder-decoder LSTM model
- `multivariate_lstm.py`: Implementation of an LSTM model for multivariate forecasting 