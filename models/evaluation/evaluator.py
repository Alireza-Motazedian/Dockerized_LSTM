"""
Evaluation utilities for LSTM models.
"""

import numpy as np
from utils.metrics import calculate_all_metrics


def evaluate_model(model, X_test, y_test, scaler=None):
    """
    Evaluate a model on test data.
    
    Parameters:
    -----------
    model : Model or object with predict method
        The model to evaluate.
    X_test : numpy.ndarray
        Test input data.
    y_test : numpy.ndarray
        Test target data.
    scaler : object, optional (default=None)
        Scaler object for inverse transformation of predictions.
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics.
    """
    # Generate predictions
    if hasattr(model, 'model'):
        y_pred = model.model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Reshape y_test and y_pred for inverse transformation
        y_test_shape = y_test.shape
        y_pred_shape = y_pred.shape
        
        # Handle different dimensions
        if len(y_test_shape) > 2:
            # For 3D outputs (e.g., sequence-to-sequence)
            y_test_reshaped = y_test.reshape(-1, y_test_shape[-1])
            y_pred_reshaped = y_pred.reshape(-1, y_pred_shape[-1])
        else:
            # For 2D outputs
            y_test_reshaped = y_test
            y_pred_reshaped = y_pred
        
        # Inverse transform
        y_test_inv = scaler.inverse_transform(y_test_reshaped)
        y_pred_inv = scaler.inverse_transform(y_pred_reshaped)
        
        # Reshape back to original shape
        if len(y_test_shape) > 2:
            y_test_inv = y_test_inv.reshape(y_test_shape)
            y_pred_inv = y_pred_inv.reshape(y_pred_shape)
        
        # Use inverse-transformed values for evaluation
        return calculate_all_metrics(y_test_inv, y_pred_inv)
    
    # Calculate metrics on original scale
    return calculate_all_metrics(y_test, y_pred)


def evaluate_model_with_prediction_intervals(model, X_test, y_test, n_samples=100, conf_level=0.95, scaler=None):
    """
    Evaluate a model with prediction intervals using Monte Carlo dropout.
    
    Parameters:
    -----------
    model : Model or object with predict method
        The model to evaluate (must have dropout layers set to training mode).
    X_test : numpy.ndarray
        Test input data.
    y_test : numpy.ndarray
        Test target data.
    n_samples : int, optional (default=100)
        Number of Monte Carlo samples.
    conf_level : float, optional (default=0.95)
        Confidence level for prediction intervals.
    scaler : object, optional (default=None)
        Scaler object for inverse transformation of predictions.
        
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics and prediction intervals.
    """
    import tensorflow as tf
    
    # Enable dropout at inference time
    class MCDropout(tf.keras.Model):
        def __init__(self, model):
            super(MCDropout, self).__init__()
            self.model = model
            
        def call(self, inputs, training=True):
            return self.model(inputs, training=training)
    
    # Create Monte Carlo dropout model
    if hasattr(model, 'model'):
        mc_model = MCDropout(model.model)
    else:
        mc_model = MCDropout(model)
    
    # Generate predictions
    all_predictions = []
    for _ in range(n_samples):
        pred = mc_model(X_test, training=True).numpy()
        all_predictions.append(pred)
    
    # Stack predictions
    all_predictions = np.stack(all_predictions, axis=0)
    
    # Calculate mean and prediction intervals
    mean_prediction = np.mean(all_predictions, axis=0)
    lower_bound = np.percentile(all_predictions, (1 - conf_level) / 2 * 100, axis=0)
    upper_bound = np.percentile(all_predictions, (1 + conf_level) / 2 * 100, axis=0)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Reshape arrays for inverse transformation
        y_test_shape = y_test.shape
        pred_shape = mean_prediction.shape
        
        # Handle different dimensions
        if len(y_test_shape) > 2:
            # For 3D outputs (e.g., sequence-to-sequence)
            y_test_reshaped = y_test.reshape(-1, y_test_shape[-1])
            mean_pred_reshaped = mean_prediction.reshape(-1, pred_shape[-1])
            lower_bound_reshaped = lower_bound.reshape(-1, pred_shape[-1])
            upper_bound_reshaped = upper_bound.reshape(-1, pred_shape[-1])
        else:
            # For 2D outputs
            y_test_reshaped = y_test
            mean_pred_reshaped = mean_prediction
            lower_bound_reshaped = lower_bound
            upper_bound_reshaped = upper_bound
        
        # Inverse transform
        y_test_inv = scaler.inverse_transform(y_test_reshaped)
        mean_pred_inv = scaler.inverse_transform(mean_pred_reshaped)
        lower_bound_inv = scaler.inverse_transform(lower_bound_reshaped)
        upper_bound_inv = scaler.inverse_transform(upper_bound_reshaped)
        
        # Reshape back to original shape
        if len(y_test_shape) > 2:
            y_test_inv = y_test_inv.reshape(y_test_shape)
            mean_pred_inv = mean_pred_inv.reshape(pred_shape)
            lower_bound_inv = lower_bound_inv.reshape(pred_shape)
            upper_bound_inv = upper_bound_inv.reshape(pred_shape)
        
        # Use inverse-transformed values for evaluation
        metrics = calculate_all_metrics(y_test_inv, mean_pred_inv)
        
        return {
            'metrics': metrics,
            'mean_prediction': mean_pred_inv,
            'lower_bound': lower_bound_inv,
            'upper_bound': upper_bound_inv,
            'prediction_interval_width': upper_bound_inv - lower_bound_inv
        }
    
    # Calculate metrics on original scale
    metrics = calculate_all_metrics(y_test, mean_prediction)
    
    return {
        'metrics': metrics,
        'mean_prediction': mean_prediction,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'prediction_interval_width': upper_bound - lower_bound
    }


def evaluate_forecast_horizon(model, X_test, y_true, horizon=10, step_size=1):
    """
    Evaluate model performance over different forecast horizons.
    
    Parameters:
    -----------
    model : Model or object with predict method
        The model to evaluate.
    X_test : numpy.ndarray
        Initial input data for forecasting.
    y_true : numpy.ndarray
        True values for the forecast horizon.
    horizon : int, optional (default=10)
        Maximum forecast horizon to evaluate.
    step_size : int, optional (default=1)
        Step size for iterative forecasting.
        
    Returns:
    --------
    results : dict
        Dictionary containing forecast evaluation at different horizons.
    """
    # Initialize containers for results
    forecasts = []
    metrics_by_horizon = []
    
    # Make a copy of the input
    current_input = X_test.copy()
    
    # Iterative forecasting
    for step in range(0, horizon, step_size):
        # Generate forecast for the current step
        if hasattr(model, 'model'):
            forecast = model.model.predict(current_input)
        else:
            forecast = model.predict(current_input)
        
        # Store the forecast
        forecasts.append(forecast)
        
        # Evaluate forecast against true values
        if step < len(y_true):
            true_values = y_true[step:step+step_size]
            forecast_values = forecast[:step_size]
            
            # Calculate metrics
            step_metrics = calculate_all_metrics(true_values, forecast_values)
            metrics_by_horizon.append({
                'horizon': step + 1,
                'metrics': step_metrics
            })
        
        # Update input for next step (roll forward and add forecast)
        if step + step_size < horizon:
            # Shift input forward
            current_input = np.roll(current_input, -step_size, axis=1)
            
            # Replace the last values with the forecast
            # Note: This assumes the forecast shape matches the input shape appropriately
            if len(current_input.shape) == 3:  # (batch, time_steps, features)
                current_input[:, -step_size:, :] = forecast.reshape(current_input[:, -step_size:, :].shape)
            else:  # (batch, features)
                current_input[:, -step_size:] = forecast.reshape(current_input[:, -step_size:].shape)
    
    # Combine all forecasts
    all_forecasts = np.concatenate(forecasts, axis=0)
    
    return {
        'forecasts': all_forecasts,
        'metrics_by_horizon': metrics_by_horizon
    }


def evaluate_with_bootstrap(model, X_test, y_test, n_bootstraps=1000, conf_level=0.95, scaler=None):
    """
    Evaluate a model with bootstrap confidence intervals.
    
    Parameters:
    -----------
    model : Model or object with predict method
        The model to evaluate.
    X_test : numpy.ndarray
        Test input data.
    y_test : numpy.ndarray
        Test target data.
    n_bootstraps : int, optional (default=1000)
        Number of bootstrap samples.
    conf_level : float, optional (default=0.95)
        Confidence level for intervals.
    scaler : object, optional (default=None)
        Scaler object for inverse transformation of predictions.
        
    Returns:
    --------
    results : dict
        Dictionary containing bootstrap evaluation results.
    """
    # Generate predictions
    if hasattr(model, 'model'):
        y_pred = model.model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        # Reshape y_test and y_pred for inverse transformation
        y_test_shape = y_test.shape
        y_pred_shape = y_pred.shape
        
        # Handle different dimensions
        if len(y_test_shape) > 2:
            # For 3D outputs (e.g., sequence-to-sequence)
            y_test_reshaped = y_test.reshape(-1, y_test_shape[-1])
            y_pred_reshaped = y_pred.reshape(-1, y_pred_shape[-1])
        else:
            # For 2D outputs
            y_test_reshaped = y_test
            y_pred_reshaped = y_pred
        
        # Inverse transform
        y_test = scaler.inverse_transform(y_test_reshaped).reshape(y_test_shape)
        y_pred = scaler.inverse_transform(y_pred_reshaped).reshape(y_pred_shape)
    
    # Compute overall metrics
    overall_metrics = calculate_all_metrics(y_test, y_pred)
    
    # Bootstrap metrics
    n_samples = len(y_test)
    bootstrap_metrics = {metric: [] for metric in overall_metrics}
    
    for _ in range(n_bootstraps):
        # Generate bootstrap indices
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Calculate metrics for this bootstrap sample
        metrics = calculate_all_metrics(y_test[indices], y_pred[indices])
        
        # Store results
        for metric, value in metrics.items():
            bootstrap_metrics[metric].append(value)
    
    # Calculate confidence intervals
    bootstrap_intervals = {}
    for metric, values in bootstrap_metrics.items():
        lower = np.percentile(values, (1 - conf_level) / 2 * 100)
        upper = np.percentile(values, (1 + conf_level) / 2 * 100)
        bootstrap_intervals[metric] = (lower, upper)
    
    return {
        'overall_metrics': overall_metrics,
        'bootstrap_metrics': bootstrap_metrics,
        'bootstrap_intervals': bootstrap_intervals
    } 