"""
Custom evaluation metrics for time series forecasting.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
        
    Returns:
    --------
    rmse : float
        Root Mean Squared Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-10):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    epsilon : float, optional (default=1e-10)
        Small value to avoid division by zero.
        
    Returns:
    --------
    mape : float
        Mean Absolute Percentage Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Handle zero and near-zero true values
    mask = np.abs(y_true) > epsilon
    
    if not np.any(mask):
        return np.nan
    
    # Calculate MAPE only on non-zero values
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
        
    Returns:
    --------
    smape : float
        Symmetric Mean Absolute Percentage Error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Handle case where both y_true and y_pred are 0
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    
    if not np.any(mask):
        return 0.0
    
    # Calculate SMAPE only on non-zero denominator
    return 200 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])


def directional_accuracy(y_true, y_pred):
    """
    Calculate Directional Accuracy (DA) - percentage of correct direction predictions.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
        
    Returns:
    --------
    da : float
        Directional Accuracy (percentage).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate the direction (up or down) of actual and predicted values
    actual_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Calculate the percentage of correct direction predictions
    correct_directions = np.mean(actual_direction == pred_direction)
    
    return 100 * correct_directions


def theil_u_statistic(y_true, y_pred):
    """
    Calculate Theil's U statistic (U2) for forecast accuracy.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
        
    Returns:
    --------
    u2 : float
        Theil's U statistic.
        
    Notes:
    ------
    U2 = 0: Perfect forecast
    U2 = 1: Same accuracy as naive forecast
    U2 > 1: Worse than naive forecast
    U2 < 1: Better than naive forecast
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate forecast error
    error = y_true[1:] - y_pred[1:]
    
    # Calculate naive forecast error (using previous value as prediction)
    naive_error = y_true[1:] - y_true[:-1]
    
    # Calculate Theil's U statistic
    numerator = np.sqrt(np.mean(np.square(error)))
    denominator = np.sqrt(np.mean(np.square(naive_error)))
    
    # Handle division by zero
    if denominator == 0:
        return np.inf
    
    return numerator / denominator


def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all available metrics for forecast evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics.
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'smape': symmetric_mean_absolute_percentage_error(y_true, y_pred)
    }
    
    # For metrics that require multiple time steps
    if len(y_true) > 1:
        metrics['directional_accuracy'] = directional_accuracy(y_true, y_pred)
        metrics['theil_u'] = theil_u_statistic(y_true, y_pred)
    
    return metrics 