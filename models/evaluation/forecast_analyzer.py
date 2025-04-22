"""
Utilities for analyzing forecast performance and characteristics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional


def analyze_forecast_errors(y_true, y_pred, scaler=None):
    """
    Analyze errors in forecasts, including error distribution and statistics.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values with shape (n_samples, n_outputs).
    y_pred : numpy.ndarray
        Predicted values with shape (n_samples, n_outputs).
    scaler : object, optional (default=None)
        Scaler object with inverse_transform method for converting scaled values back to original scale.
        
    Returns:
    --------
    error_analysis : dict
        Dictionary containing error analysis results.
    """
    # Inverse transform values if scaler is provided
    if scaler is not None:
        # Handle different output dimensions
        if len(y_true.shape) == 2:
            y_true_orig = scaler.inverse_transform(y_true)
            y_pred_orig = scaler.inverse_transform(y_pred)
        elif len(y_true.shape) == 1:
            # Reshape to 2D for inverse_transform
            y_true_reshaped = y_true.reshape(-1, 1)
            y_pred_reshaped = y_pred.reshape(-1, 1)
            y_true_orig = scaler.inverse_transform(y_true_reshaped).flatten()
            y_pred_orig = scaler.inverse_transform(y_pred_reshaped).flatten()
        else:
            # For 3D outputs, reshape for inverse_transform then restore shape
            original_shape = y_true.shape
            y_true_reshaped = y_true.reshape(-1, y_true.shape[-1])
            y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])
            y_true_orig = scaler.inverse_transform(y_true_reshaped).reshape(original_shape)
            y_pred_orig = scaler.inverse_transform(y_pred_reshaped).reshape(original_shape)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    # Calculate errors
    errors = y_true_orig - y_pred_orig
    
    # Error statistics
    mae = np.mean(np.abs(errors))
    mse = np.mean(np.square(errors))
    rmse = np.sqrt(mse)
    
    # Calculate percentage errors if not close to zero
    if np.any(np.abs(y_true_orig) > 1e-10):
        percentage_errors = np.where(
            np.abs(y_true_orig) > 1e-10,
            errors / np.abs(y_true_orig) * 100,
            0
        )
        mape = np.mean(np.abs(percentage_errors))
    else:
        mape = np.nan
        percentage_errors = np.zeros_like(errors)
    
    # Error distribution characteristics
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_skew = np.mean(((errors - error_mean) / error_std) ** 3) if error_std > 0 else 0
    error_kurtosis = np.mean(((errors - error_mean) / error_std) ** 4) if error_std > 0 else 0
    
    # Quantiles
    quantiles = np.percentile(errors, [0, 25, 50, 75, 100])
    
    return {
        'errors': errors,
        'percentage_errors': percentage_errors,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'error_mean': error_mean,
        'error_std': error_std,
        'error_skew': error_skew,
        'error_kurtosis': error_kurtosis,
        'error_quantiles': {
            'min': quantiles[0],
            'q1': quantiles[1],
            'median': quantiles[2],
            'q3': quantiles[3],
            'max': quantiles[4]
        }
    }


def analyze_trend_seasonality(y_true, y_pred, timestamp_freq=None):
    """
    Analyze how well the model captures trend and seasonality components.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values with shape (n_samples, n_outputs).
    y_pred : numpy.ndarray
        Predicted values with shape (n_samples, n_outputs).
    timestamp_freq : str, optional (default=None)
        Frequency string for pandas DatetimeIndex (e.g., 'D' for daily, 'H' for hourly).
        If provided, seasonality analysis will be more specific to the frequency.
        
    Returns:
    --------
    decomposition_analysis : dict
        Dictionary containing trend and seasonality analysis results.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Convert to 1D if needed
    if len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    
    # Create time series
    if timestamp_freq:
        # Create a date range with the specified frequency
        index = pd.date_range(start='2000-01-01', periods=len(y_true), freq=timestamp_freq)
    else:
        # Use a default range
        index = pd.RangeIndex(start=0, stop=len(y_true))
    
    # Create pandas Series
    true_series = pd.Series(y_true, index=index)
    pred_series = pd.Series(y_pred, index=index)
    
    # Try to determine period for seasonal decomposition
    if timestamp_freq:
        if timestamp_freq == 'D':
            period = 7  # Weekly seasonality
        elif timestamp_freq == 'H':
            period = 24  # Daily seasonality
        elif timestamp_freq == 'M':
            period = 12  # Yearly seasonality
        else:
            # Default period
            period = min(len(y_true) // 2, 12)
    else:
        # Try to detect periodicity or use default
        period = min(len(y_true) // 2, 12)
    
    # Ensure we have enough data points
    if len(y_true) < 2 * period:
        period = len(y_true) // 2
    
    try:
        # Decompose true series
        true_decomposition = seasonal_decompose(true_series, model='additive', period=period)
        true_trend = true_decomposition.trend
        true_seasonal = true_decomposition.seasonal
        true_residual = true_decomposition.resid
        
        # Decompose predicted series
        pred_decomposition = seasonal_decompose(pred_series, model='additive', period=period)
        pred_trend = pred_decomposition.trend
        pred_seasonal = pred_decomposition.seasonal
        pred_residual = pred_decomposition.resid
        
        # Fill NaN values (seasonal_decompose introduces NaNs at the edges)
        true_trend = true_trend.fillna(method='bfill').fillna(method='ffill')
        true_seasonal = true_seasonal.fillna(method='bfill').fillna(method='ffill')
        true_residual = true_residual.fillna(method='bfill').fillna(method='ffill')
        
        pred_trend = pred_trend.fillna(method='bfill').fillna(method='ffill')
        pred_seasonal = pred_seasonal.fillna(method='bfill').fillna(method='ffill')
        pred_residual = pred_residual.fillna(method='bfill').fillna(method='ffill')
        
        # Calculate metrics for each component
        trend_mse = mean_squared_error(true_trend, pred_trend)
        seasonal_mse = mean_squared_error(true_seasonal, pred_seasonal)
        residual_mse = mean_squared_error(true_residual, pred_residual)
        
        # Calculate correlation between true and predicted components
        trend_corr = np.corrcoef(true_trend, pred_trend)[0, 1]
        seasonal_corr = np.corrcoef(true_seasonal, pred_seasonal)[0, 1]
        
        # Calculate component strengths
        total_var = np.var(true_series)
        trend_strength = 1 - np.var(true_series - true_trend) / total_var if total_var > 0 else 0
        seasonal_strength = 1 - np.var(true_series - true_seasonal) / np.var(true_series - true_trend) if np.var(true_series - true_trend) > 0 else 0
        
        return {
            'trend_mse': trend_mse,
            'seasonal_mse': seasonal_mse,
            'residual_mse': residual_mse,
            'trend_correlation': trend_corr,
            'seasonal_correlation': seasonal_corr,
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength,
            'components': {
                'true_trend': true_trend.values,
                'true_seasonal': true_seasonal.values,
                'true_residual': true_residual.values,
                'pred_trend': pred_trend.values,
                'pred_seasonal': pred_seasonal.values,
                'pred_residual': pred_residual.values
            }
        }
    except:
        # If decomposition fails, return simplified analysis
        return {
            'error': 'Seasonal decomposition failed, possibly due to data characteristics or insufficient data points'
        }


def analyze_forecast_autocorrelation(y_true, y_pred, max_lag=20):
    """
    Analyze autocorrelation in true values and predictions to see if the model captures temporal dependencies.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values with shape (n_samples, n_outputs) or (n_samples,).
    y_pred : numpy.ndarray
        Predicted values with shape (n_samples, n_outputs) or (n_samples,).
    max_lag : int, optional (default=20)
        Maximum lag for autocorrelation calculation.
        
    Returns:
    --------
    autocorrelation_analysis : dict
        Dictionary containing autocorrelation analysis results.
    """
    from statsmodels.tsa.stattools import acf
    
    # Ensure we're working with flattened arrays
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    
    # Calculate autocorrelation for true values and predictions
    true_acf = acf(y_true, nlags=max_lag, fft=True)
    pred_acf = acf(y_pred, nlags=max_lag, fft=True)
    
    # Calculate autocorrelation of errors
    errors = y_true - y_pred
    error_acf = acf(errors, nlags=max_lag, fft=True)
    
    # Calculate correlation between true and predicted autocorrelations
    acf_correlation = np.corrcoef(true_acf, pred_acf)[0, 1]
    
    # Find the lag where autocorrelation first drops below 0.2
    true_memory = max_lag
    for i, val in enumerate(true_acf):
        if i > 0 and abs(val) < 0.2:
            true_memory = i
            break
    
    pred_memory = max_lag
    for i, val in enumerate(pred_acf):
        if i > 0 and abs(val) < 0.2:
            pred_memory = i
            break
    
    # Check for significant autocorrelation in errors (indicates missed patterns)
    error_significance = 1.96 / np.sqrt(len(errors))  # 95% confidence interval
    significant_error_lags = [i for i, val in enumerate(error_acf) if i > 0 and abs(val) > error_significance]
    
    return {
        'true_acf': true_acf,
        'pred_acf': pred_acf,
        'error_acf': error_acf,
        'acf_correlation': acf_correlation,
        'true_memory': true_memory,
        'pred_memory': pred_memory,
        'significant_error_lags': significant_error_lags
    }


def analyze_prediction_intervals(y_true, y_pred_low, y_pred, y_pred_high):
    """
    Analyze prediction intervals for forecasts.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values with shape (n_samples, n_outputs) or (n_samples,).
    y_pred_low : numpy.ndarray
        Lower bound predictions with shape (n_samples, n_outputs) or (n_samples,).
    y_pred : numpy.ndarray
        Mean predictions with shape (n_samples, n_outputs) or (n_samples,).
    y_pred_high : numpy.ndarray
        Upper bound predictions with shape (n_samples, n_outputs) or (n_samples,).
        
    Returns:
    --------
    interval_analysis : dict
        Dictionary containing prediction interval analysis results.
    """
    # Ensure all arrays are flat
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred_low = y_pred_low.flatten()
        y_pred = y_pred.flatten()
        y_pred_high = y_pred_high.flatten()
    
    # Calculate coverage (percentage of true values falling within the prediction interval)
    within_interval = (y_true >= y_pred_low) & (y_true <= y_pred_high)
    coverage = np.mean(within_interval) * 100
    
    # Calculate interval width
    interval_width = y_pred_high - y_pred_low
    mean_interval_width = np.mean(interval_width)
    normalized_interval_width = mean_interval_width / np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) > 0 else np.nan
    
    # Calculate interval sharpness (narrower intervals are better)
    sharpness = 1 / mean_interval_width if mean_interval_width > 0 else np.inf
    
    # Calculate mean squared error for each prediction bound
    mse_low = mean_squared_error(y_true, y_pred_low)
    mse_mean = mean_squared_error(y_true, y_pred)
    mse_high = mean_squared_error(y_true, y_pred_high)
    
    # Calculate interval quality scores
    # Interval Score (IS) - penalizes for width and missed points
    alpha = 0.05  # assuming 95% prediction interval
    missed_low = y_true < y_pred_low
    missed_high = y_true > y_pred_high
    interval_score = (mean_interval_width + 
                     2/alpha * np.mean((y_pred_low - y_true) * missed_low) +
                     2/alpha * np.mean((y_true - y_pred_high) * missed_high))
    
    # Continuous Ranked Probability Score approximation (lower is better)
    crps_approx = np.mean(np.abs(y_true - y_pred)) + 0.5 * mean_interval_width
    
    return {
        'coverage': coverage,
        'mean_interval_width': mean_interval_width,
        'normalized_interval_width': normalized_interval_width,
        'sharpness': sharpness,
        'mse_low': mse_low,
        'mse_mean': mse_mean,
        'mse_high': mse_high,
        'interval_score': interval_score,
        'crps_approximation': crps_approx,
        'coverage_by_point': within_interval
    }


def analyze_extreme_errors(y_true, y_pred, percentile_threshold=95):
    """
    Analyze extreme errors in forecasts to identify problematic patterns.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values with shape (n_samples, n_outputs) or (n_samples,).
    y_pred : numpy.ndarray
        Predicted values with shape (n_samples, n_outputs) or (n_samples,).
    percentile_threshold : int, optional (default=95)
        Percentile threshold to define extreme errors.
        
    Returns:
    --------
    extreme_error_analysis : dict
        Dictionary containing extreme error analysis results.
    """
    # Ensure arrays are flat
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    
    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)
    
    # Identify extreme errors (above specified percentile)
    threshold = np.percentile(abs_errors, percentile_threshold)
    extreme_indices = np.where(abs_errors >= threshold)[0]
    extreme_errors = abs_errors[extreme_indices]
    extreme_values_true = y_true[extreme_indices]
    extreme_values_pred = y_pred[extreme_indices]
    
    # Calculate statistics for extreme errors
    extreme_error_mean = np.mean(extreme_errors)
    extreme_error_std = np.std(extreme_errors)
    
    # Check if extreme errors occur more frequently for high or low values
    extreme_value_correlation = np.corrcoef(extreme_values_true, extreme_errors)[0, 1] if len(extreme_errors) > 1 else 0
    
    # Check for patterns in extreme error positions
    diff_indices = np.diff(extreme_indices)
    consecutive_errors = np.sum(diff_indices == 1)
    
    # Check if extreme errors are associated with rapid changes
    if len(extreme_indices) > 0 and np.min(extreme_indices) > 0 and np.max(extreme_indices) < len(y_true) - 1:
        # Calculate changes before extreme errors
        changes_before = np.abs(y_true[extreme_indices] - y_true[extreme_indices - 1])
        mean_change_before = np.mean(changes_before)
        
        # Calculate changes at normal points for comparison
        normal_indices = np.setdiff1d(np.arange(1, len(y_true)), extreme_indices)
        normal_changes = np.abs(y_true[normal_indices] - y_true[normal_indices - 1])
        mean_normal_change = np.mean(normal_changes)
        
        # Ratio of changes (>1 means extreme errors happen more on rapid changes)
        change_ratio = mean_change_before / mean_normal_change if mean_normal_change > 0 else np.inf
    else:
        change_ratio = np.nan
    
    return {
        'extreme_indices': extreme_indices,
        'extreme_errors': extreme_errors,
        'extreme_values_true': extreme_values_true,
        'extreme_values_pred': extreme_values_pred,
        'threshold': threshold,
        'extreme_error_mean': extreme_error_mean,
        'extreme_error_std': extreme_error_std,
        'extreme_value_correlation': extreme_value_correlation,
        'consecutive_errors': consecutive_errors,
        'change_ratio': change_ratio
    }


def analyze_forecast_robustness(model, X_test, y_test, noise_levels=[0.01, 0.05, 0.1], n_iterations=5):
    """
    Analyze forecast robustness to input noise.
    
    Parameters:
    -----------
    model : Model or object with predict method
        Model to evaluate.
    X_test : numpy.ndarray
        Test input data with shape (n_samples, seq_length, n_features).
    y_test : numpy.ndarray
        Test target data.
    noise_levels : list, optional (default=[0.01, 0.05, 0.1])
        Levels of noise to add to test data (as proportion of data standard deviation).
    n_iterations : int, optional (default=5)
        Number of iterations for each noise level.
        
    Returns:
    --------
    robustness_analysis : dict
        Dictionary containing robustness analysis results.
    """
    # Calculate data statistics for scaling noise
    data_std = np.std(X_test)
    
    # Initialize results storage
    robustness_results = {level: {'rmse': [], 'mse': [], 'mae': []} for level in noise_levels}
    
    # Generate baseline prediction
    if hasattr(model, 'model'):
        baseline_pred = model.model.predict(X_test)
    else:
        baseline_pred = model.predict(X_test)
    
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    
    # Test with different noise levels
    for noise_level in noise_levels:
        noise_std = data_std * noise_level
        
        for _ in range(n_iterations):
            # Create noisy input
            noise = np.random.normal(0, noise_std, size=X_test.shape)
            X_noisy = X_test + noise
            
            # Generate prediction
            if hasattr(model, 'model'):
                noisy_pred = model.model.predict(X_noisy)
            else:
                noisy_pred = model.predict(X_noisy)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, noisy_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - noisy_pred))
            
            # Store results
            robustness_results[noise_level]['mse'].append(mse)
            robustness_results[noise_level]['rmse'].append(rmse)
            robustness_results[noise_level]['mae'].append(mae)
    
    # Calculate summary statistics
    summary = {}
    for level in noise_levels:
        summary[level] = {
            'mean_mse': np.mean(robustness_results[level]['mse']),
            'std_mse': np.std(robustness_results[level]['mse']),
            'relative_error_increase': np.mean(robustness_results[level]['mse']) / baseline_mse - 1
        }
    
    return {
        'baseline_mse': baseline_mse,
        'detailed_results': robustness_results,
        'summary': summary
    } 