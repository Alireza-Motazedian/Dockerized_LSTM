"""
Preprocessing utilities for time series data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


def normalize_data(data, method='minmax', feature_range=(0, 1)):
    """
    Normalize time series data using the specified method.
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        The time series data to normalize.
    method : str, optional (default='minmax')
        Normalization method. Options: 'minmax', 'standard'.
    feature_range : tuple, optional (default=(0, 1))
        Range for MinMaxScaler output.
        
    Returns:
    --------
    normalized_data : numpy.ndarray
        Normalized data with the same shape as input.
    scaler : object
        Fitted scaler object for later inverse transformation.
    """
    # Convert to numpy array if DataFrame
    is_dataframe = isinstance(data, pd.DataFrame)
    data_values = data.values if is_dataframe else data
    
    # Reshape for 1D arrays (univariate case)
    if len(data_values.shape) == 1:
        data_values = data_values.reshape(-1, 1)
    
    # Apply normalization
    if method.lower() == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method.lower() == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    # Fit and transform
    normalized_data = scaler.fit_transform(data_values)
    
    # Return in the same format as input
    if is_dataframe and len(data.shape) > 1:
        normalized_data = pd.DataFrame(
            normalized_data, 
            index=data.index, 
            columns=data.columns
        )
    elif is_dataframe:
        normalized_data = pd.Series(
            normalized_data.flatten(), 
            index=data.index, 
            name=data.name
        )
    
    return normalized_data, scaler


def create_sequences(data, seq_length, horizon=1, target_column=None):
    """
    Create sequences from time series data for LSTM input.
    
    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        Time series data.
    seq_length : int
        Input sequence length.
    horizon : int, optional (default=1)
        Prediction horizon (how many steps ahead to predict).
    target_column : int or str, optional (default=None)
        For multivariate data, which column to use as the target.
        If None, use the entire data as input and target.
        
    Returns:
    --------
    X : numpy.ndarray
        Input sequences with shape (n_samples, seq_length, n_features).
    y : numpy.ndarray
        Target values with shape (n_samples, 1) or (n_samples, n_targets).
    """
    # Convert to numpy array if DataFrame
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.values
    
    # Ensure 2D data
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    X, y = [], []
    
    # Create sequences
    for i in range(n_samples - seq_length - horizon + 1):
        # Input sequence
        X.append(data[i:(i + seq_length)])
        
        # Target
        if target_column is not None:
            if isinstance(target_column, str) and isinstance(data, pd.DataFrame):
                target_idx = data.columns.get_loc(target_column)
            else:
                target_idx = target_column
            y.append(data[i + seq_length:i + seq_length + horizon, target_idx])
        else:
            y.append(data[i + seq_length:i + seq_length + horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    # For single-step forecasting with a specific target column, reshape y
    if horizon == 1 and target_column is not None:
        y = y.reshape(-1, 1)
    
    return X, y


def train_val_test_split(X, y, val_size=0.2, test_size=0.1, random_state=None, shuffle=False):
    """
    Split data into training, validation, and test sets.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data.
    y : numpy.ndarray
        Target data.
    val_size : float, optional (default=0.2)
        Proportion of data for validation set.
    test_size : float, optional (default=0.1)
        Proportion of data for test set.
    random_state : int, optional (default=None)
        Random seed for reproducibility.
    shuffle : bool, optional (default=False)
        Whether to shuffle the data before splitting.
        Note: For time series, typically we don't shuffle.
        
    Returns:
    --------
    X_train : numpy.ndarray
        Training input data.
    X_val : numpy.ndarray
        Validation input data.
    X_test : numpy.ndarray
        Test input data.
    y_train : numpy.ndarray
        Training target data.
    y_val : numpy.ndarray
        Validation target data.
    y_test : numpy.ndarray
        Test target data.
    """
    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=shuffle
    )
    
    # Second split: create training and validation sets
    # Adjust validation size to account for the test split
    adjusted_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=adjusted_val_size, 
        random_state=random_state,
        shuffle=shuffle
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def detect_stationarity(time_series, test='adfuller', alpha=0.05):
    """
    Detect if a time series is stationary.
    
    Parameters:
    -----------
    time_series : numpy.ndarray or pandas.Series
        The time series to test.
    test : str, optional (default='adfuller')
        Test to use. Options: 'adfuller', 'kpss'.
    alpha : float, optional (default=0.05)
        Significance level.
        
    Returns:
    --------
    is_stationary : bool
        True if time series is stationary.
    pvalue : float
        p-value from the test.
    test_statistic : float
        Test statistic value.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    # Convert to 1D array if needed
    if isinstance(time_series, np.ndarray) and len(time_series.shape) > 1:
        time_series = time_series.flatten()
    
    if test.lower() == 'adfuller':
        # ADF test: null hypothesis is non-stationarity
        result = adfuller(time_series)
        test_statistic, pvalue = result[0], result[1]
        is_stationary = pvalue < alpha
    
    elif test.lower() == 'kpss':
        # KPSS test: null hypothesis is stationarity
        result = kpss(time_series)
        test_statistic, pvalue = result[0], result[1]
        is_stationary = pvalue >= alpha
    
    else:
        raise ValueError(f"Unsupported test: {test}")
    
    return is_stationary, pvalue, test_statistic


def detrend_time_series(time_series, method='diff'):
    """
    Detrend a time series.
    
    Parameters:
    -----------
    time_series : numpy.ndarray or pandas.Series
        The time series to detrend.
    method : str, optional (default='diff')
        Method for detrending. Options: 'diff', 'subtract_mean', 'linear'.
        
    Returns:
    --------
    detrended : numpy.ndarray or pandas.Series
        Detrended time series.
    """
    is_pandas = isinstance(time_series, (pd.Series, pd.DataFrame))
    
    if method == 'diff':
        if is_pandas:
            detrended = time_series.diff().dropna()
        else:
            detrended = np.diff(time_series)
    
    elif method == 'subtract_mean':
        if is_pandas:
            detrended = time_series - time_series.mean()
        else:
            detrended = time_series - np.mean(time_series)
    
    elif method == 'linear':
        import numpy as np
        from scipy import signal
        
        # Convert to numpy if pandas
        if is_pandas:
            values = time_series.values
            index = time_series.index
        else:
            values = time_series
            index = None
        
        # Detrend using linear fit
        detrended_values = signal.detrend(values)
        
        # Convert back to original format
        if is_pandas and isinstance(time_series, pd.Series):
            detrended = pd.Series(detrended_values, index=index, name=time_series.name)
        elif is_pandas and isinstance(time_series, pd.DataFrame):
            detrended = pd.DataFrame(detrended_values, index=index, columns=time_series.columns)
        else:
            detrended = detrended_values
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return detrended 