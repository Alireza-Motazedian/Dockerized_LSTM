"""
Visualization utilities for LSTM time series forecasting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd


def set_plotting_style():
    """Set the default plotting style for visualizations."""
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 2


def plot_time_series(time_series, title="Time Series Data", xlabel="Time", ylabel="Value", 
                     figsize=(12, 6), ax=None, save_path=None):
    """
    Plot a time series.
    
    Parameters:
    -----------
    time_series : array-like or pandas.Series
        Time series data to plot.
    title : str, optional (default="Time Series Data")
        Plot title.
    xlabel : str, optional (default="Time")
        X-axis label.
    ylabel : str, optional (default="Value")
        Y-axis label.
    figsize : tuple, optional (default=(12, 6))
        Figure size.
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes to plot on. If None, a new figure and axes are created.
    save_path : str, optional (default=None)
        Path to save the plot. If None, the plot is not saved.
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    set_plotting_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(time_series, pd.Series) or isinstance(time_series, pd.DataFrame):
        time_series.plot(ax=ax)
    else:
        ax.plot(time_series)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return ax


def plot_forecast(y_true, y_pred, title="LSTM Forecast vs Actual", 
                  xlabel="Time Steps", ylabel="Value", figsize=(12, 6), 
                  ax=None, save_path=None, confidence_intervals=None):
    """
    Plot forecast values against actual values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values (forecast).
    title : str, optional (default="LSTM Forecast vs Actual")
        Plot title.
    xlabel : str, optional (default="Time Steps")
        X-axis label.
    ylabel : str, optional (default="Value")
        Y-axis label.
    figsize : tuple, optional (default=(12, 6))
        Figure size.
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes to plot on. If None, a new figure and axes are created.
    save_path : str, optional (default=None)
        Path to save the plot. If None, the plot is not saved.
    confidence_intervals : tuple, optional (default=None)
        Lower and upper confidence intervals as (lower, upper).
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    set_plotting_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure numpy arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Plot actual values
    ax.plot(y_true, label='Actual', color='#1f77b4', linewidth=2)
    
    # Plot predicted values
    ax.plot(y_pred, label='Forecast', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        x = np.arange(len(y_pred))
        ax.fill_between(x, lower, upper, color='#ff7f0e', alpha=0.2, label='95% Confidence Interval')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return ax


def plot_training_history(history, metrics=['loss', 'val_loss'], 
                          title="Training History", figsize=(12, 6), 
                          ax=None, save_path=None):
    """
    Plot training history from model.fit().
    
    Parameters:
    -----------
    history : dict or tensorflow.keras.callbacks.History
        Training history from model.fit().
    metrics : list, optional (default=['loss', 'val_loss'])
        Metrics to plot.
    title : str, optional (default="Training History")
        Plot title.
    figsize : tuple, optional (default=(12, 6))
        Figure size.
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes to plot on. If None, a new figure and axes are created.
    save_path : str, optional (default=None)
        Path to save the plot. If None, the plot is not saved.
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    set_plotting_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Handle tensorflow History object
    if not isinstance(history, dict):
        history = history.history
    
    for metric in metrics:
        if metric in history:
            ax.plot(history[metric], label=metric)
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='best')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return ax


def plot_prediction_errors(y_true, y_pred, title="Prediction Errors", 
                          figsize=(15, 10), save_path=None):
    """
    Create plots for analyzing prediction errors.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    title : str, optional (default="Prediction Errors")
        Base title for plots.
    figsize : tuple, optional (default=(15, 10))
        Figure size.
    save_path : str, optional (default=None)
        Path to save the plot. If None, the plot is not saved.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    """
    set_plotting_style()
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Error histogram
    axes[0, 0].hist(errors, bins=20, color='#1f77b4', alpha=0.7)
    axes[0, 0].set_title('Error Distribution')
    axes[0, 0].set_xlabel('Error')
    axes[0, 0].set_ylabel('Frequency')
    
    # Scatter plot of actual vs predicted
    axes[0, 1].scatter(y_true, y_pred, alpha=0.5, color='#1f77b4')
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 1].set_title('Actual vs Predicted')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    
    # Time series of errors
    axes[1, 0].plot(errors, color='#1f77b4')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_title('Errors Over Time')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Error')
    
    # QQ plot of errors
    from scipy import stats
    stats.probplot(errors, plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Errors')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_correlation_matrix(data, title="Feature Correlation Matrix", 
                           figsize=(10, 8), save_path=None):
    """
    Plot correlation matrix for multivariate time series data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Multivariate time series data.
    title : str, optional (default="Feature Correlation Matrix")
        Plot title.
    figsize : tuple, optional (default=(10, 8))
        Figure size.
    save_path : str, optional (default=None)
        Path to save the plot. If None, the plot is not saved.
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    set_plotting_style()
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame for correlation matrix plot")
    
    # Calculate correlation matrix
    corr = data.corr()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw heatmap with mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", ax=ax)
    
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return ax


def plot_lstm_states(lstm_layer_model, X_sample, cell_states=True, hidden_states=True,
                    title="LSTM Internal States", figsize=(15, 10), save_path=None):
    """
    Visualize LSTM internal states for a sample input sequence.
    
    Parameters:
    -----------
    lstm_layer_model : keras.Model
        Keras model with LSTM layer set to return_state=True.
    X_sample : array-like
        Sample input sequence with shape (1, seq_length, n_features).
    cell_states : bool, optional (default=True)
        Whether to plot cell states.
    hidden_states : bool, optional (default=True)
        Whether to plot hidden states.
    title : str, optional (default="LSTM Internal States")
        Plot title.
    figsize : tuple, optional (default=(15, 10))
        Figure size.
    save_path : str, optional (default=None)
        Path to save the plot. If None, the plot is not saved.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    """
    set_plotting_style()
    
    # Make prediction with states
    states = lstm_layer_model.predict(X_sample)
    
    # First element is the output, then cell states and hidden states
    output = states[0]
    
    # Number of subplots depends on what states to visualize
    n_plots = 1  # For output
    if cell_states:
        n_plots += 1
    if hidden_states:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16)
    
    # Plot LSTM output
    plot_idx = 0
    seq_length = output.shape[1]
    n_units = output.shape[2]
    
    # Create a heatmap for the output
    im = axes[plot_idx].imshow(output[0].T, aspect='auto', cmap='viridis')
    axes[plot_idx].set_title('LSTM Layer Output')
    axes[plot_idx].set_xlabel('Time Step')
    axes[plot_idx].set_ylabel('Unit')
    axes[plot_idx].set_yticks(range(n_units))
    fig.colorbar(im, ax=axes[plot_idx], label='Activation')
    
    # Plot cell states if requested
    if cell_states:
        plot_idx += 1
        cell_state = states[1]  # Cell state is the second element
        im = axes[plot_idx].imshow(cell_state[0].reshape(1, -1), aspect='auto', cmap='coolwarm')
        axes[plot_idx].set_title('Cell State (final time step)')
        axes[plot_idx].set_xlabel('Unit')
        axes[plot_idx].set_yticks([])
        fig.colorbar(im, ax=axes[plot_idx], label='State Value')
    
    # Plot hidden states if requested
    if hidden_states:
        plot_idx += 1
        hidden_state = states[2]  # Hidden state is the third element
        im = axes[plot_idx].imshow(hidden_state[0].reshape(1, -1), aspect='auto', cmap='coolwarm')
        axes[plot_idx].set_title('Hidden State (final time step)')
        axes[plot_idx].set_xlabel('Unit')
        axes[plot_idx].set_yticks([])
        fig.colorbar(im, ax=axes[plot_idx], label='State Value')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_forecast_vs_traditional(y_true, lstm_forecast, traditional_forecasts, 
                                title="LSTM vs Traditional Methods", 
                                figsize=(12, 6), save_path=None):
    """
    Compare LSTM forecast with traditional forecasting methods.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values.
    lstm_forecast : array-like
        LSTM model forecast.
    traditional_forecasts : dict
        Dictionary of traditional model forecasts. 
        Key is model name, value is forecast.
    title : str, optional
        Plot title.
    figsize : tuple, optional (default=(12, 6))
        Figure size.
    save_path : str, optional (default=None)
        Path to save the plot. If None, the plot is not saved.
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten arrays
    y_true = np.array(y_true).flatten()
    lstm_forecast = np.array(lstm_forecast).flatten()
    
    # Plot actual values
    ax.plot(y_true, label='Actual', color='black', linewidth=2)
    
    # Plot LSTM forecast
    ax.plot(lstm_forecast, label='LSTM', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Plot traditional methods
    colors = plt.cm.tab10.colors
    for i, (name, forecast) in enumerate(traditional_forecasts.items()):
        forecast = np.array(forecast).flatten()
        ax.plot(forecast, label=name, color=colors[i+2], linewidth=1.5, linestyle='-.')
    
    ax.set_title(title)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return ax 