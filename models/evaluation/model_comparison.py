"""
Utilities for comparing different forecasting models.
"""

import numpy as np
import pandas as pd
from utils.metrics import calculate_all_metrics


def compare_models(models, X_test, y_test, scaler=None):
    """
    Compare multiple models on the same test data.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to compare. Keys are model names, values are model objects.
    X_test : numpy.ndarray
        Test input data.
    y_test : numpy.ndarray
        Test target data.
    scaler : object, optional (default=None)
        Scaler object for inverse transformation of predictions.
        
    Returns:
    --------
    comparison_results : dict
        Dictionary containing comparison results for each model.
    """
    # Dictionary to store results
    comparison_results = {}
    
    # Evaluate each model
    for name, model in models.items():
        print(f"Evaluating model: {name}")
        
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
            
            # Calculate metrics on inverse-transformed data
            metrics = calculate_all_metrics(y_test_inv, y_pred_inv)
            
            # Store results
            comparison_results[name] = {
                'metrics': metrics,
                'predictions': y_pred_inv
            }
        else:
            # Calculate metrics on original scale
            metrics = calculate_all_metrics(y_test, y_pred)
            
            # Store results
            comparison_results[name] = {
                'metrics': metrics,
                'predictions': y_pred
            }
    
    return comparison_results


def compare_models_across_datasets(models, datasets, scaler=None):
    """
    Compare multiple models across multiple datasets.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to compare. Keys are model names, values are model objects.
    datasets : dict
        Dictionary of datasets to evaluate on. Keys are dataset names, values are tuples of (X_test, y_test).
    scaler : object or dict, optional (default=None)
        Scaler object for inverse transformation of predictions. If dict, keys should match dataset names.
        
    Returns:
    --------
    comparison_results : dict
        Dictionary containing comparison results for each model on each dataset.
    """
    # Dictionary to store results
    comparison_results = {}
    
    # Evaluate each model on each dataset
    for dataset_name, (X_test, y_test) in datasets.items():
        print(f"Evaluating on dataset: {dataset_name}")
        
        # Get the appropriate scaler for this dataset
        dataset_scaler = None
        if scaler is not None:
            if isinstance(scaler, dict):
                dataset_scaler = scaler.get(dataset_name)
            else:
                dataset_scaler = scaler
        
        # Compare models on this dataset
        dataset_results = compare_models(models, X_test, y_test, dataset_scaler)
        
        # Store results
        comparison_results[dataset_name] = dataset_results
    
    return comparison_results


def compare_with_traditional_models(lstm_model, traditional_models, X_test, y_test, scaler=None):
    """
    Compare LSTM model with traditional forecasting models.
    
    Parameters:
    -----------
    lstm_model : object
        LSTM model to compare.
    traditional_models : dict
        Dictionary of traditional models to compare. Keys are model names, values are model objects.
    X_test : numpy.ndarray
        Test input data.
    y_test : numpy.ndarray
        Test target data.
    scaler : object, optional (default=None)
        Scaler object for inverse transformation of predictions.
        
    Returns:
    --------
    comparison_results : dict
        Dictionary containing comparison results for each model.
    """
    # Combine LSTM and traditional models
    all_models = {'LSTM': lstm_model}
    all_models.update(traditional_models)
    
    # Compare all models
    return compare_models(all_models, X_test, y_test, scaler)


def create_comparison_table(comparison_results, metrics=['mse', 'rmse', 'mae', 'mape']):
    """
    Create a comparison table from model comparison results.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary containing comparison results from compare_models().
    metrics : list, optional (default=['mse', 'rmse', 'mae', 'mape'])
        List of metrics to include in the table.
        
    Returns:
    --------
    comparison_table : pandas.DataFrame
        DataFrame containing comparison metrics for each model.
    """
    # Dictionary to store table data
    table_data = {metric: [] for metric in metrics}
    table_data['Model'] = []
    
    # Extract metrics for each model
    for model_name, results in comparison_results.items():
        table_data['Model'].append(model_name)
        
        for metric in metrics:
            if metric in results['metrics']:
                table_data[metric].append(results['metrics'][metric])
            else:
                table_data[metric].append(np.nan)
    
    # Create DataFrame
    comparison_table = pd.DataFrame(table_data)
    
    # Set Model as index
    comparison_table.set_index('Model', inplace=True)
    
    return comparison_table


def create_multi_dataset_comparison_table(comparison_results, metrics=['mse', 'rmse', 'mae', 'mape']):
    """
    Create a comparison table for multiple datasets.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary containing comparison results from compare_models_across_datasets().
    metrics : list, optional (default=['mse', 'rmse', 'mae', 'mape'])
        List of metrics to include in the table.
        
    Returns:
    --------
    comparison_table : pandas.DataFrame
        DataFrame containing comparison metrics for each model on each dataset.
    """
    # Dictionary to store table data
    table_data = []
    
    # Extract metrics for each dataset and model
    for dataset_name, dataset_results in comparison_results.items():
        for model_name, model_results in dataset_results.items():
            # Create a row for this dataset and model
            row = {
                'Dataset': dataset_name,
                'Model': model_name
            }
            
            # Add metrics
            for metric in metrics:
                if metric in model_results['metrics']:
                    row[metric] = model_results['metrics'][metric]
                else:
                    row[metric] = np.nan
            
            # Add row to table data
            table_data.append(row)
    
    # Create DataFrame
    comparison_table = pd.DataFrame(table_data)
    
    return comparison_table


def perform_statistical_tests(comparison_results, reference_model=None, alpha=0.05):
    """
    Perform statistical tests to compare models.
    
    Parameters:
    -----------
    comparison_results : dict
        Dictionary containing comparison results from compare_models().
    reference_model : str, optional (default=None)
        Name of the reference model to compare others against.
        If None, the first model in the dictionary is used.
    alpha : float, optional (default=0.05)
        Significance level for the statistical tests.
        
    Returns:
    --------
    test_results : dict
        Dictionary containing statistical test results.
    """
    from scipy import stats
    
    # Get predictions for each model
    model_predictions = {}
    for model_name, results in comparison_results.items():
        model_predictions[model_name] = results['predictions']
    
    # Get reference model name if not provided
    if reference_model is None:
        reference_model = list(comparison_results.keys())[0]
    
    # Check if reference model exists
    if reference_model not in model_predictions:
        raise ValueError(f"Reference model '{reference_model}' not found in comparison results")
    
    # Dictionary to store test results
    test_results = {}
    
    # Get reference model predictions
    reference_preds = model_predictions[reference_model]
    
    # Perform tests for each model against the reference
    for model_name, preds in model_predictions.items():
        # Skip reference model
        if model_name == reference_model:
            continue
        
        # Flatten predictions if needed
        ref_flat = reference_preds.flatten()
        model_flat = preds.flatten()
        
        # Perform Wilcoxon signed-rank test
        wilcoxon_result = stats.wilcoxon(ref_flat, model_flat)
        
        # Perform paired t-test
        ttest_result = stats.ttest_rel(ref_flat, model_flat)
        
        # Store results
        test_results[model_name] = {
            'wilcoxon_statistic': wilcoxon_result.statistic,
            'wilcoxon_pvalue': wilcoxon_result.pvalue,
            'wilcoxon_significant': wilcoxon_result.pvalue < alpha,
            'ttest_statistic': ttest_result.statistic,
            'ttest_pvalue': ttest_result.pvalue,
            'ttest_significant': ttest_result.pvalue < alpha
        }
    
    return test_results 