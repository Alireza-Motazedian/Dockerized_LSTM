"""
Training utilities for LSTM models.
"""

import os
import time
import numpy as np
import tensorflow as tf
from .callbacks import create_callbacks


def train_model(model, X_train, y_train, X_val=None, y_val=None, validation_split=0.2,
               epochs=100, batch_size=32, callbacks=None, verbose=1, shuffle=True,
               early_stopping=True, patience=10, checkpoint_path=None, tensorboard=False,
               log_dir='logs', save_best_model=True, save_history=True, history_dir='history'):
    """
    Train an LSTM model with standard best practices.
    
    Parameters:
    -----------
    model : Model or object with fit method
        Model to train.
    X_train : numpy.ndarray
        Training input data.
    y_train : numpy.ndarray
        Training target data.
    X_val : numpy.ndarray, optional (default=None)
        Validation input data. If None, validation_split is used.
    y_val : numpy.ndarray, optional (default=None)
        Validation target data. If None, validation_split is used.
    validation_split : float, optional (default=0.2)
        Fraction of training data to use for validation if X_val and y_val are None.
    epochs : int, optional (default=100)
        Number of epochs to train for.
    batch_size : int, optional (default=32)
        Batch size for training.
    callbacks : list, optional (default=None)
        List of callbacks to use during training.
    verbose : int, optional (default=1)
        Verbosity mode (0, 1, or 2).
    shuffle : bool, optional (default=True)
        Whether to shuffle the training data before each epoch.
    early_stopping : bool, optional (default=True)
        Whether to use early stopping.
    patience : int, optional (default=10)
        Number of epochs with no improvement after which training will be stopped.
    checkpoint_path : str, optional (default=None)
        Path to save model checkpoints.
    tensorboard : bool, optional (default=False)
        Whether to use TensorBoard.
    log_dir : str, optional (default='logs')
        Log directory for TensorBoard.
    save_best_model : bool, optional (default=True)
        Whether to save the best model after training.
    save_history : bool, optional (default=True)
        Whether to save training history.
    history_dir : str, optional (default='history')
        Directory to save training history.
        
    Returns:
    --------
    history : dict
        Training history.
    """
    # Create directory for best model if needed
    if save_best_model and checkpoint_path is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(os.path.dirname(model_dir), 'lstm_model_best.h5')
        checkpoint_path = model_dir
    
    # Create callbacks if not provided
    if callbacks is None:
        callbacks = create_callbacks(
            checkpoint_path=checkpoint_path,
            early_stopping=early_stopping,
            patience=patience,
            tensorboard=tensorboard,
            log_dir=log_dir
        )
    
    # Set validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        validation_split = None
    
    # Record training start time
    start_time = time.time()
    
    # Train the model
    if hasattr(model, 'model'):
        # Handle our model wrapper classes
        history = model.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle
        )
    else:
        # Handle raw Keras models
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=shuffle
        )
    
    # Record training end time
    training_time = time.time() - start_time
    
    # Print training summary
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Total epochs: {len(history.history['loss'])}")
    
    if 'val_loss' in history.history:
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = np.min(history.history['val_loss'])
        print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    
    # Save the final model if not saving checkpoints
    if save_best_model and checkpoint_path is None:
        if hasattr(model, 'model'):
            model.model.save(os.path.join(os.path.dirname(__file__), '..', 'lstm_model_final.h5'))
        else:
            model.save(os.path.join(os.path.dirname(__file__), '..', 'lstm_model_final.h5'))
    
    # Save training history
    if save_history:
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        
        # Save as numpy arrays
        for metric, values in history.history.items():
            np.save(os.path.join(history_dir, f'{metric}.npy'), np.array(values))
    
    return history


def train_with_cross_validation(model_factory, X, y, n_splits=5, epochs=100, batch_size=32, 
                              callbacks=None, verbose=1, return_models=False, random_state=None):
    """
    Train a model using k-fold cross-validation.
    
    Parameters:
    -----------
    model_factory : function
        Function that returns a new model instance.
    X : numpy.ndarray
        Input data.
    y : numpy.ndarray
        Target data.
    n_splits : int, optional (default=5)
        Number of cross-validation folds.
    epochs : int, optional (default=100)
        Number of epochs to train for.
    batch_size : int, optional (default=32)
        Batch size for training.
    callbacks : list, optional (default=None)
        List of callbacks to use during training.
    verbose : int, optional (default=1)
        Verbosity mode (0, 1, or 2).
    return_models : bool, optional (default=False)
        Whether to return trained models.
    random_state : int, optional (default=None)
        Random seed for reproducibility.
        
    Returns:
    --------
    results : dict
        Cross-validation results.
    """
    from sklearn.model_selection import KFold
    
    # Create K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # List to store results
    fold_histories = []
    fold_models = []
    fold_scores = []
    
    # Train and evaluate the model for each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create a new model
        model = model_factory()
        
        # Train the model
        history = train_model(
            model,
            X_train_fold, y_train_fold,
            X_val=X_val_fold, y_val=y_val_fold,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            checkpoint_path=None,  # Don't save checkpoints for CV
            save_best_model=False,  # Don't save models for CV
            save_history=False      # Don't save history for CV
        )
        
        # Evaluate the model
        if hasattr(model, 'model'):
            score = model.model.evaluate(X_val_fold, y_val_fold, verbose=0)
        else:
            score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        
        fold_histories.append(history.history)
        fold_scores.append(score)
        
        if return_models:
            fold_models.append(model)
    
    # Calculate average score across folds
    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\nCross-validation results:")
    print(f"Average score: {avg_score:.6f} ± {std_score:.6f}")
    
    # Return results
    results = {
        'histories': fold_histories,
        'scores': fold_scores,
        'avg_score': avg_score,
        'std_score': std_score
    }
    
    if return_models:
        results['models'] = fold_models
    
    return results


def train_with_learning_curve(model_factory, X_train, y_train, X_val, y_val,
                             train_sizes=np.linspace(0.1, 1.0, 5), epochs=100,
                             batch_size=32, callbacks=None, verbose=1):
    """
    Train a model with different training set sizes to create a learning curve.
    
    Parameters:
    -----------
    model_factory : function
        Function that returns a new model instance.
    X_train : numpy.ndarray
        Training input data.
    y_train : numpy.ndarray
        Training target data.
    X_val : numpy.ndarray
        Validation input data.
    y_val : numpy.ndarray
        Validation target data.
    train_sizes : numpy.ndarray, optional (default=np.linspace(0.1, 1.0, 5))
        Relative or absolute training set sizes.
    epochs : int, optional (default=100)
        Number of epochs to train for.
    batch_size : int, optional (default=32)
        Batch size for training.
    callbacks : list, optional (default=None)
        List of callbacks to use during training.
    verbose : int, optional (default=1)
        Verbosity mode (0, 1, or 2).
        
    Returns:
    --------
    results : dict
        Learning curve results.
    """
    from sklearn.model_selection import train_test_split
    
    # Calculate absolute training set sizes
    n_samples = X_train.shape[0]
    train_sizes_abs = np.array([int(n_samples * size) for size in train_sizes])
    
    # Lists to store results
    train_scores = []
    val_scores = []
    histories = []
    
    # Train and evaluate the model for each training set size
    for size in train_sizes_abs:
        print(f"\nTraining with {size} samples")
        
        # Sample training data
        if size < n_samples:
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train, y_train, train_size=size, random_state=42
            )
        else:
            X_train_sample, y_train_sample = X_train, y_train
        
        # Create a new model
        model = model_factory()
        
        # Train the model
        history = train_model(
            model,
            X_train_sample, y_train_sample,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            checkpoint_path=None,  # Don't save checkpoints for learning curve
            save_best_model=False,  # Don't save models for learning curve
            save_history=False      # Don't save history for learning curve
        )
        
        # Evaluate the model
        if hasattr(model, 'model'):
            train_score = model.model.evaluate(X_train_sample, y_train_sample, verbose=0)
            val_score = model.model.evaluate(X_val, y_val, verbose=0)
        else:
            train_score = model.evaluate(X_train_sample, y_train_sample, verbose=0)
            val_score = model.evaluate(X_val, y_val, verbose=0)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        histories.append(history.history)
    
    # Return results
    return {
        'train_sizes': train_sizes_abs,
        'train_scores': train_scores,
        'val_scores': val_scores,
        'histories': histories
    } 