"""
Custom callbacks for LSTM model training.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau, 
    TensorBoard
)


def create_callbacks(checkpoint_path=None, early_stopping=True, patience=10, 
                    reduce_lr=True, lr_patience=5, lr_factor=0.5, lr_min=1e-6,
                    tensorboard=False, log_dir='logs'):
    """
    Create a list of callbacks for model training.
    
    Parameters:
    -----------
    checkpoint_path : str, optional (default=None)
        Path to save model checkpoints.
        If None, no checkpoint will be saved.
    early_stopping : bool, optional (default=True)
        Whether to use early stopping.
    patience : int, optional (default=10)
        Number of epochs with no improvement after which training will be stopped.
    reduce_lr : bool, optional (default=True)
        Whether to reduce learning rate when a metric has stopped improving.
    lr_patience : int, optional (default=5)
        Number of epochs with no improvement after which learning rate will be reduced.
    lr_factor : float, optional (default=0.5)
        Factor by which the learning rate will be reduced.
    lr_min : float, optional (default=1e-6)
        Lower bound on the learning rate.
    tensorboard : bool, optional (default=False)
        Whether to use TensorBoard callback.
    log_dir : str, optional (default='logs')
        Log directory for TensorBoard.
        
    Returns:
    --------
    callbacks : list
        List of callbacks.
    """
    callbacks = []
    
    # Model checkpoint
    if checkpoint_path:
        # Ensure directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                save_weights_only=False,
                verbose=1
            )
        )
    
    # Early stopping
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                mode='min',
                restore_best_weights=True,
                verbose=1
            )
        )
    
    # Reduce learning rate
    if reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_factor,
                patience=lr_patience,
                min_lr=lr_min,
                mode='min',
                verbose=1
            )
        )
    
    # TensorBoard
    if tensorboard:
        # Ensure log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        callbacks.append(
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        )
    
    return callbacks


class SaveForecastCallback(tf.keras.callbacks.Callback):
    """
    Callback to save model forecasts during training.
    """
    
    def __init__(self, validation_data, save_dir='forecasts', save_freq=5):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        validation_data : tuple
            Tuple of (X_val, y_val) for generating forecasts.
        save_dir : str, optional (default='forecasts')
            Directory to save forecasts.
        save_freq : int, optional (default=5)
            Frequency (in epochs) at which to save forecasts.
        """
        super(SaveForecastCallback, self).__init__()
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]
        self.save_dir = save_dir
        self.save_freq = save_freq
        
        # Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Save forecasts at the end of epochs that are multiples of save_freq.
        
        Parameters:
        -----------
        epoch : int
            Current epoch.
        logs : dict, optional (default=None)
            Dictionary of metrics for the current epoch.
        """
        # Skip if not a multiple of save_freq
        if (epoch + 1) % self.save_freq != 0:
            return
        
        # Generate predictions
        y_pred = self.model.predict(self.X_val)
        
        # Save predictions
        import numpy as np
        save_path = os.path.join(self.save_dir, f'forecast_epoch_{epoch+1}.npy')
        np.save(save_path, y_pred)
        
        # Optionally plot and save visualizations
        try:
            import matplotlib.pyplot as plt
            
            # Plot sample forecasts vs actuals
            plt.figure(figsize=(12, 6))
            
            # Get a sample for visualization (first sequence)
            y_true_sample = self.y_val[0].flatten()
            y_pred_sample = y_pred[0].flatten()
            
            plt.plot(y_true_sample, label='Actual')
            plt.plot(y_pred_sample, label='Forecast')
            plt.title(f'Forecast vs Actual (Epoch {epoch+1})')
            plt.legend()
            
            # Save plot
            plot_path = os.path.join(self.save_dir, f'forecast_plot_epoch_{epoch+1}.png')
            plt.savefig(plot_path)
            plt.close()
        
        except Exception as e:
            # Skip visualization if there's an error
            print(f"Error generating visualization: {e}")


class LossHistory(tf.keras.callbacks.Callback):
    """
    Callback to track and save training history.
    """
    
    def __init__(self, save_dir='history'):
        """
        Initialize the callback.
        
        Parameters:
        -----------
        save_dir : str, optional (default='history')
            Directory to save history.
        """
        super(LossHistory, self).__init__()
        self.save_dir = save_dir
        self.losses = []
        self.val_losses = []
        
        # Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Save losses at the end of each epoch.
        
        Parameters:
        -----------
        epoch : int
            Current epoch.
        logs : dict, optional (default=None)
            Dictionary of metrics for the current epoch.
        """
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        # Save history as numpy arrays
        import numpy as np
        np.save(os.path.join(self.save_dir, 'training_loss.npy'), np.array(self.losses))
        np.save(os.path.join(self.save_dir, 'validation_loss.npy'), np.array(self.val_losses))
        
        # Plot and save loss curves
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses, label='Training Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'))
            plt.close()
        
        except Exception as e:
            # Skip visualization if there's an error
            print(f"Error generating loss plot: {e}") 