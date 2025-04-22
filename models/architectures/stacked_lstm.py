"""
Stacked LSTM model for time series forecasting.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from ..model_registry import register_model


class StackedLSTM:
    """
    Multi-layer LSTM model for more complex time series forecasting.
    """
    
    def __init__(self, input_shape=(10, 1), lstm_units=[50, 25], dropout_rate=0.2, 
                output_units=1, activation='linear'):
        """
        Initialize a Stacked LSTM model.
        
        Parameters:
        -----------
        input_shape : tuple, optional (default=(10, 1))
            Shape of input data (sequence_length, features).
        lstm_units : list, optional (default=[50, 25])
            List of units for each LSTM layer.
        dropout_rate : float or list, optional (default=0.2)
            Dropout rate(s) for regularization.
            If float, the same rate is used for all layers.
            If list, must match the length of lstm_units.
        output_units : int, optional (default=1)
            Number of output units.
        activation : str, optional (default='linear')
            Activation function for the output layer.
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        
        # If dropout_rate is a single value, replicate it for each layer
        if isinstance(dropout_rate, (int, float)):
            self.dropout_rate = [dropout_rate] * len(lstm_units)
        else:
            if len(dropout_rate) != len(lstm_units):
                raise ValueError("If dropout_rate is a list, it must have the same length as lstm_units")
            self.dropout_rate = dropout_rate
        
        self.output_units = output_units
        self.activation = activation
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the stacked LSTM model.
        
        Returns:
        --------
        model : tensorflow.keras.Model
            Compiled LSTM model.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # First LSTM layer
        x = LSTM(
            self.lstm_units[0],
            return_sequences=len(self.lstm_units) > 1  # Return sequences for all but the last layer
        )(inputs)
        
        # Apply dropout if needed
        if self.dropout_rate[0] > 0:
            x = Dropout(self.dropout_rate[0])(x)
        
        # Rest of the LSTM layers
        for i in range(1, len(self.lstm_units)):
            return_sequences = i < len(self.lstm_units) - 1  # Return sequences for all but the last layer
            x = LSTM(self.lstm_units[i], return_sequences=return_sequences)(x)
            
            # Apply dropout if needed
            if self.dropout_rate[i] > 0:
                x = Dropout(self.dropout_rate[i])(x)
        
        # Output layer
        outputs = Dense(self.output_units, activation=self.activation)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def compile(self, optimizer='adam', loss='mse', metrics=['mae']):
        """
        Compile the model.
        
        Parameters:
        -----------
        optimizer : str or tensorflow.keras.optimizers.Optimizer, optional (default='adam')
            Optimizer for training.
        loss : str or tensorflow.keras.losses.Loss, optional (default='mse')
            Loss function.
        metrics : list, optional (default=['mae'])
            Metrics to evaluate during training and testing.
        
        Returns:
        --------
        self : StackedLSTM
            Self for method chaining.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self
    
    def summary(self):
        """
        Print a summary of the model.
        
        Returns:
        --------
        summary : str
            Model summary.
        """
        return self.model.summary()
    
    def fit(self, *args, **kwargs):
        """
        Train the model.
        
        Parameters:
        -----------
        *args : tuple
            Arguments to pass to keras.Model.fit().
        **kwargs : dict
            Keyword arguments to pass to keras.Model.fit().
        
        Returns:
        --------
        history : tensorflow.keras.callbacks.History
            Training history.
        """
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        """
        Generate predictions.
        
        Parameters:
        -----------
        *args : tuple
            Arguments to pass to keras.Model.predict().
        **kwargs : dict
            Keyword arguments to pass to keras.Model.predict().
        
        Returns:
        --------
        predictions : numpy.ndarray
            Model predictions.
        """
        return self.model.predict(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model.
        
        Parameters:
        -----------
        *args : tuple
            Arguments to pass to keras.Model.evaluate().
        **kwargs : dict
            Keyword arguments to pass to keras.Model.evaluate().
        
        Returns:
        --------
        evaluation : scalar or list
            Model evaluation results.
        """
        return self.model.evaluate(*args, **kwargs)
    
    def save(self, filepath, *args, **kwargs):
        """
        Save the model.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model.
        *args : tuple
            Arguments to pass to keras.Model.save().
        **kwargs : dict
            Keyword arguments to pass to keras.Model.save().
        """
        self.model.save(filepath, *args, **kwargs)


# Register the model
register_model('stacked_lstm', StackedLSTM) 