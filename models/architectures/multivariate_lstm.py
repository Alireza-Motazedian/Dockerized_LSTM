"""
Multivariate LSTM model for time series forecasting with multiple features.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from ..model_registry import register_model


class MultivariateLSTM:
    """
    LSTM model for multivariate time series forecasting.
    """
    
    def __init__(self, input_shape=(10, 5), lstm_units=50, dropout_rate=0.2, 
                output_units=1, activation='linear'):
        """
        Initialize a Multivariate LSTM model.
        
        Parameters:
        -----------
        input_shape : tuple, optional (default=(10, 5))
            Shape of input data (sequence_length, features).
            For multivariate data, the number of features is typically > 1.
        lstm_units : int, optional (default=50)
            Number of LSTM units.
        dropout_rate : float, optional (default=0.2)
            Dropout rate for regularization.
        output_units : int, optional (default=1)
            Number of output units. For multistep forecasting, this would be > 1.
        activation : str, optional (default='linear')
            Activation function for the output layer.
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.output_units = output_units
        self.activation = activation
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the multivariate LSTM model.
        
        Returns:
        --------
        model : tensorflow.keras.Model
            Compiled LSTM model.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # LSTM layer
        x = LSTM(self.lstm_units)(inputs)
        
        # Dropout for regularization
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        
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
        self : MultivariateLSTM
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
register_model('multivariate_lstm', MultivariateLSTM) 