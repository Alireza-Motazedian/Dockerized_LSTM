"""
Sequence-to-Sequence LSTM model for multi-step time series forecasting.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from ..model_registry import register_model


class Seq2SeqLSTM:
    """
    Sequence-to-Sequence LSTM model for multi-step time series forecasting.
    """
    
    def __init__(self, input_shape=(10, 1), latent_dim=50, dropout_rate=0.2, 
                output_seq_length=5, output_features=1, activation='linear'):
        """
        Initialize a Sequence-to-Sequence LSTM model.
        
        Parameters:
        -----------
        input_shape : tuple, optional (default=(10, 1))
            Shape of input data (sequence_length, features).
        latent_dim : int, optional (default=50)
            Number of LSTM units in the encoder and decoder.
        dropout_rate : float, optional (default=0.2)
            Dropout rate for regularization.
        output_seq_length : int, optional (default=5)
            Length of the output sequence (forecast horizon).
        output_features : int, optional (default=1)
            Number of features to predict at each time step.
        activation : str, optional (default='linear')
            Activation function for the output layer.
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.output_seq_length = output_seq_length
        self.output_features = output_features
        self.activation = activation
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the Sequence-to-Sequence LSTM model.
        
        Returns:
        --------
        model : tensorflow.keras.Model
            Compiled LSTM model.
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # Encoder
        encoder = LSTM(self.latent_dim)(inputs)
        
        # Dropout for regularization
        if self.dropout_rate > 0:
            encoder = Dropout(self.dropout_rate)(encoder)
        
        # Repeat the encoder output for each time step in the output sequence
        repeat = RepeatVector(self.output_seq_length)(encoder)
        
        # Decoder
        decoder = LSTM(self.latent_dim, return_sequences=True)(repeat)
        
        # Dropout for regularization
        if self.dropout_rate > 0:
            decoder = Dropout(self.dropout_rate)(decoder)
        
        # Output layer
        # TimeDistributed applies a Dense layer to each time step in the output sequence
        outputs = TimeDistributed(
            Dense(self.output_features, activation=self.activation)
        )(decoder)
        
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
        self : Seq2SeqLSTM
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
register_model('seq2seq_lstm', Seq2SeqLSTM) 