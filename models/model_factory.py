"""
Factory for creating LSTM model instances.
"""

import json
import os
from tensorflow.keras.models import load_model
from .model_registry import get_model_class


def create_model(architecture_name, **kwargs):
    """
    Create a model instance based on the specified architecture.
    
    Parameters:
    -----------
    architecture_name : str
        Name of the architecture to instantiate.
    **kwargs : dict
        Additional arguments to pass to the model constructor.
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Instantiated model.
    """
    model_class = get_model_class(architecture_name)
    if model_class is None:
        raise ValueError(f"Unknown architecture: {architecture_name}")
    
    return model_class(**kwargs)


def create_model_from_config(config):
    """
    Create a model instance from a configuration dictionary or file.
    
    Parameters:
    -----------
    config : dict or str
        Configuration dictionary or path to a JSON configuration file.
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Instantiated model.
    """
    # If config is a file path, load it
    if isinstance(config, str):
        if not os.path.exists(config):
            raise FileNotFoundError(f"Configuration file not found: {config}")
        
        with open(config, 'r') as f:
            config = json.load(f)
    
    # Extract architecture and other parameters
    architecture = config.get('architecture')
    if not architecture:
        raise ValueError("Configuration must specify 'architecture'")
    
    # Extract model parameters
    model_params = {}
    for key, value in config.items():
        # Skip non-model parameters
        if key not in ['architecture', 'compile', 'training']:
            model_params[key] = value
    
    # Create the model
    model = create_model(architecture, **model_params)
    
    # Compile the model if compilation settings are provided
    if 'compile' in config:
        compile_params = config['compile']
        optimizer_name = compile_params.get('optimizer', 'adam')
        learning_rate = compile_params.get('learning_rate', 0.001)
        
        # Configure optimizer with learning rate
        if optimizer_name.lower() == 'adam':
            from tensorflow.keras.optimizers import Adam
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'rmsprop':
            from tensorflow.keras.optimizers import RMSprop
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            from tensorflow.keras.optimizers import SGD
            optimizer = SGD(learning_rate=learning_rate)
        else:
            optimizer = optimizer_name
        
        model.compile(
            optimizer=optimizer,
            loss=compile_params.get('loss', 'mse'),
            metrics=compile_params.get('metrics', ['mae'])
        )
    
    return model


def load_saved_model(model_path):
    """
    Load a saved model from file.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file.
        
    Returns:
    --------
    model : tensorflow.keras.Model
        Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return load_model(model_path)


def get_best_model():
    """
    Get the best model based on validation loss.
    
    Returns:
    --------
    model : tensorflow.keras.Model
        Best model.
    """
    best_model_path = os.path.join(os.path.dirname(__file__), 'lstm_model_best.h5')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model file not found: {best_model_path}")
    
    return load_saved_model(best_model_path)


def get_lstm_layer_model(model, return_sequences=True, return_state=True):
    """
    Create a model that outputs LSTM layer activations and states.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        LSTM model.
    return_sequences : bool, optional (default=True)
        Whether to return sequences from the LSTM layer.
    return_state : bool, optional (default=True)
        Whether to return states from the LSTM layer.
        
    Returns:
    --------
    lstm_layer_model : tensorflow.keras.Model
        Model that outputs LSTM layer activations and states.
    """
    from tensorflow.keras.models import Model
    
    # Find the LSTM layer
    lstm_layer = None
    for layer in model.layers:
        if 'lstm' in layer.name.lower():
            lstm_layer = layer
            break
    
    if lstm_layer is None:
        raise ValueError("No LSTM layer found in the model")
    
    # Create a new model that returns LSTM layer output and states
    from tensorflow.keras.layers import LSTM, Input
    
    # Get the input shape from the model
    input_shape = model.input_shape[1:]
    
    # Create a new input layer
    input_layer = Input(shape=input_shape)
    
    # Get LSTM layer configuration
    config = lstm_layer.get_config()
    units = config['units']
    
    # Create a new LSTM layer with return_sequences and return_state set
    lstm = LSTM(
        units=units,
        return_sequences=return_sequences,
        return_state=return_state
    )(input_layer)
    
    # Create the model
    if return_state:
        lstm_layer_model = Model(inputs=input_layer, outputs=lstm)
    else:
        lstm_layer_model = Model(inputs=input_layer, outputs=[lstm])
    
    # Copy weights from the original model's LSTM layer
    lstm_layer_weights = lstm_layer.get_weights()
    lstm_layer_model.layers[1].set_weights(lstm_layer_weights)
    
    return lstm_layer_model 