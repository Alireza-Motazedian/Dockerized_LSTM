"""
Utilities for analyzing and visualizing LSTM internal states.
"""

import numpy as np
import tensorflow as tf
from ..model_factory import get_lstm_layer_model


def extract_lstm_states(model, X_sample):
    """
    Extract LSTM internal states (cell and hidden states) for a sample input.
    
    Parameters:
    -----------
    model : Model or object with predict method
        LSTM model to analyze.
    X_sample : numpy.ndarray
        Sample input data with shape (n_samples, seq_length, n_features).
        
    Returns:
    --------
    lstm_states : dict
        Dictionary containing LSTM states, layer outputs, and gate activations.
    """
    # Get LSTM layer model that returns states
    if hasattr(model, 'model'):
        lstm_layer_model = get_lstm_layer_model(model.model, return_sequences=True, return_state=True)
    else:
        lstm_layer_model = get_lstm_layer_model(model, return_sequences=True, return_state=True)
    
    # Predict with the LSTM layer model
    states = lstm_layer_model.predict(X_sample)
    
    # Extract output sequences, cell state, and hidden state
    output_sequences = states[0]  # Shape: (n_samples, seq_length, lstm_units)
    cell_state = states[1]        # Shape: (n_samples, lstm_units)
    hidden_state = states[2]      # Shape: (n_samples, lstm_units)
    
    return {
        'output_sequences': output_sequences,
        'cell_state': cell_state,
        'hidden_state': hidden_state
    }


def analyze_lstm_gates(model, X_sample):
    """
    Analyze LSTM gate activations for a sample input.
    
    Parameters:
    -----------
    model : Model or object with predict method
        LSTM model to analyze.
    X_sample : numpy.ndarray
        Sample input data with shape (n_samples, seq_length, n_features).
        
    Returns:
    --------
    gate_activations : dict
        Dictionary containing LSTM gate activations.
    """
    # This requires creating a custom model with access to the internal gates
    # Note: TensorFlow does not directly expose gate activations in Keras LSTM layers
    # We need to reimplement the LSTM cell to extract gate values
    
    # Find the LSTM layer and get its weights
    if hasattr(model, 'model'):
        keras_model = model.model
    else:
        keras_model = model
    
    lstm_layer = None
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            lstm_layer = layer
            break
    
    if lstm_layer is None:
        raise ValueError("No LSTM layer found in the model")
    
    # Get LSTM weights
    weights = lstm_layer.get_weights()
    kernel = weights[0]  # Input weights
    recurrent_kernel = weights[1]  # Recurrent weights
    bias = weights[2]  # Bias
    
    # LSTM has 4 gates: input, forget, cell, output
    # Each gate has weights for input, recurrent, and bias
    units = lstm_layer.units
    input_dim = lstm_layer.input_shape[-1]
    
    # Split weights for each gate
    # The order is: i (input), f (forget), c (cell), o (output)
    kernel_i = kernel[:, :units]
    kernel_f = kernel[:, units:units*2]
    kernel_c = kernel[:, units*2:units*3]
    kernel_o = kernel[:, units*3:]
    
    recurrent_kernel_i = recurrent_kernel[:, :units]
    recurrent_kernel_f = recurrent_kernel[:, units:units*2]
    recurrent_kernel_c = recurrent_kernel[:, units*2:units*3]
    recurrent_kernel_o = recurrent_kernel[:, units*3:]
    
    bias_i = bias[:units]
    bias_f = bias[units:units*2]
    bias_c = bias[units*2:units*3]
    bias_o = bias[units*3:]
    
    # Create a custom LSTM cell to extract gate values
    class LSTMWithGates(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(LSTMWithGates, self).__init__(**kwargs)
            self.units = units
            self.state_size = [units, units]  # [h, c]
            
            # Set weights directly
            self.kernel_i = tf.Variable(kernel_i, trainable=False)
            self.kernel_f = tf.Variable(kernel_f, trainable=False)
            self.kernel_c = tf.Variable(kernel_c, trainable=False)
            self.kernel_o = tf.Variable(kernel_o, trainable=False)
            
            self.recurrent_kernel_i = tf.Variable(recurrent_kernel_i, trainable=False)
            self.recurrent_kernel_f = tf.Variable(recurrent_kernel_f, trainable=False)
            self.recurrent_kernel_c = tf.Variable(recurrent_kernel_c, trainable=False)
            self.recurrent_kernel_o = tf.Variable(recurrent_kernel_o, trainable=False)
            
            self.bias_i = tf.Variable(bias_i, trainable=False)
            self.bias_f = tf.Variable(bias_f, trainable=False)
            self.bias_c = tf.Variable(bias_c, trainable=False)
            self.bias_o = tf.Variable(bias_o, trainable=False)
        
        def call(self, inputs, states, training=None):
            h_tm1 = states[0]  # Previous hidden state
            c_tm1 = states[1]  # Previous cell state
            
            # Calculate gate values
            x_i = tf.matmul(inputs, self.kernel_i) + tf.matmul(h_tm1, self.recurrent_kernel_i) + self.bias_i
            x_f = tf.matmul(inputs, self.kernel_f) + tf.matmul(h_tm1, self.recurrent_kernel_f) + self.bias_f
            x_c = tf.matmul(inputs, self.kernel_c) + tf.matmul(h_tm1, self.recurrent_kernel_c) + self.bias_c
            x_o = tf.matmul(inputs, self.kernel_o) + tf.matmul(h_tm1, self.recurrent_kernel_o) + self.bias_o
            
            # Apply gate activations
            i = tf.sigmoid(x_i)  # Input gate
            f = tf.sigmoid(x_f)  # Forget gate
            c = f * c_tm1 + i * tf.tanh(x_c)  # New cell state
            o = tf.sigmoid(x_o)  # Output gate
            h = o * tf.tanh(c)  # New hidden state
            
            return h, [h, c, i, f, o, tf.tanh(x_c)]
    
    # Create RNN with our custom cell
    custom_rnn = tf.keras.layers.RNN(
        LSTMWithGates(),
        return_sequences=True,
        return_state=True
    )
    
    # Create a model with our custom RNN
    inputs = tf.keras.Input(shape=X_sample.shape[1:])
    outputs = custom_rnn(inputs)
    custom_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Run prediction to get gate values
    states = custom_model.predict(X_sample)
    
    # Extract gate activations
    output_sequences = states[0]  # Shape: (n_samples, seq_length, lstm_units)
    final_h = states[1]  # Final hidden state
    final_c = states[2]  # Final cell state
    final_i = states[3]  # Final input gate activations
    final_f = states[4]  # Final forget gate activations
    final_o = states[5]  # Final output gate activations
    final_c_tilde = states[6]  # Final candidate cell state
    
    return {
        'output_sequences': output_sequences,
        'final_hidden_state': final_h,
        'final_cell_state': final_c,
        'input_gate': final_i,
        'forget_gate': final_f,
        'output_gate': final_o,
        'candidate_cell': final_c_tilde
    }


def analyze_lstm_feature_importance(model, X_sample, perturbation_std=0.1, n_perturbations=10):
    """
    Analyze feature importance in LSTM model using perturbation analysis.
    
    Parameters:
    -----------
    model : Model or object with predict method
        LSTM model to analyze.
    X_sample : numpy.ndarray
        Sample input data with shape (n_samples, seq_length, n_features).
    perturbation_std : float, optional (default=0.1)
        Standard deviation of Gaussian noise for perturbation.
    n_perturbations : int, optional (default=10)
        Number of perturbations per feature.
        
    Returns:
    --------
    feature_importance : numpy.ndarray
        Array of feature importance scores.
    """
    # Generate baseline prediction
    if hasattr(model, 'model'):
        baseline_pred = model.model.predict(X_sample)
    else:
        baseline_pred = model.predict(X_sample)
    
    # Get input shape
    n_samples, seq_length, n_features = X_sample.shape
    
    # Initialize feature importance scores
    feature_importance = np.zeros((seq_length, n_features))
    
    # Analyze each feature in each time step
    for t in range(seq_length):
        for f in range(n_features):
            # Initialize perturbation effects
            perturbation_effects = []
            
            # Perform multiple perturbations
            for _ in range(n_perturbations):
                # Create perturbed input
                X_perturbed = X_sample.copy()
                
                # Add Gaussian noise to the feature at time step t
                noise = np.random.normal(0, perturbation_std, size=n_samples)
                X_perturbed[:, t, f] += noise
                
                # Generate prediction with perturbed input
                if hasattr(model, 'model'):
                    perturbed_pred = model.model.predict(X_perturbed)
                else:
                    perturbed_pred = model.predict(X_perturbed)
                
                # Calculate effect of perturbation (mean absolute difference)
                effect = np.mean(np.abs(perturbed_pred - baseline_pred))
                perturbation_effects.append(effect)
            
            # Average effect across perturbations
            feature_importance[t, f] = np.mean(perturbation_effects)
    
    # Normalize feature importance
    feature_importance = feature_importance / np.max(feature_importance)
    
    return feature_importance


def analyze_lstm_memory(model, X_test, y_test, max_shifts=10):
    """
    Analyze LSTM memory capabilities by testing with shifted inputs.
    
    Parameters:
    -----------
    model : Model or object with predict method
        LSTM model to analyze.
    X_test : numpy.ndarray
        Test input data with shape (n_samples, seq_length, n_features).
    y_test : numpy.ndarray
        Test target data.
    max_shifts : int, optional (default=10)
        Maximum number of time steps to shift the input.
        
    Returns:
    --------
    memory_analysis : dict
        Dictionary containing memory analysis results.
    """
    from sklearn.metrics import mean_squared_error
    
    # Initialize results
    shift_errors = []
    
    # Evaluate model on different shifts
    for shift in range(max_shifts + 1):
        # Shift the input
        if shift == 0:
            # No shift, use original data
            X_shifted = X_test
        else:
            # Create shifted input
            X_shifted = np.zeros_like(X_test)
            
            # Shift the sequence by 'shift' time steps
            X_shifted[:, shift:, :] = X_test[:, :-shift, :]
            
            # Zero-pad the initial time steps
            X_shifted[:, :shift, :] = 0
        
        # Generate predictions
        if hasattr(model, 'model'):
            y_pred = model.model.predict(X_shifted)
        else:
            y_pred = model.predict(X_shifted)
        
        # Calculate error
        mse = mean_squared_error(y_test, y_pred)
        shift_errors.append(mse)
    
    # Calculate memory capacity
    # Memory capacity is defined as the shift at which error exceeds 2x the baseline (no shift) error
    baseline_error = shift_errors[0]
    memory_capacity = max_shifts
    
    for shift, error in enumerate(shift_errors):
        if shift > 0 and error > 2 * baseline_error:
            memory_capacity = shift - 1
            break
    
    return {
        'shift_errors': shift_errors,
        'memory_capacity': memory_capacity
    } 