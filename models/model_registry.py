"""
Registry of available LSTM model architectures.
"""

import importlib
import os


# Dictionary of registered model architectures
_MODEL_REGISTRY = {}


def register_model(name, model_class):
    """
    Register a model architecture.
    
    Parameters:
    -----------
    name : str
        Name of the model architecture.
    model_class : class
        Model class.
    """
    _MODEL_REGISTRY[name] = model_class


def get_model_class(name):
    """
    Get a model class by name.
    
    Parameters:
    -----------
    name : str
        Name of the model architecture.
        
    Returns:
    --------
    model_class : class
        Model class, or None if not found.
    """
    # Ensure model classes are imported
    _ensure_models_imported()
    
    return _MODEL_REGISTRY.get(name)


def get_available_models():
    """
    Get a list of available model architectures.
    
    Returns:
    --------
    model_names : list
        List of available model architecture names.
    """
    # Ensure model classes are imported
    _ensure_models_imported()
    
    return list(_MODEL_REGISTRY.keys())


def _ensure_models_imported():
    """
    Ensure all model classes are imported and registered.
    """
    # Import all model modules in architectures directory
    arch_dir = os.path.join(os.path.dirname(__file__), 'architectures')
    
    # If directory doesn't exist yet, return early
    if not os.path.exists(arch_dir):
        return
    
    for filename in os.listdir(arch_dir):
        # Skip non-Python files and __init__.py
        if not filename.endswith('.py') or filename == '__init__.py':
            continue
        
        # Import the module
        module_name = filename[:-3]  # Remove .py extension
        try:
            importlib.import_module(f'.architectures.{module_name}', package='models')
        except ImportError:
            # Log the error but continue
            print(f"Error importing model architecture: {module_name}")


# Initialize the registry by importing all model architectures
_ensure_models_imported() 