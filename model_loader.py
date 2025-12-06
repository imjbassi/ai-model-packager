import os
import torch
import tensorflow as tf


def load_model(path):
    """
    Load an AI model from the given path.
    
    Supports PyTorch (.pth, .pt) and TensorFlow (.h5, .keras) model formats.
    PyTorch models are loaded in evaluation mode on CPU by default.
    
    Args:
        path (str): Path to the model file.
        
    Returns:
        The loaded model object (torch.nn.Module or tf.keras.Model).
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model format is not supported.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Normalize path for consistent extension checking
    normalized_path = path.lower()
    
    # Determine file extension
    _, ext = os.path.splitext(normalized_path)
    
    # PyTorch formats
    if ext in (".pth", ".pt"):
        # Load PyTorch model on CPU and set to evaluation mode
        # Using weights_only=False for backward compatibility, but consider
        # setting to True for security if loading trusted models only
        model = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        if hasattr(model, "eval"):
            model.eval()
        return model
    
    # TensorFlow/Keras formats
    elif ext in (".h5", ".keras"):
        # Load TensorFlow/Keras model
        model = tf.keras.models.load_model(path)
        return model
    
    else:
        raise ValueError(
            f"Unsupported model format: {path}. "
            "Supported formats are .pth/.pt (PyTorch) and .h5/.keras (TensorFlow/Keras)."
        )