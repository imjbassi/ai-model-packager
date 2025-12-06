import os
import torch
import tensorflow as tf


def load_model(path):
    """
    Load an AI model from the given path.
    
    Supports PyTorch (.pth) and TensorFlow (.h5) model formats.
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
    
    if normalized_path.endswith(".pth"):
        # Load PyTorch model on CPU and set to evaluation mode
        model = torch.load(path, map_location=torch.device("cpu"))
        if hasattr(model, 'eval'):
            model.eval()
        return model
    elif normalized_path.endswith(".h5"):
        # Load TensorFlow/Keras model
        model = tf.keras.models.load_model(path)
        return model
    else:
        raise ValueError(
            f"Unsupported model format: {path}. "
            "Supported formats are .pth (PyTorch) and .h5 (TensorFlow/Keras)."
        )