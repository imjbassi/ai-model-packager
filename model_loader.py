import torch
import tensorflow as tf
import os

def load_model(path):
    """
    Load an AI model from the given path.
    Supports .pth (PyTorch) and .h5 (TensorFlow) formats.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.endswith(".pth"):
        model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        return model
    elif path.endswith(".h5"):
        model = tf.keras.models.load_model(path)
        return model
    else:
        raise ValueError("Unsupported model format, must be .pth or .h5")