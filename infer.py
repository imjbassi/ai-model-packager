```python
import argparse
import os
import sys


def run_pytorch_inference(model_path: str, input_image: str = None):
    """
    Run inference using a PyTorch model.
    
    Args:
        model_path: Path to the PyTorch model file (.pth)
        input_image: Optional path to an input image. If not provided, uses dummy data.
    
    Returns:
        Model output tensor
    """
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    
    # Load model and set to evaluation mode
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
        model.eval()
    except Exception as e:
        sys.exit(f"Error loading PyTorch model: {e}")
    
    # Prepare input tensor
    if input_image and os.path.exists(input_image):
        print(f"Processing image: {input_image}")
        
        # Standard ImageNet preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(input_image).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
        except Exception as e:
            sys.exit(f"Error processing image: {e}")
    else:
        if input_image:
            print(f"Warning: Image file not found: {input_image}")
        print("Using dummy input tensor (no image provided)")
        input_tensor = torch.randn(1, 3, 224, 224)
    
    # Run inference
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Display top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            print("\nTop 5 predictions:")
            for i in range(5):
                idx = top5_indices[0][i].item()
                prob = top5_prob[0][i].item()
                print(f"   {i+1}. Class {idx}: {prob:.4f} ({prob*100:.1f}%)")
    except Exception as e:
        sys.exit(f"Error during inference: {e}")
    
    return outputs


def run_tensorflow_inference(model_path: str, input_image: str = None):
    """
    Run inference using a TensorFlow/Keras model.
    
    Args:
        model_path: Path to the TensorFlow model file (.h5)
        input_image: Optional path to an input image. If provided, uses the image;
                     otherwise uses dummy data.
    """
    import numpy as np
    import tensorflow as tf
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        sys.exit(f"Error loading TensorFlow model: {e}")
    
    # Prepare input
    if input_image and os.path.exists(input_image):
        print(f"Processing image: {input_image}")
        try:
            from PIL import Image
            
            # Load and preprocess image
            image = Image.open(input_image).convert('RGB')
            image = image.resize((224, 224))
            input_data = np.array(image).astype(np.float32)
            
            # Normalize to [0, 1] range
            input_data = input_data / 255.0
            
            # Add batch dimension
            input_data = np.expand_dims(input_data, axis=0)
        except Exception as e:
            sys.exit(f"Error processing image: {e}")
    else:
        if input_image:
            print(f"Warning: Image file not found: {input_image}")
        print("Using dummy input tensor (no image provided)")
        # Create dummy input (batch_size=1, height=224, width=224, channels=3)
        input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)
    
    # Run inference
    try:
        output = model(input_data)
        print("\nTensorFlow inference output shape:", output.shape)
        print("TensorFlow inference output:", output.numpy())
        
        # If output looks like classification probabilities, show top predictions
        if len(output.shape) == 2 and output.shape[0] == 1:
            probabilities = tf.nn.softmax(output, axis=1).numpy()[0]
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            
            print("\nTop 5 predictions:")
            for i, idx in enumerate(top5_indices):
                prob = probabilities[idx]
                print(f"   {i+1}. Class {idx}: {prob:.4f} ({prob*100:.1f}%)")
    except Exception as e:
        sys.exit(f"Error during inference: {e}")


def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(
        description="Run inference on PyTorch or TensorFlow models"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to model file (.pth for PyTorch, .h5 for TensorFlow)"
    )
    parser.add_argument(
        "--test-input",
        help="Path to input image for testing"
    )
    args = parser.parse_args()
    
    model_path = args.model
    
    # Validate model file exists
    if not os.path.exists(model_path):
        sys.exit(f"Error: Model file not found: {model_path}")
    
    # Route to appropriate inference function based on file extension
    if model_path.endswith(".pth"):
        run_pytorch_inference(model_path, args.test_input)
    elif model_path.endswith(".h5"):
        run_tensorflow_inference(model_path, args.test_input)
    else:
        sys.exit(f"Error: Unsupported model format. Expected .pth or .h5, got: {model_path}")


if __name__ == "__main__":
    main()
```