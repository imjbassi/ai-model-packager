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
        print(f"Error loading PyTorch model: {e}", file=sys.stderr)
        sys.exit(1)
    
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
            print(f"Error processing image: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if input_image:
            print(f"Warning: Image file not found: {input_image}", file=sys.stderr)
        print("Using dummy input tensor (no image provided)")
        input_tensor = torch.randn(1, 3, 224, 224)
    
    # Run inference
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # Check if output is suitable for classification
            if len(outputs.shape) == 2 and outputs.shape[0] == 1:
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Display top 5 predictions
                num_classes = outputs.shape[1]
                top_k = min(5, num_classes)
                top_prob, top_indices = torch.topk(probabilities, top_k)
                
                print(f"\nTop {top_k} predictions:")
                for i in range(top_k):
                    idx = top_indices[0][i].item()
                    prob = top_prob[0][i].item()
                    print(f"   {i+1}. Class {idx}: {prob:.4f} ({prob*100:.1f}%)")
            else:
                print(f"\nPyTorch inference output shape: {outputs.shape}")
                print(f"PyTorch inference output: {outputs}")
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
    
    return outputs


def run_tensorflow_inference(model_path: str, input_image: str = None):
    """
    Run inference using a TensorFlow/Keras model.
    
    Args:
        model_path: Path to the TensorFlow model file (.h5)
        input_image: Optional path to an input image. If not provided, uses dummy data.
    
    Returns:
        Model output tensor
    """
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading TensorFlow model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Prepare input
    if input_image and os.path.exists(input_image):
        print(f"Processing image: {input_image}")
        try:
            # Load and preprocess image
            image = Image.open(input_image).convert('RGB')
            image = image.resize((224, 224))
            input_data = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1] range
            input_data = input_data / 255.0
            
            # Add batch dimension
            input_data = np.expand_dims(input_data, axis=0)
        except Exception as e:
            print(f"Error processing image: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if input_image:
            print(f"Warning: Image file not found: {input_image}", file=sys.stderr)
        print("Using dummy input tensor (no image provided)")
        # Create dummy input (batch_size=1, height=224, width=224, channels=3)
        # Use uniform distribution [0, 1] to match normalized image data
        input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Run inference
    try:
        output = model(input_data, training=False)
        print(f"\nTensorFlow inference output shape: {output.shape}")
        
        # If output looks like classification logits/probabilities, show top predictions
        if len(output.shape) == 2 and output.shape[0] == 1:
            probabilities = tf.nn.softmax(output, axis=1).numpy()[0]
            num_classes = len(probabilities)
            top_k = min(5, num_classes)
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            
            print(f"\nTop {top_k} predictions:")
            for i, idx in enumerate(top_indices):
                prob = probabilities[idx]
                print(f"   {i+1}. Class {idx}: {prob:.4f} ({prob*100:.1f}%)")
        else:
            print(f"TensorFlow inference output: {output.numpy()}")
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
    
    return output


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
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Route to appropriate inference function based on file extension
    if model_path.endswith(".pth"):
        run_pytorch_inference(model_path, args.test_input)
    elif model_path.endswith(".h5"):
        run_tensorflow_inference(model_path, args.test_input)
    else:
        print(f"Error: Unsupported model format. Expected .pth or .h5, got: {model_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```