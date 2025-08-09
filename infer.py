import argparse
import sys
import os

def run_pytorch_inference(model_path: str, input_image: str = None):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    
    model = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    
    if input_image and os.path.exists(input_image):
        # Let's process a real image
        print(f"Processing image: {input_image}")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(input_image).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
    else:
        # No image? No problem - we'll make one up for testing
        print("Using dummy input tensor")
        input_tensor = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Let's see what the model thinks (top 5 guesses)
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        print("Top 5 predictions:")
        for i in range(5):
            idx = top5_indices[0][i].item()
            prob = top5_prob[0][i].item()
            print(f"   {i+1}. Class {idx}: {prob:.4f} ({prob*100:.1f}%)")
    
    return outputs

def run_tensorflow_inference(model_path: str):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    import numpy as np
    dummy = np.random.randn(1, 224, 224, 3).astype(np.float32)
    out = model(dummy)
    print("TensorFlow inference output:", out.numpy())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--test-input", help="Path to input image for testing")
    args = parser.parse_args()
    
    path = args.model
    if path.endswith(".pth"):
        run_pytorch_inference(path, args.test_input)
    elif path.endswith(".h5"):
        run_tensorflow_inference(path)
    else:
        sys.exit("Unsupported model format")

if __name__ == "__main__":
    main()