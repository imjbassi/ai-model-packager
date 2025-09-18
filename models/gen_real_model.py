#!/usr/bin/env python3
"""
Let's grab a real ResNet-18 model for our demo
"""
import torch
import torchvision.models as models
import os

def generate_resnet18_model():
    """Downloads and saves a pre-trained ResNet-18 model (the real deal!)"""
    print("Generating pre-trained ResNet-18 model...")
    print("This will download the model weights from PyTorch Hub...")
    
    # Grab the pre-trained ResNet-18 (it knows how to recognize 1000 different things!)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    
    # Save it to a file
    output_path = "../resnet18_full.pth"
    torch.save(model, output_path)
    
    # Let's see how big this thing is
    file_size = os.path.getsize(output_path)
    size_mb = file_size / (1024 * 1024)
    
    print(f"Model saved successfully!")
    print(f"File: {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Architecture: ResNet-18 (18 layers deep)")
    print(f"Parameters: ~11.7 million (that's a lot of numbers!)")
    print(f"Input: 224x224 RGB images")
    print(f"Output: 1000 ImageNet classes (cats, dogs, cars, you name it)")
    
    # Optional TorchScript export
    try:
        import torch, torchvision.models as models
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
        example = torch.randn(1,3,224,224)
        scripted = torch.jit.trace(m, example)
        scripted.save("resnet18_scripted.pt")
        print("Also saved TorchScript model: resnet18_scripted.pt")
    except Exception as e:
        print(f"Skipping TorchScript export: {e}")
    
    return output_path

if __name__ == "__main__":
    generate_resnet18_model()
