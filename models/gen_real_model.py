#!/usr/bin/env python3
"""
Generate a real ResNet-18 model for demonstration purposes.

This script downloads a pre-trained ResNet-18 model from PyTorch Hub
and saves it to disk for use in packaging demonstrations.
"""
import os
import torch
import torchvision.models as models


def generate_resnet18_model():
    """
    Download and save a pre-trained ResNet-18 model.
    
    Downloads the ResNet-18 model with ImageNet pre-trained weights from
    PyTorch Hub and saves it to disk. The model is set to evaluation mode
    before saving.
    
    Returns:
        str: Path to the saved model file.
        
    Raises:
        RuntimeError: If model download or save operation fails.
    """
    print("Generating pre-trained ResNet-18 model...")
    print("Downloading model weights from PyTorch Hub...")
    
    # Download pre-trained ResNet-18 model (1000 ImageNet classes)
    # Note: 'pretrained' parameter is deprecated in newer PyTorch versions
    # but maintained here for backwards compatibility
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Determine output path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "resnet18_full.pth")
    output_path = os.path.normpath(output_path)
    
    # Save the complete model (architecture + weights)
    torch.save(model, output_path)
    
    # Calculate and display file size
    file_size = os.path.getsize(output_path)
    size_mb = file_size / (1024 * 1024)
    
    print(f"Model saved successfully!")
    print(f"File: {output_path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Architecture: ResNet-18 (18 layers)")
    print(f"Parameters: ~11.7 million")
    print(f"Input: 224x224 RGB images")
    print(f"Output: 1000 ImageNet classes")
    
    return output_path


if __name__ == "__main__":
    generate_resnet18_model()