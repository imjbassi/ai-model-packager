#!/usr/bin/env python3
"""
Generate a real ResNet-18 model for demonstration purposes.

This script downloads a pre-trained ResNet-18 model from PyTorch Hub
and saves it to disk for use in packaging demonstrations.
"""
import os
import sys
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
    try:
        print("Generating pre-trained ResNet-18 model...")
        print("Downloading model weights from PyTorch Hub...")
        
        # Download pre-trained ResNet-18 model (1000 ImageNet classes)
        # Use weights parameter for newer PyTorch versions, fall back to pretrained for older versions
        try:
            # PyTorch >= 0.13 uses weights parameter
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # Fall back to deprecated pretrained parameter for older PyTorch versions
            model = models.resnet18(pretrained=True)
        
        # Set model to evaluation mode
        model.eval()
        
        # Determine output path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "..", "resnet18_full.pth")
        output_path = os.path.normpath(output_path)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the complete model (architecture + weights)
        torch.save(model, output_path)
        
        # Verify the file was created successfully
        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to save model to {output_path}")
        
        # Calculate and display file size
        file_size = os.path.getsize(output_path)
        size_mb = file_size / (1024 * 1024)
        
        print("Model saved successfully!")
        print(f"File: {output_path}")
        print(f"Size: {size_mb:.1f} MB")
        print("Architecture: ResNet-18 (18 layers)")
        print("Parameters: ~11.7 million")
        print("Input: 224x224 RGB images")
        print("Output: 1000 ImageNet classes")
        
        return output_path
        
    except Exception as e:
        print(f"Error generating model: {e}", file=sys.stderr)
        raise RuntimeError(f"Failed to generate ResNet-18 model: {e}") from e


def main():
    """Main entry point for the script."""
    try:
        generate_resnet18_model()
        return 0
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())