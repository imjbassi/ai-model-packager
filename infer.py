import argparse, sys
from pathlib import Path

def infer_pytorch(model_path, sample="sample.jpg"):
    import torch, torchvision.transforms as T
    from PIL import Image
    model = torch.jit.load(model_path) if model_path.endswith(".pt") else torch.load(model_path, map_location="cpu")
    model.eval()
    img = Image.open(sample).convert("RGB")
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, 5)
    print("Top 5 predictions:")
    for i, (p, idx) in enumerate(zip(topk.values, topk.indices), 1):
        print(f"  {i}. Class {idx.item()}: {p.item():.4f}")

def infer_tensorflow(model_path, sample="sample.jpg"):
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    model = tf.keras.models.load_model(model_path)
    img = Image.open(sample).convert("RGB").resize((224,224))
    x = np.array(img)/255.0
    x = (x - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    x = x[np.newaxis, ...]
    logits = model.predict(x)
    probs = tf.nn.softmax(logits[0]).numpy()
    top_idx = probs.argsort()[::-1][:5]
    print("Top 5 predictions:")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i}. Class {idx}: {probs[idx]:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", default="sample.jpg")
    args = ap.parse_args()

    ext = Path(args.model).suffix.lower()
    try:
        if ext in (".pth", ".pt", ".bin"):
            infer_pytorch(args.model, args.sample)
        elif ext in (".h5", ".hdf5", ".keras", ".pb"):
            infer_tensorflow(args.model, args.sample)
        else:
            # default try PyTorch first, then TensorFlow
            try:
                infer_pytorch(args.model, args.sample)
            except Exception:
                infer_tensorflow(args.model, args.sample)
    except Exception as e:
        print(f"ERROR: inference failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
