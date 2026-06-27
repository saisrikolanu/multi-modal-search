import torch
import clip
from pathlib import Path
import numpy as np
import pickle
from PIL import Image
import os

def generate_embeddings():
    """Generate CLIP embeddings for all images"""
    
    print("🤖 Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"   ✓ Model loaded on {device}")
    
    # Find all images
    images_folder = Path("data/images")
    image_files = sorted(list(images_folder.glob("*.jpg")))
    
    if not image_files:
        print("❌ No images found. Run download_images.py first!")
        return
    
    print(f"\n📸 Found {len(image_files)} images")
    print(f"🔄 Generating embeddings...")
    
    embeddings = {}
    image_list = []
    
    with torch.no_grad():
        for idx, image_path in enumerate(image_files):
            try:
                # Load and preprocess image
                image = Image.open(image_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Generate embedding
                image_embedding = model.encode_image(image_tensor)
                image_embedding = image_embedding.cpu().numpy()
                
                # Store
                filename = image_path.name
                embeddings[filename] = image_embedding[0]
                image_list.append(filename)
                
                if (idx + 1) % 50 == 0:
                    print(f"   ✓ Processed {idx + 1}/{len(image_files)} images...")
                
            except Exception as e:
                print(f"   ⚠️  Failed to process {image_path.name}: {str(e)}")
                continue
    
    # Save embeddings
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)
    
    with open("data/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    with open("data/image_list.pkl", "wb") as f:
        pickle.dump(image_list, f)
    
    print(f"\n✅ Embeddings generated!")
    print(f"📊 Total embeddings: {len(embeddings)}")
    print(f"💾 Saved to: data/embeddings.pkl")

if __name__ == "__main__":
    generate_embeddings()