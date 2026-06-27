import pickle
import numpy as np
import faiss
from pathlib import Path

def build_index():
    """Build FAISS index for fast similarity search"""
    
    print("📂 Loading embeddings...")
    
    # Load embeddings
    try:
        with open("data/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        
        with open("data/image_list.pkl", "rb") as f:
            image_list = pickle.load(f)
    except FileNotFoundError:
        print("❌ Embeddings not found. Run generate_embeddings.py first!")
        return
    
    print(f"   ✓ Loaded {len(embeddings)} embeddings")
    
    # Convert embeddings to numpy array (FAISS format)
    print("\n🔄 Preparing embeddings for indexing...")
    embedding_dim = list(embeddings.values())[0].shape[0]
    embeddings_array = np.zeros((len(image_list), embedding_dim), dtype=np.float32)
    
    for idx, image_name in enumerate(image_list):
        embeddings_array[idx] = embeddings[image_name]
    
    print(f"   ✓ Embeddings shape: {embeddings_array.shape}")
    
    # Create FAISS index
    print("\n🏗️  Building FAISS index...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_array)
    
    print(f"   ✓ Index size: {index.ntotal} vectors")
    
    # Save index
    print("\n💾 Saving index...")
    faiss.write_index(index, "data/faiss_index.bin")
    
    print(f"✅ Index built and saved!")
    print(f"📊 Index details:")
    print(f"   - Vectors: {index.ntotal}")
    print(f"   - Dimension: {embedding_dim}")
    print(f"   - File: data/faiss_index.bin")

if __name__ == "__main__":
    build_index()