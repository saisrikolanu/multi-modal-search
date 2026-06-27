import streamlit as st
import torch
import clip
import faiss
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Multi-Modal Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stImage {
        border-radius: 8px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load CLIP model and FAISS index (cached for performance)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

@st.cache_resource
def load_faiss_index():
    """Load pre-built FAISS index and image list"""
    try:
        index = faiss.read_index("data/faiss_index.bin")
        
        with open("data/image_list.pkl", "rb") as f:
            image_list = pickle.load(f)
        
        with open("data/embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        
        return index, image_list, embeddings
    except FileNotFoundError:
        return None, None, None

def search_by_image(image_file, model, preprocess, device, index, image_list, top_k=9):
    """Search for similar images given an uploaded image"""
    
    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Generate embedding
    with torch.no_grad():
        query_embedding = model.encode_image(image_tensor)
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k + 1)  # +1 to exclude exact match
    
    # Get results (skip first result which is the query image itself)
    results = []
    for idx, distance in zip(indices[0][1:], distances[0][1:]):
        if idx < len(image_list):
            results.append({
                "filename": image_list[idx],
                "distance": float(distance),
                "similarity": float(1 / (1 + distance))  # Convert to similarity score
            })
    
    return results, image

def search_by_text(query_text, model, preprocess, device, index, image_list, top_k=9):
    """Search for images given a text query"""
    
    # Encode text
    text_tokens = clip.tokenize(query_text).to(device)
    
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = text_embedding.cpu().numpy().astype(np.float32)
    
    # Search FAISS index
    distances, indices = index.search(text_embedding, top_k)
    
    # Get results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(image_list):
            results.append({
                "filename": image_list[idx],
                "distance": float(distance),
                "similarity": float(1 / (1 + distance))
            })
    
    return results

# Main app
st.title("🔍 Multi-Modal Search Engine")
st.markdown("Search images using text descriptions or similar images")

# Check if data exists
index, image_list, embeddings = load_faiss_index()

if index is None:
    st.error("❌ Data not found! Please run these commands first:")
    st.code("""
    python download_images.py
    python generate_embeddings.py
    python build_index.py
    """)
else:
    # Load models
    model, preprocess, device = load_models()
    
    # Sidebar with info
    with st.sidebar:
        st.header("📊 Database Info")
        st.metric("Total Images", len(image_list))
        st.metric("Embedding Dimension", embeddings[image_list[0]].shape[0])
        st.metric("Device", device.upper())
        
        st.divider()
        
        st.header("ℹ️ How it works")
        st.markdown("""
        1. **Text Search**: Describe what you're looking for
        2. **Image Search**: Upload an image to find similar ones
        3. **Results**: See ranked results with similarity scores
        
        The search uses **CLIP embeddings** and **FAISS** for fast semantic matching.
        """)
    
    # Search tabs
    tab1, tab2 = st.tabs(["🔤 Text Search", "📸 Image Search"])
    
    # TAB 1: Text Search
    with tab1:
        st.subheader("Search by Text Description")
        
        query_text = st.text_input(
            "Describe what you're looking for:",
            placeholder="e.g., 'A sunset over the ocean', 'Cute dogs playing'",
            key="text_search"
        )
        
        if query_text:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                search_button = st.button("🔍 Search", key="text_search_btn")
            
            if search_button:
                with st.spinner("🤖 Searching..."):
                    results = search_by_text(query_text, model, preprocess, device, index, image_list, top_k=9)
                
                st.success(f"Found {len(results)} results!")
                
                # Display results
                cols = st.columns(3)
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        image_path = Path("data/images") / result["filename"]
                        
                        if image_path.exists():
                            image = Image.open(image_path)
                            st.image(
                                image,
                                use_column_width=True,
                                caption=f"Similarity: {result['similarity']:.1%}"
                            )
    
    # TAB 2: Image Search
    with tab2:
        st.subheader("Search by Similar Image")
        
        uploaded_image = st.file_uploader(
            "Upload an image to find similar ones:",
            type=["jpg", "jpeg", "png"],
            key="image_upload"
        )
        
        if uploaded_image is not None:
            with st.spinner("🤖 Searching..."):
                results, query_image = search_by_image(
                    uploaded_image,
                    model,
                    preprocess,
                    device,
                    index,
                    image_list,
                    top_k=9
                )
            
            # Display query image and results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Your Image")
                st.image(query_image, use_column_width=True)
            
            with col2:
                st.subheader("Most Similar")
                if results:
                    best_result = results[0]
                    image_path = Path("data/images") / best_result["filename"]
                    
                    if image_path.exists():
                        image = Image.open(image_path)
                        st.image(
                            image,
                            use_column_width=True,
                            caption=f"Similarity: {best_result['similarity']:.1%}"
                        )
            
            # Show all results
            st.subheader("All Results")
            cols = st.columns(3)
            
            for idx, result in enumerate(results):
                with cols[idx % 3]:
                    image_path = Path("data/images") / result["filename"]
                    
                    if image_path.exists():
                        image = Image.open(image_path)
                        st.image(
                            image,
                            use_column_width=True,
                            caption=f"Similarity: {result['similarity']:.1%}"
                        )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    Built with CLIP, FAISS, and Streamlit | Multi-Modal Search Engine
    </div>
    """, unsafe_allow_html=True)