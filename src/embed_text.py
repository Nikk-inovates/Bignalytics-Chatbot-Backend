import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def split_text(text, chunk_size=500, overlap=100):
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap

    if not chunks:
        raise ValueError("Text splitting failed. No chunks created.")
    
    return chunks

def embed_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    if not chunks or not isinstance(chunks, list):
        raise ValueError("Chunks must be a non-empty list of strings.")

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        return model, index, embeddings, chunks
    
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

def save_faiss_index(index, chunks, path="embeddings/faiss_index.faiss"):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)

        chunks_path = os.path.splitext(path)[0] + "_chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        print("✅ FAISS index and chunks saved successfully.")
    
    except Exception as e:
        raise IOError(f"Error saving FAISS index or chunks: {e}")

def load_faiss_index(path="embeddings/faiss_index.faiss"):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index file not found at: {path}")
        
        index = faiss.read_index(path)

        chunks_path = os.path.splitext(path)[0] + "_chunks.pkl"
        if not os.path.exists(chunks_path):
            raise FileNotFoundError("Chunks file not found.")

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        if not chunks:
            raise ValueError("Loaded chunks are empty.")

        return index, chunks

    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index or chunks: {e}")