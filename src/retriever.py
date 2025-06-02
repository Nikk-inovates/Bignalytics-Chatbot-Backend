import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_chunks(query, model, vectors, chunks, top_k=3):
    """
    Returns the top_k most relevant text chunks to the input query based on cosine similarity.

    Args:
        query (str): User's input query.
        model: SentenceTransformer model with .encode().
        vectors (array-like): Precomputed vector embeddings of text chunks.
        chunks (list): List of original text chunks.
        top_k (int): Number of most similar chunks to return.

    Returns:
        list: Top-k most relevant chunks.
    """
    if not query or not query.strip():
        raise ValueError("❌ Query is empty. Please enter a valid question.")

    if len(vectors) != len(chunks):
        raise ValueError("❌ Vectors and chunks count mismatch.")

    try:
        query_vec = model.encode([query])
        scores = cosine_similarity(query_vec, vectors).flatten()

        if np.isnan(scores).any():
            raise ValueError("❌ NaN found in similarity scores.")

        top_indices = scores.argsort()[::-1][:top_k]
        return [chunks[i] for i in top_indices]

    except Exception as e:
        raise Exception(f"❌ Failed to retrieve top chunks: {e}")