"""
Utility functions for indexing
"""

import numpy as np
from typing import List, Dict


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length
    
    Args:
        vectors: NumPy array of vectors
    
    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    return vectors / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine similarity score
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple vectors
    
    Args:
        query: Query vector (1D)
        vectors: Matrix of vectors (2D)
    
    Returns:
        Array of similarity scores
    """
    query_norm = query / np.linalg.norm(query)
    vectors_norm = normalize_vectors(vectors)
    return np.dot(vectors_norm, query_norm)


def merge_duplicate_chunks(chunks: List[Dict], threshold: float = 0.95) -> List[Dict]:
    """
    Merge near-duplicate chunks based on text similarity
    
    Args:
        chunks: List of chunk dictionaries
        threshold: Similarity threshold for considering duplicates
    
    Returns:
        Deduplicated list of chunks
    """
    # Simple implementation - can be improved with MinHash/LSH
    seen = set()
    unique_chunks = []
    
    for chunk in chunks:
        text = chunk['text']
        
        # Simple hash-based deduplication
        text_hash = hash(text)
        
        if text_hash not in seen:
            seen.add(text_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks


