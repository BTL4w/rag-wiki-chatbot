"""
Document retriever using vector search
Supports dense retrieval and hybrid search (BM25 + dense)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import yaml
from pathlib import Path


class Retriever:
    """Retrieve relevant documents using vector search"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.top_k = self.config['retrieval']['top_k']
        self.use_hybrid = self.config['retrieval'].get('hybrid_search', False)
        
        if self.use_hybrid:
            self.bm25_weight = self.config['retrieval'].get('bm25_weight', 0.3)
            self.dense_weight = self.config['retrieval'].get('dense_weight', 0.7)
        
        self.indexer = None
        self.embedder = None
        self.bm25_index = None
    
    def load_components(self):
        """Load embedder and indexer"""
        from ..indexer import Embedder, VectorIndexer
        
        print("Loading embedder...")
        self.embedder = Embedder()
        
        print("Loading vector index...")
        self.indexer = VectorIndexer()
        self.indexer.load_index()
        
        if self.use_hybrid:
            print("Loading BM25 index...")
            self._load_bm25_index()
    
    def _load_bm25_index(self):
        """Load or build BM25 index"""
        from rank_bm25 import BM25Okapi
        
        # Get all chunks
        chunks = self.indexer.chunks_metadata
        
        # Tokenize documents
        tokenized_corpus = [doc['text'].lower().split() for doc in chunks]
        
        # Create BM25 index
        self.bm25_index = BM25Okapi(tokenized_corpus)
    
    def dense_search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Dense vector search
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of retrieved chunks with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        # Embed query
        query_vector = self.embedder.embed_query(query)
        
        # Search index
        indices, scores = self.indexer.search(query_vector, top_k)
        
        # Get chunks
        chunks = self.indexer.get_chunks(indices)
        
        # Attach scores
        results = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy['score'] = float(scores[i])
            chunk_copy['score_type'] = 'dense'
            results.append(chunk_copy)
        
        return results
    
    def bm25_search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        BM25 sparse search
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of retrieved chunks with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded")
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Get chunks
        chunks = self.indexer.get_chunks(top_indices.tolist())
        
        # Attach scores
        results = []
        for i, chunk in enumerate(chunks):
            chunk_copy = chunk.copy()
            chunk_copy['score'] = float(scores[top_indices[i]])
            chunk_copy['score_type'] = 'bm25'
            results.append(chunk_copy)
        
        return results
    
    def hybrid_search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Hybrid search combining BM25 and dense retrieval
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of retrieved chunks with combined scores
        """
        if top_k is None:
            top_k = self.top_k
        
        # Get results from both methods
        dense_results = self.dense_search(query, top_k * 2)
        bm25_results = self.bm25_search(query, top_k * 2)
        
        # Normalize scores
        dense_scores = [r['score'] for r in dense_results]
        bm25_scores = [r['score'] for r in bm25_results]
        
        dense_max = max(dense_scores) if dense_scores else 1.0
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        
        # Combine results by chunk ID
        combined = {}
        
        for result in dense_results:
            chunk_id = result['global_chunk_id']
            normalized_score = result['score'] / dense_max if dense_max > 0 else 0
            combined[chunk_id] = {
                'chunk': result,
                'dense_score': normalized_score,
                'bm25_score': 0.0
            }
        
        for result in bm25_results:
            chunk_id = result['global_chunk_id']
            normalized_score = result['score'] / bm25_max if bm25_max > 0 else 0
            
            if chunk_id in combined:
                combined[chunk_id]['bm25_score'] = normalized_score
            else:
                combined[chunk_id] = {
                    'chunk': result,
                    'dense_score': 0.0,
                    'bm25_score': normalized_score
                }
        
        # Calculate combined scores
        results = []
        for chunk_id, data in combined.items():
            chunk = data['chunk'].copy()
            chunk['score'] = (
                self.dense_weight * data['dense_score'] +
                self.bm25_weight * data['bm25_score']
            )
            chunk['score_type'] = 'hybrid'
            chunk['dense_score'] = data['dense_score']
            chunk['bm25_score'] = data['bm25_score']
            results.append(chunk)
        
        # Sort by combined score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant documents
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of retrieved chunks
        """
        if self.indexer is None or self.embedder is None:
            raise ValueError("Components not loaded. Call load_components() first.")
        
        if self.use_hybrid:
            return self.hybrid_search(query, top_k)
        else:
            return self.dense_search(query, top_k)


def main():
    """Test retriever"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test retriever')
    parser.add_argument('--config', default='src/config.yaml', help='Path to config file')
    parser.add_argument('--query', required=True, help='Query text')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    args = parser.parse_args()
    
    retriever = Retriever(args.config)
    retriever.load_components()
    
    results = retriever.retrieve(args.query, args.top_k)
    
    print(f"\nQuery: {args.query}")
    print(f"Found {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['title']}] - {result['section']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Text: {result['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()


