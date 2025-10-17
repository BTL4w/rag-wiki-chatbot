"""
Build and manage vector database index for embeddings
Supports FAISS, ChromaDB, and other vector databases
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import yaml
from tqdm import tqdm


class VectorIndexer:
    """Build and manage vector database index"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.provider = self.config['vector_db']['provider']
        self.index_path = Path(self.config['vector_db']['index_path'])
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.dimension = self.config['embedding']['dimension']
        self.metric = self.config['vector_db']['metric']
        
        self.index = None
        self.chunks_metadata = []
    
    def _create_faiss_index(self):
        """Create FAISS index"""
        import faiss
        
        index_type = self.config['vector_db']['index_type']
        
        if index_type == "Flat":
            # Simple flat index (exact search)
            if self.metric == "cosine":
                index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
            elif self.metric == "l2":
                index = faiss.IndexFlatL2(self.dimension)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
        
        elif index_type == "HNSW":
            # HNSW index (fast approximate search)
            m = self.config['vector_db'].get('hnsw_m', 32)
            
            if self.metric == "cosine":
                index = faiss.IndexHNSWFlat(self.dimension, m, faiss.METRIC_INNER_PRODUCT)
            elif self.metric == "l2":
                index = faiss.IndexHNSWFlat(self.dimension, m, faiss.METRIC_L2)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            ef_construction = self.config['vector_db'].get('hnsw_ef_construction', 200)
            index.hnsw.efConstruction = ef_construction
        
        elif index_type == "IVF":
            # IVF index (inverted file index)
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            
            if self.metric == "cosine":
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return index
    
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Build vector index from embeddings
        
        Args:
            embeddings: NumPy array of embeddings (n_vectors, dimension)
            chunks: List of chunk metadata dictionaries
        """
        print(f"Building {self.provider} index with {len(embeddings)} vectors...")
        
        # Handle empty embeddings
        if len(embeddings) == 0:
            print("⚠️  Warning: No embeddings to index!")
            print("   Please ensure you have valid parsed documents and chunks.")
            return
        
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        self.chunks_metadata = chunks
        
        if self.provider == "faiss":
            self._build_faiss_index(embeddings)
        elif self.provider == "chromadb":
            self._build_chromadb_index(embeddings, chunks)
        else:
            raise ValueError(f"Unknown vector DB provider: {self.provider}")
        
        print("Index built successfully!")
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index"""
        import faiss
        
        self.index = self._create_faiss_index()
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        # For IVF, need to train first
        if isinstance(self.index, faiss.IndexIVF):
            print("Training IVF index...")
            self.index.train(embeddings)
        
        # Add vectors
        print("Adding vectors to index...")
        self.index.add(embeddings)
        
        print(f"Index contains {self.index.ntotal} vectors")
    
    def _build_chromadb_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """Build ChromaDB index"""
        import chromadb
        from chromadb.config import Settings
        
        # Create ChromaDB client
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.index_path)
        ))
        
        # Create or get collection
        collection_name = "wiki_chunks"
        try:
            collection = client.get_collection(collection_name)
            client.delete_collection(collection_name)
        except:
            pass
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine" if self.metric == "cosine" else "l2"}
        )
        
        # Add to collection in batches
        batch_size = 1000
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Adding to ChromaDB"):
            batch_end = min(i + batch_size, len(embeddings))
            
            collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=[c['text'] for c in chunks[i:batch_end]],
                metadatas=[{
                    'chunk_id': c['global_chunk_id'],
                    'title': c['title'],
                    'section': c['section'],
                    'page_id': c['page_id']
                } for c in chunks[i:batch_end]],
                ids=[f"chunk_{c['global_chunk_id']}" for c in chunks[i:batch_end]]
            )
        
        self.index = collection
    
    def save_index(self):
        """Save index and metadata to disk"""
        print(f"Saving index to {self.index_path}...")
        
        if self.provider == "faiss":
            import faiss
            
            index_file = self.index_path / "index.faiss"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata separately
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            
            # Save config
            config_file = self.index_path / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'provider': self.provider,
                    'dimension': self.dimension,
                    'metric': self.metric,
                    'index_type': self.config['vector_db']['index_type'],
                    'num_vectors': self.index.ntotal
                }, f, indent=2)
        
        elif self.provider == "chromadb":
            # ChromaDB auto-persists
            pass
        
        print("Index saved successfully!")
    
    def load_index(self):
        """Load index and metadata from disk"""
        print(f"Loading index from {self.index_path}...")
        
        if self.provider == "faiss":
            import faiss
            
            index_file = self.index_path / "index.faiss"
            if not index_file.exists():
                raise FileNotFoundError(f"Index file not found: {index_file}")
            
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            
            # Set search parameters
            if isinstance(self.index, faiss.IndexHNSW):
                ef_search = self.config['vector_db'].get('hnsw_ef_search', 100)
                self.index.hnsw.efSearch = ef_search
        
        elif self.provider == "chromadb":
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.index_path)
            ))
            
            self.index = client.get_collection("wiki_chunks")
        
        print(f"Index loaded successfully!")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            Tuple of (indices, distances/scores)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        if self.provider == "faiss":
            import faiss
            
            # Ensure float32 and 2D
            query_vector = query_vector.astype('float32').reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(query_vector, top_k)
            
            return indices[0].tolist(), distances[0].tolist()
        
        elif self.provider == "chromadb":
            results = self.index.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k
            )
            
            # Extract chunk IDs and distances
            ids = results['ids'][0]
            distances = results['distances'][0]
            
            # Convert IDs to indices
            indices = [int(id.split('_')[1]) for id in ids]
            
            return indices, distances
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def get_chunks(self, indices: List[int]) -> List[Dict]:
        """
        Get chunk metadata by indices
        
        Args:
            indices: List of chunk indices
        
        Returns:
            List of chunk dictionaries
        """
        return [self.chunks_metadata[i] for i in indices if i < len(self.chunks_metadata)]


def main():
    """Build index from embeddings and chunks"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build vector index')
    parser.add_argument('--config', default='src/config.yaml', help='Path to config file')
    parser.add_argument('--embeddings', default='data/chunks/embeddings.npy', help='Embeddings file')
    parser.add_argument('--chunks', default='data/chunks/chunks.jsonl', help='Chunks file')
    args = parser.parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    
    # Load chunks
    print(f"Loading chunks from {args.chunks}...")
    chunks = []
    with open(args.chunks, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # Build index
    indexer = VectorIndexer(args.config)
    indexer.build_index(embeddings, chunks)
    indexer.save_index()
    
    print("Done!")


if __name__ == "__main__":
    main()


