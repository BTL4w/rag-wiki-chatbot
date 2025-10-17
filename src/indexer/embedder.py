"""
Create embeddings for text chunks
Supports multiple embedding providers (Sentence Transformers, OpenAI, etc.)
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import yaml
from tqdm import tqdm

# Disable TensorFlow (we don't need it for embeddings)
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'


class Embedder:
    """Generate embeddings for text chunks"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.provider = self.config['embedding']['provider']
        self.model_name = self.config['embedding']['model_name']
        self.batch_size = self.config['embedding']['batch_size']
        self.normalize = self.config['embedding']['normalize']
        self.dimension = self.config['embedding']['dimension']
        
        # Setup cache
        self.cache_dir = Path(self.config['embedding']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'embedding_cache.pkl'
        self.cache = self._load_cache()
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load embedding model based on provider"""
        if self.provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            print(f"Loading Sentence Transformer model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            return model
            
        elif self.provider == "openai":
            import openai
            api_key = self.config['embedding'].get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in config or environment")
            openai.api_key = api_key
            return None  # OpenAI uses API calls
            
        elif self.provider == "huggingface":
            from transformers import AutoTokenizer, AutoModel
            print(f"Loading HuggingFace model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            return {'tokenizer': tokenizer, 'model': model}
        
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")
    
    def _load_cache(self) -> Dict:
        """Load embedding cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save embedding cache"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def embed_texts_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Embed using Sentence Transformers"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        return np.array(embeddings)
    
    def embed_texts_openai(self, texts: List[str]) -> np.ndarray:
        """Embed using OpenAI API"""
        import openai
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = openai.embeddings.create(
                input=batch,
                model=self.model_name
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        if self.normalize:
            embeddings = np.array([self._normalize_vector(e) for e in embeddings])
        
        return embeddings
    
    def embed_texts_huggingface(self, texts: List[str]) -> np.ndarray:
        """Embed using HuggingFace transformers"""
        import torch
        
        tokenizer = self.model['tokenizer']
        model = self.model['model']
        
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**encoded)
                # Mean pooling
                embeddings_batch = outputs.last_hidden_state.mean(dim=1)
            
            embeddings.append(embeddings_batch.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        
        if self.normalize:
            embeddings = np.array([self._normalize_vector(e) for e in embeddings])
        
        return embeddings
    
    def embed_texts(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Embed a list of texts
        
        Args:
            texts: List of text strings
            use_cache: Whether to use cached embeddings
        
        Returns:
            NumPy array of embeddings (n_texts, dimension)
        """
        if not texts:
            return np.array([])
        
        # Check cache
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            if use_cache and text in self.cache:
                embeddings.append(self.cache[text])
            else:
                embeddings.append(None)
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            if self.provider == "sentence-transformers":
                new_embeddings = self.embed_texts_sentence_transformers(texts_to_embed)
            elif self.provider == "openai":
                new_embeddings = self.embed_texts_openai(texts_to_embed)
            elif self.provider == "huggingface":
                new_embeddings = self.embed_texts_huggingface(texts_to_embed)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            # Update cache and results
            for i, idx in enumerate(indices_to_embed):
                embeddings[idx] = new_embeddings[i]
                self.cache[texts_to_embed[i]] = new_embeddings[i]
        
        return np.array(embeddings)
    
    def embed_chunks(self, chunks: List[Dict], save_path: Optional[Path] = None) -> np.ndarray:
        """
        Embed all chunks from chunks list
        
        Args:
            chunks: List of chunk dictionaries
            save_path: Optional path to save embeddings
        
        Returns:
            NumPy array of embeddings
        """
        print(f"Embedding {len(chunks)} chunks using {self.provider}...")
        
        # Handle empty chunks
        if not chunks:
            print("⚠️  Warning: No chunks to embed!")
            empty_embeddings = np.array([]).reshape(0, self.dimension)
            if save_path:
                np.save(save_path, empty_embeddings)
            return empty_embeddings
        
        texts = [chunk['text'] for chunk in chunks]
        
        # Embed in batches with progress bar
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.embed_texts(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Save cache periodically
        self._save_cache()
        
        # Save embeddings if path provided
        if save_path:
            np.save(save_path, embeddings)
            print(f"Saved embeddings to: {save_path}")
        
        print(f"Created embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query
        
        Args:
            query: Query text
        
        Returns:
            NumPy array of embedding vector
        """
        embeddings = self.embed_texts([query], use_cache=False)
        return embeddings[0]


def main():
    """Test embedding functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create embeddings for chunks')
    parser.add_argument('--config', default='src/config.yaml', help='Path to config file')
    parser.add_argument('--input', default='data/chunks/chunks.jsonl', help='Input chunks file')
    parser.add_argument('--output', default='data/chunks/embeddings.npy', help='Output embeddings file')
    args = parser.parse_args()
    
    # Load chunks
    chunks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    # Create embeddings
    embedder = Embedder(args.config)
    embeddings = embedder.embed_chunks(chunks, save_path=Path(args.output))
    
    print(f"Embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()


