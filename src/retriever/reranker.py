"""
Rerank retrieved documents using cross-encoder models
Improves precision by reordering initial retrieval results
"""

from typing import List, Dict
import yaml
import numpy as np


class Reranker:
    """Rerank documents using cross-encoder"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.use_reranker = self.config['retrieval'].get('use_reranker', False)
        self.final_k = self.config['retrieval'].get('final_k', 5)
        
        if self.use_reranker:
            self.model_name = self.config['retrieval']['reranker_model']
            self.model = self._load_model()
        else:
            self.model = None
    
    def _load_model(self):
        """Load cross-encoder model"""
        from sentence_transformers import CrossEncoder
        
        print(f"Loading reranker model: {self.model_name}")
        model = CrossEncoder(self.model_name, trust_remote_code=True)
        return model
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = None) -> List[Dict]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Query text
            documents: List of document dictionaries
            top_k: Number of documents to return (default: use config final_k)
        
        Returns:
            Reranked list of documents
        """
        if not self.use_reranker or self.model is None:
            # Just return top documents without reranking
            return documents[:top_k or self.final_k]
        
        if not documents:
            return []
        
        if top_k is None:
            top_k = self.final_k
        
        # Prepare query-document pairs
        pairs = [[query, doc['text']] for doc in documents]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Attach reranker scores
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
            doc['original_rank'] = i + 1
        
        # Sort by reranker score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
    
    def score_pairs(self, query: str, texts: List[str]) -> np.ndarray:
        """
        Score query-text pairs
        
        Args:
            query: Query text
            texts: List of texts to score
        
        Returns:
            Array of scores
        """
        if not self.use_reranker or self.model is None:
            raise ValueError("Reranker not enabled")
        
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(pairs)
        return scores


def main():
    """Test reranker"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Test reranker')
    parser.add_argument('--config', default='src/config.yaml', help='Path to config file')
    parser.add_argument('--query', required=True, help='Query text')
    parser.add_argument('--docs', required=True, help='JSON file with documents')
    args = parser.parse_args()
    
    # Load documents
    with open(args.docs, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Rerank
    reranker = Reranker(args.config)
    reranked = reranker.rerank(args.query, documents)
    
    print(f"Query: {args.query}\n")
    print("Reranked results:")
    
    for i, doc in enumerate(reranked, 1):
        print(f"{i}. Score: {doc['rerank_score']:.4f} (was rank {doc['original_rank']})")
        print(f"   {doc['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()


