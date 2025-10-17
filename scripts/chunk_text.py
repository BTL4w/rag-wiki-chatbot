"""
Chunk parsed Wikipedia text into segments for embedding
Supports semantic and sliding window chunking strategies
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from tqdm import tqdm
import re


class TextChunker:
    """Chunk text into segments for embedding"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.input_dir = Path(self.config['data']['parsed_dir'])
        self.output_dir = Path(self.config['data']['chunks_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.strategy = self.config['chunking']['strategy']
        self.chunk_size = self.config['chunking']['chunk_size']
        self.overlap = self.config['chunking']['overlap']
        self.min_chunk_size = self.config['chunking'].get('min_chunk_size', 50)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (words * 1.3 for multilingual)
        For production, use actual tokenizer from embedding model
        """
        words = len(text.split())
        return int(words * 1.3)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        Handles Vietnamese and English punctuation
        """
        # Split on sentence endings
        sentences = re.split(r'[.!?।]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def semantic_chunking(self, sections: Dict[str, str], page_title: str) -> List[Dict]:
        """
        Semantic-aware chunking: preserve section boundaries
        Split long sections at sentence boundaries
        
        Args:
            sections: Dictionary of section_name -> content
            page_title: Title of the page
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        chunk_id = 0
        
        for section_name, content in sections.items():
            sentences = self.split_into_sentences(content)
            
            if not sentences:
                continue
            
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self.estimate_tokens(sentence)
                
                # If adding this sentence exceeds chunk_size, save current chunk
                if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    
                    if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                        chunks.append({
                            'chunk_id': chunk_id,
                            'text': chunk_text,
                            'section': section_name,
                            'title': page_title,
                            'tokens': current_tokens
                        })
                        chunk_id += 1
                    
                    # Keep last sentences for overlap
                    overlap_text = ' '.join(current_chunk[-(len(current_chunk)//3):])
                    overlap_tokens = self.estimate_tokens(overlap_text)
                    
                    if overlap_tokens <= self.overlap:
                        current_chunk = current_chunk[-(len(current_chunk)//3):]
                        current_tokens = overlap_tokens
                    else:
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            # Save remaining chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'section': section_name,
                        'title': page_title,
                        'tokens': current_tokens
                    })
                    chunk_id += 1
        
        return chunks
    
    def sliding_window_chunking(self, text: str, page_title: str) -> List[Dict]:
        """
        Sliding window chunking with overlap
        
        Args:
            text: Full text to chunk
            page_title: Title of the page
        
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunk_id = 0
        current_chunk = []
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                
                if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'section': 'full_text',
                        'title': page_title,
                        'tokens': current_tokens
                    })
                    chunk_id += 1
                
                # Move back for overlap
                overlap_sentences = int(len(current_chunk) * (self.overlap / self.chunk_size))
                i = i - overlap_sentences if overlap_sentences > 0 else i
                
                current_chunk = []
                current_tokens = 0
                continue
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Save remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'section': 'full_text',
                    'title': page_title,
                    'tokens': current_tokens
                })
        
        return chunks
    
    def chunk_page(self, parsed_data: Dict) -> List[Dict]:
        """
        Chunk a parsed page
        
        Args:
            parsed_data: Parsed page data
        
        Returns:
            List of chunks with metadata
        """
        page_id = parsed_data['page_id']
        page_title = parsed_data['title']
        url = parsed_data.get('url', '')
        metadata = parsed_data.get('metadata', {})
        
        if self.strategy == 'semantic':
            chunks = self.semantic_chunking(parsed_data['sections'], page_title)
        else:  # sliding_window
            chunks = self.sliding_window_chunking(parsed_data['full_text'], page_title)
        
        # Add page metadata to each chunk
        for chunk in chunks:
            chunk['page_id'] = page_id
            chunk['url'] = url
            chunk['page_metadata'] = metadata
        
        return chunks
    
    def chunk_all(self):
        """Chunk all parsed pages"""
        print("Chunking parsed Wikipedia pages...")
        
        # Get all parsed files
        parsed_files = list(self.input_dir.glob("parsed_*.json"))
        print(f"Found {len(parsed_files)} parsed pages")
        
        all_chunks = []
        total_chunks = 0
        
        for filepath in tqdm(parsed_files, desc="Chunking pages"):
            try:
                # Load parsed page
                with open(filepath, 'r', encoding='utf-8') as f:
                    parsed_data = json.load(f)
                
                # Chunk page
                chunks = self.chunk_page(parsed_data)
                
                if chunks:
                    all_chunks.extend(chunks)
                    total_chunks += len(chunks)
                    
            except Exception as e:
                print(f"\nError chunking {filepath}: {e}")
        
        # Assign global chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk['global_chunk_id'] = i
        
        # Check if we have any data
        if not parsed_files:
            print("\n⚠️  Warning: No parsed files found!")
            print(f"   Looking in: {self.input_dir}")
            print("\n💡 Tips:")
            print("   1. Make sure you ran: python scripts/parse_wikitext.py")
            print("   2. Check if download step completed successfully")
            print("   3. Check data/parsed/ directory for files")
            return
        
        if not all_chunks:
            print("\n⚠️  Warning: No chunks created from parsed files!")
            return
        
        # Save all chunks to JSONL
        output_file = self.output_dir / 'chunks.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"\nChunking complete!")
        print(f"Total chunks created: {total_chunks}")
        print(f"Average chunks per page: {total_chunks / len(parsed_files):.1f}")
        
        # Calculate statistics
        avg_tokens = sum(c['tokens'] for c in all_chunks) / len(all_chunks) if all_chunks else 0
        
        # Save summary
        summary = {
            'total_pages': len(parsed_files),
            'total_chunks': total_chunks,
            'avg_chunks_per_page': total_chunks / len(parsed_files) if parsed_files else 0,
            'avg_tokens_per_chunk': avg_tokens,
            'strategy': self.strategy,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'output_file': str(output_file)
        }
        
        with open(self.output_dir / 'chunking_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Chunk Wikipedia text')
    parser.add_argument('--config', default='src/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    chunker = TextChunker(args.config)
    chunker.chunk_all()


if __name__ == "__main__":
    main()


