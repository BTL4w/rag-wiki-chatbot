"""
Main QA Pipeline - Glue all components together
Query -> Retrieval -> Reranking -> Generation -> Answer
"""

import time
from typing import Dict, List, Optional
import yaml
from pathlib import Path
import json


class QAPipeline:
    """End-to-end QA pipeline"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.cache_enabled = self.config['system'].get('cache_enabled', False)
        self.cache = {} if self.cache_enabled else None
        
        # Components (lazy loading)
        self.retriever = None
        self.reranker = None
        self.generator = None
    
    def load_components(self):
        """Load all pipeline components"""
        print("Loading RAG pipeline components...")
        
        # Load retriever
        print("Loading retriever...")
        from .retriever import Retriever
        self.retriever = Retriever()
        self.retriever.load_components()
        
        # Load reranker
        print("Loading reranker...")
        from .retriever import Reranker
        self.reranker = Reranker()
        
        # Load generator
        print("Loading generator...")
        from .generator import Generator
        self.generator = Generator()
        
        print("Pipeline loaded successfully!")
    
    def query(self, question: str, return_chunks: bool = False, 
              debug: bool = False) -> Dict:
        """
        Process a question through the full pipeline
        
        Args:
            question: User question
            return_chunks: Whether to include retrieved chunks in response
            debug: Whether to include debug information
        
        Returns:
            Dictionary with answer and metadata
        """
        if not all([self.retriever, self.reranker, self.generator]):
            raise ValueError("Pipeline not loaded. Call load_components() first.")
        
        # Check cache
        if self.cache_enabled and question in self.cache:
            cached = self.cache[question].copy()
            cached['from_cache'] = True
            return cached
        
        start_time = time.time()
        
        # Step 1: Retrieval
        retrieval_start = time.time()
        retrieved_chunks = self.retriever.retrieve(question)
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_chunks:
            return {
                'question': question,
                'answer': 'Không tìm thấy thông tin liên quan trong tài liệu.',
                'sources': [],
                'num_retrieved': 0,
                'retrieval_time': retrieval_time,
                'total_time': time.time() - start_time
            }
        
        # Step 2: Reranking
        rerank_start = time.time()
        reranked_chunks = self.reranker.rerank(question, retrieved_chunks)
        rerank_time = time.time() - rerank_start
        
        # Step 3: Generation
        gen_start = time.time()
        result = self.generator.generate(question, reranked_chunks)
        gen_time = time.time() - gen_start
        
        # Compile response
        response = {
            'question': question,
            'answer': result['answer'],
            'sources': result['sources'],
            'num_retrieved': len(retrieved_chunks),
            'num_reranked': len(reranked_chunks),
            'retrieval_time': retrieval_time,
            'rerank_time': rerank_time,
            'generation_time': gen_time,
            'total_time': time.time() - start_time,
            'from_cache': False
        }
        
        # Add chunks if requested
        if return_chunks:
            response['retrieved_chunks'] = retrieved_chunks
            response['reranked_chunks'] = reranked_chunks
        
        # Add debug info if requested
        if debug:
            response['debug'] = {
                'retrieval_scores': [c.get('score', 0) for c in retrieved_chunks[:5]],
                'rerank_scores': [c.get('rerank_score', 0) for c in reranked_chunks],
                'config': {
                    'retriever': self.config['retrieval'],
                    'generator': self.config['generation']
                }
            }
        
        # Cache result
        if self.cache_enabled:
            self.cache[question] = response.copy()
        
        return response
    
    def batch_query(self, questions: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Process multiple questions
        
        Args:
            questions: List of questions
            show_progress: Whether to show progress bar
        
        Returns:
            List of response dictionaries
        """
        results = []
        
        iterator = questions
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(questions, desc="Processing questions")
        
        for question in iterator:
            result = self.query(question)
            results.append(result)
        
        return results
    
    def format_response(self, response: Dict, include_sources: bool = True) -> str:
        """
        Format response for display
        
        Args:
            response: Response dictionary from query()
            include_sources: Whether to include source list
        
        Returns:
            Formatted string
        """
        lines = []
        
        lines.append(f"Câu hỏi: {response['question']}")
        lines.append("")
        lines.append(f"Trả lời: {response['answer']}")
        
        if include_sources and response['sources']:
            lines.append("")
            lines.append(f"Nguồn tham khảo ({len(response['sources'])}):")
            for source in response['sources']:
                lines.append(f"  [{source['index']}] {source['title']} - {source['section']}")
                if source.get('url'):
                    lines.append(f"      🔗 {source['url']}")
        
        lines.append("")
        lines.append(f"Thời gian xử lý: {response['total_time']:.2f}s")
        
        return '\n'.join(lines)
    
    def save_conversation(self, filepath: str, conversations: List[Dict]):
        """
        Save conversation history
        
        Args:
            filepath: Output file path
            conversations: List of QA pairs
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, filepath: str) -> List[Dict]:
        """
        Load conversation history
        
        Args:
            filepath: Input file path
        
        Returns:
            List of QA pairs
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    """Interactive QA interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG Wiki QA Pipeline')
    parser.add_argument('--config', default='src/config.yaml', help='Config file')
    parser.add_argument('--question', help='Single question (non-interactive)')
    parser.add_argument('--questions-file', help='File with multiple questions (one per line)')
    parser.add_argument('--output', help='Output file for batch processing')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = QAPipeline(args.config)
    pipeline.load_components()
    
    # Single question mode
    if args.question:
        response = pipeline.query(args.question, debug=args.debug)
        print("\n" + pipeline.format_response(response))
        return
    
    # Batch mode
    if args.questions_file:
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        results = pipeline.batch_query(questions)
        
        if args.output:
            pipeline.save_conversation(args.output, results)
            print(f"Results saved to {args.output}")
        else:
            for result in results:
                print("\n" + "="*80)
                print(pipeline.format_response(result))
        return
    
    # Interactive mode
    print("\n" + "="*80)
    print("RAG Wiki Chatbot - Interactive Mode")
    print("="*80)
    print("Nhập câu hỏi của bạn (hoặc 'quit' để thoát)")
    print("="*80 + "\n")
    
    conversation = []
    
    while True:
        try:
            question = input("\n💬 Bạn: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q', 'thoát']:
                print("\nTạm biệt!")
                break
            
            # Process question
            response = pipeline.query(question, debug=args.debug)
            conversation.append(response)
            
            # Display response
            print(f"\n🤖 Bot: {response['answer']}")
            
            if response['sources']:
                print(f"\n📚 Nguồn ({len(response['sources'])}):")
                for source in response['sources']:
                    print(f"  [{source['index']}] {source['title']} - {source['section']}")
                    if source.get('url'):
                        print(f"      🔗 {source['url']}")
            
            print(f"\n⏱️ Thời gian: {response['total_time']:.2f}s")
            
            if args.debug:
                print(f"\n🔍 Debug:")
                print(f"  Retrieved: {response['num_retrieved']}")
                print(f"  Reranked: {response['num_reranked']}")
                print(f"  Retrieval: {response['retrieval_time']:.3f}s")
                print(f"  Rerank: {response['rerank_time']:.3f}s")
                print(f"  Generation: {response['generation_time']:.3f}s")
        
        except KeyboardInterrupt:
            print("\n\nĐã ngắt. Tạm biệt!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # Save conversation if any
    if conversation:
        save = input("\nLưu cuộc hội thoại? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"conversation_{int(time.time())}.json"
            pipeline.save_conversation(filename, conversation)
            print(f"Đã lưu vào {filename}")


if __name__ == "__main__":
    main()


