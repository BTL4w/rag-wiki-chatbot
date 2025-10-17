"""
Convenience script to run the entire RAG pipeline from start to finish
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


def run_command(cmd, description):
    """Run a command and track its execution"""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed!")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run RAG Wiki Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with topic-based download
  python run_pipeline.py --topic "artificial intelligence machine learning" --limit 100
  
  # Run full pipeline with category-based download
  python run_pipeline.py --category "Artificial intelligence" --limit 50
  
  # Skip download and process existing data
  python run_pipeline.py --skip-download
  
  # Run only specific step
  python run_pipeline.py --step parse
        """
    )
    
    parser.add_argument('--step', choices=['download', 'parse', 'chunk', 'embed', 'index', 'all'],
                       default='all', help='Which step to run')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip download step (use existing data)')
    
    # Download options
    download_group = parser.add_argument_group('Download options')
    download_mode = download_group.add_mutually_exclusive_group()
    download_mode.add_argument('--topic', type=str,
                              help='Download pages by topic/keyword (e.g., "artificial intelligence")')
    download_mode.add_argument('--category', type=str,
                              help='Download pages from Wikipedia category (e.g., "Machine learning")')
    download_mode.add_argument('--all-pages', action='store_true',
                              help='Download all Wikipedia pages')
    
    download_group.add_argument('--limit', type=int,
                               help='Maximum number of pages to download')
    download_group.add_argument('--no-resume', action='store_true',
                               help='Start fresh download (ignore previous progress)')
    
    args = parser.parse_args()
    
    # Build download command with options
    download_cmd = 'python scripts/download_wiki.py'
    download_desc = 'Download Wikipedia articles'
    
    if args.topic:
        download_cmd += f' --topic "{args.topic}"'
        download_desc = f'Download pages about: {args.topic}'
    elif args.category:
        download_cmd += f' --category "{args.category}"'
        download_desc = f'Download from category: {args.category}'
    elif args.all_pages:
        download_cmd += ' --all'
        download_desc = 'Download all Wikipedia pages'
    
    if args.limit:
        download_cmd += f' --limit {args.limit}'
        download_desc += f' (max {args.limit} pages)'
    
    if args.no_resume:
        download_cmd += ' --no-resume'
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         RAG Wiki Chatbot - Pipeline Runner               ║
    ║                                                           ║
    ║  This script will run the complete data pipeline:        ║
    ║  1. Download Wikipedia articles                          ║
    ║  2. Parse wikitext to plain text                         ║
    ║  3. Chunk text into segments                             ║
    ║  4. Create embeddings                                    ║
    ║  5. Build vector index                                   ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Show download configuration
    if not args.skip_download and (args.step == 'all' or args.step == 'download'):
        print(f"📥 Download Configuration:")
        if args.topic:
            print(f"   Mode: Topic Search")
            print(f"   Topic: {args.topic}")
        elif args.category:
            print(f"   Mode: Category")
            print(f"   Category: {args.category}")
        else:
            print(f"   Mode: Default (topic search or all pages)")
        
        if args.limit:
            print(f"   Limit: {args.limit} pages")
        if args.no_resume:
            print(f"   Resume: No (fresh start)")
        else:
            print(f"   Resume: Yes (continue from previous)")
        print()
    
    steps = []
    
    if args.step == 'all':
        if not args.skip_download:
            steps.append(('download', download_cmd, download_desc))
        steps.extend([
            ('parse', 'python scripts/parse_wikitext.py', 'Parse wikitext'),
            ('chunk', 'python scripts/chunk_text.py', 'Chunk text'),
            ('embed', 'python -m src.indexer.embedder --input data/chunks/chunks.jsonl --output data/chunks/embeddings.npy', 'Create embeddings'),
            ('index', 'python -m src.indexer.indexer --embeddings data/chunks/embeddings.npy --chunks data/chunks/chunks.jsonl', 'Build index')
        ])
    else:
        step_map = {
            'download': (download_cmd, download_desc),
            'parse': ('python scripts/parse_wikitext.py', 'Parse wikitext'),
            'chunk': ('python scripts/chunk_text.py', 'Chunk text'),
            'embed': ('python -m src.indexer.embedder --input data/chunks/chunks.jsonl --output data/chunks/embeddings.npy', 'Create embeddings'),
            'index': ('python -m src.indexer.indexer --embeddings data/chunks/embeddings.npy --chunks data/chunks/chunks.jsonl', 'Build index')
        }
        cmd, desc = step_map[args.step]
        steps.append((args.step, cmd, desc))
    
    # Run all steps
    overall_start = time.time()
    success_count = 0
    
    for step_name, cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1
        else:
            print(f"\n❌ Pipeline failed at step: {step_name}")
            print("Fix the error and try again, or run individual steps.")
            sys.exit(1)
    
    overall_elapsed = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"✅ Pipeline completed successfully!")
    print(f"Completed {success_count}/{len(steps)} steps")
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    print("Next steps:")
    print("  Run the chatbot:  python -m src.qa_pipeline")
    # print("  2. Run evaluation:   python eval/evaluate.py --test-set eval/test_questions.json")
    # print("  3. Try the notebook: jupyter notebook notebooks/demo.ipynb")
    print()


if __name__ == "__main__":
    main()

