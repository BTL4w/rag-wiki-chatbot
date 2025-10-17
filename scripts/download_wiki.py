"""
Download Wikipedia articles using MediaWiki API
Supports incremental downloads and resume capability
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm
import yaml


class WikiDownloader:
    """Download Wikipedia articles via MediaWiki API"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.api_url = self.config['wikipedia']['api_base_url']
        self.language = self.config['wikipedia']['language']
        self.output_dir = Path(self.config['data']['raw_wiki_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.output_dir / "download_progress.json"
        self.session = requests.Session()
        
        # Set User-Agent header (required by Wikipedia API)
        self.session.headers.update({
            'User-Agent': 'RAG-Wiki-Bot/1.0 (Educational Project; Python/requests)'
        })
        
    def search_pages_by_topic(self, topic: str, limit: Optional[int] = None) -> List[str]:
        """
        Search Wikipedia pages by topic/keyword
        
        Args:
            topic: Topic or keyword to search for
            limit: Maximum number of pages to retrieve
        
        Returns:
            List of page titles matching the topic
        """
        pages = []
        max_pages = limit or self.config['wikipedia'].get('max_pages', 500)
        offset = 0
        
        pbar = tqdm(desc=f"Searching for '{topic}'", unit="pages")
        
        while len(pages) < max_pages:
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': topic,
                'srnamespace': 0,  # Main namespace (articles)
                'srlimit': 50,  # Max per request
                'sroffset': offset,
                'format': 'json'
            }
            
            try:
                response = self.session.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'query' not in data or 'search' not in data['query']:
                    break
                
                batch = [page['title'] for page in data['query']['search']]
                if not batch:
                    break
                
                pages.extend(batch)
                pbar.update(len(batch))
                
                if 'continue' not in data:
                    break
                
                offset = data['continue'].get('sroffset', 0)
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"\nError searching pages: {e}")
                break
        
        pbar.close()
        return pages[:max_pages]
    
    def get_pages_by_category(self, category: str, limit: Optional[int] = None) -> List[str]:
        """
        Get all pages in a Wikipedia category
        
        Args:
            category: Category name (e.g., "Artificial intelligence" or "Category:Artificial intelligence")
            limit: Maximum number of pages to retrieve
        
        Returns:
            List of page titles in the category
        """
        # Ensure category name has proper prefix
        if not category.startswith("Category:"):
            category = f"Category:{category}"
        
        pages = []
        continue_token = None
        max_pages = limit or self.config['wikipedia'].get('max_pages', 500)
        
        pbar = tqdm(desc=f"Fetching from {category}", unit="pages")
        
        while len(pages) < max_pages:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': category,
                'cmlimit': 'max',
                'cmnamespace': 0,  # Only articles
                'format': 'json'
            }
            
            if continue_token:
                params['cmcontinue'] = continue_token
            
            try:
                response = self.session.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'query' not in data or 'categorymembers' not in data['query']:
                    break
                
                batch = [page['title'] for page in data['query']['categorymembers']]
                if not batch:
                    break
                
                pages.extend(batch)
                pbar.update(len(batch))
                
                if 'continue' not in data:
                    break
                
                continue_token = data['continue'].get('cmcontinue')
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"\nError fetching category pages: {e}")
                break
        
        pbar.close()
        return pages[:max_pages]
    
    def get_all_pages(self, namespace: int = 0, limit: Optional[int] = None) -> List[str]:
        """
        Get list of all page titles from Wikipedia
        
        Args:
            namespace: Wikipedia namespace (0=articles, 14=categories, etc.)
            limit: Maximum number of pages to retrieve
        
        Returns:
            List of page titles
        """
        pages = []
        continue_token = None
        
        max_pages = limit or self.config['wikipedia'].get('max_pages')
        
        pbar = tqdm(desc="Fetching page list", unit="pages")
        
        while True:
            params = {
                'action': 'query',
                'list': 'allpages',
                'aplimit': 'max',
                'apnamespace': namespace,
                'format': 'json'
            }
            
            if continue_token:
                params['apcontinue'] = continue_token
            
            try:
                response = self.session.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                batch = [page['title'] for page in data['query']['allpages']]
                pages.extend(batch)
                pbar.update(len(batch))
                
                if max_pages and len(pages) >= max_pages:
                    pages = pages[:max_pages]
                    break
                
                if 'continue' not in data:
                    break
                    
                continue_token = data['continue']['apcontinue']
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"\nError fetching page list: {e}")
                break
        
        pbar.close()
        return pages
    
    def download_page(self, title: str) -> Optional[Dict]:
        """
        Download a single Wikipedia page with metadata
        
        Args:
            title: Page title
        
        Returns:
            Dictionary with page data or None if failed
        """
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'revisions|categories|info',
            'rvprop': 'content|timestamp|ids',
            'rvslots': 'main',
            'inprop': 'url',
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            
            if page_id == '-1':  # Page not found
                return None
            
            page = pages[page_id]
            
            if 'revisions' not in page:
                return None
            
            revision = page['revisions'][0]
            
            return {
                'page_id': page['pageid'],
                'title': page['title'],
                'url': page.get('fullurl', ''),
                'content': revision['slots']['main']['*'],
                'timestamp': revision['timestamp'],
                'revid': revision['revid'],
                'categories': [cat['title'] for cat in page.get('categories', [])],
                'length': len(revision['slots']['main']['*'])
            }
            
        except Exception as e:
            print(f"\nError downloading '{title}': {e}")
            return None
    
    def load_progress(self) -> Dict:
        """Load download progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'completed': [], 'failed': []}
    
    def save_progress(self, progress: Dict):
        """Save download progress to file"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    
    def save_page(self, page_data: Dict):
        """Save page data to JSON file"""
        filename = f"page_{page_data['page_id']}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
    
    def download_pages(self, page_list: List[str], resume: bool = True, mode: str = "all"):
        """
        Download a list of Wikipedia pages
        
        Args:
            page_list: List of page titles to download
            resume: Whether to resume from previous progress
            mode: Download mode (for progress tracking)
        """
        print(f"Found {len(page_list)} pages to download")
        
        # Load progress
        progress = self.load_progress() if resume else {'completed': [], 'failed': []}
        completed_set = set(progress['completed'])
        
        # Filter out already completed
        remaining = [p for p in page_list if p not in completed_set]
        print(f"Remaining: {len(remaining)} pages")
        
        if not remaining:
            print("No new pages to download!")
            return
        
        # Download pages
        failed_count = 0
        for title in tqdm(remaining, desc="Downloading pages"):
            page_data = self.download_page(title)
            
            if page_data:
                self.save_page(page_data)
                progress['completed'].append(title)
            else:
                progress['failed'].append(title)
                failed_count += 1
            
            # Save progress periodically
            if len(progress['completed']) % 10 == 0:
                self.save_progress(progress)
            
            # Rate limiting
            time.sleep(0.1)
        
        # Final save
        self.save_progress(progress)
        
        print(f"\nDownload complete!")
        print(f"Successfully downloaded: {len(progress['completed'])} pages")
        print(f"Failed: {len(progress['failed'])} pages")
        
        # Save summary
        summary = {
            'mode': mode,
            'total_pages': len(page_list),
            'completed': len(progress['completed']),
            'failed': len(progress['failed']),
            'language': self.language,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.output_dir / 'download_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def download_all(self, resume: bool = True):
        """
        Download all Wikipedia pages
        
        Args:
            resume: Whether to resume from previous progress
        """
        print(f"Downloading from {self.language} Wikipedia...")
        all_pages = self.get_all_pages()
        self.download_pages(all_pages, resume=resume, mode="all")
    
    def download_by_topic(self, topic: str, limit: Optional[int] = None, resume: bool = True):
        """
        Download Wikipedia pages by topic
        
        Args:
            topic: Topic or keyword to search for
            limit: Maximum number of pages to download
            resume: Whether to resume from previous progress
        """
        print(f"Downloading pages about '{topic}' from {self.language} Wikipedia...")
        pages = self.search_pages_by_topic(topic, limit=limit)
        self.download_pages(pages, resume=resume, mode=f"topic:{topic}")
    
    def download_by_category(self, category: str, limit: Optional[int] = None, resume: bool = True):
        """
        Download Wikipedia pages from a category
        
        Args:
            category: Category name
            limit: Maximum number of pages to download
            resume: Whether to resume from previous progress
        """
        print(f"Downloading pages from category '{category}' ({self.language} Wikipedia)...")
        pages = self.get_pages_by_category(category, limit=limit)
        self.download_pages(pages, resume=resume, mode=f"category:{category}")


def main():
    parser = argparse.ArgumentParser(
        description='Download Wikipedia articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all pages (default behavior)
  python download_wiki.py
  
  # Download pages about AI and machine learning
  python download_wiki.py --topic "artificial intelligence machine learning" --limit 100
  
  # Download pages from a specific category
  python download_wiki.py --category "Artificial intelligence" --limit 50
  
  # Download with custom config
  python download_wiki.py --config custom_config.yaml --topic "deep learning"
        """
    )
    
    parser.add_argument('--config', default='src/config.yaml', 
                        help='Path to config file')
    parser.add_argument('--no-resume', action='store_true', 
                        help='Start fresh (ignore progress)')
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--topic', type=str,
                            help='Download pages by topic/keyword (e.g., "artificial intelligence")')
    mode_group.add_argument('--category', type=str,
                            help='Download pages from a Wikipedia category (e.g., "Machine learning")')
    mode_group.add_argument('--all', action='store_true',
                            help='Download all Wikipedia pages (default if no mode specified)')
    
    parser.add_argument('--limit', type=int,
                        help='Maximum number of pages to download')
    
    args = parser.parse_args()
    
    downloader = WikiDownloader(args.config)
    resume = not args.no_resume
    
    # Execute based on selected mode
    if args.topic:
        print(f"\n{'='*60}")
        print(f"Mode: Download by TOPIC")
        print(f"Topic: {args.topic}")
        if args.limit:
            print(f"Limit: {args.limit} pages")
        print(f"{'='*60}\n")
        downloader.download_by_topic(args.topic, limit=args.limit, resume=resume)
    
    elif args.category:
        print(f"\n{'='*60}")
        print(f"Mode: Download by CATEGORY")
        print(f"Category: {args.category}")
        if args.limit:
            print(f"Limit: {args.limit} pages")
        print(f"{'='*60}\n")
        downloader.download_by_category(args.category, limit=args.limit, resume=resume)
    
    else:
        # Default: download all
        print(f"\n{'='*60}")
        print(f"Mode: Download ALL pages")
        if args.limit:
            print(f"Limit: {args.limit} pages")
        print(f"{'='*60}\n")
        downloader.download_all(resume=resume)


if __name__ == "__main__":
    main()


