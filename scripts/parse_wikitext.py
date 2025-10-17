"""
Parse Wikipedia wikitext to plain text
Removes markup, templates, and extracts clean content
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import mwparserfromhell
from tqdm import tqdm
import yaml
import re


class WikiParser:
    """Parse Wikipedia wikitext to clean plain text"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.input_dir = Path(self.config['data']['raw_wiki_dir'])
        self.output_dir = Path(self.config['data']['parsed_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_sections = self.config['parsing'].get('keep_sections', [])
        self.remove_sections = self.config['parsing'].get('remove_sections', [])
        self.min_length = self.config['parsing'].get('min_text_length', 100)
    
    def parse_wikitext(self, wikitext: str) -> str:
        """
        Parse wikitext to plain text
        
        Args:
            wikitext: Raw Wikipedia markup text
        
        Returns:
            Clean plain text
        """
        try:
            wikicode = mwparserfromhell.parse(wikitext)
            
            # Remove templates (mostly infoboxes, navboxes)
            for template in wikicode.filter_templates():
                try:
                    wikicode.remove(template)
                except ValueError:
                    pass
            
            # Convert to plain text
            text = wikicode.strip_code()
            
            # Clean up
            text = self.clean_text(text)
            
            return text
            
        except Exception as e:
            print(f"Error parsing wikitext: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        
        # Remove citation markers like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove file/image references
        text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces and newlines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = re.sub(r'\n{2,}', '\n\n', text)
        
        return text.strip()
    
    def extract_sections(self, wikitext: str) -> Dict[str, str]:
        """
        Extract sections from wikitext
        
        Args:
            wikitext: Raw wikitext
        
        Returns:
            Dictionary of section_name -> content
        """
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        lines = wikitext.split('\n')
        
        for line in lines:
            # Check for section headers (== Section ==)
            section_match = re.match(r'^(={2,})\s*(.+?)\s*\1$', line)
            
            if section_match:
                # Save previous section
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections[current_section] = content
                
                # Start new section
                current_section = section_match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections[current_section] = content
        
        return sections
    
    def should_keep_section(self, section_name: str) -> bool:
        """Check if section should be kept based on config"""
        # Check if in remove list
        for remove_pattern in self.remove_sections:
            if remove_pattern.lower() in section_name.lower():
                return False
        
        # If keep_sections is specified and not empty, only keep those
        if self.keep_sections:
            for keep_pattern in self.keep_sections:
                if keep_pattern.lower() in section_name.lower():
                    return True
            return False
        
        return True
    
    def parse_page(self, page_data: Dict) -> Optional[Dict]:
        """
        Parse a single Wikipedia page
        
        Args:
            page_data: Raw page data from download
        
        Returns:
            Parsed page data or None if too short
        """
        try:
            # Extract sections
            sections = self.extract_sections(page_data['content'])
            
            # Parse each section
            parsed_sections = {}
            for section_name, content in sections.items():
                if self.should_keep_section(section_name):
                    parsed_text = self.parse_wikitext(content)
                    if parsed_text:
                        parsed_sections[section_name] = parsed_text
            
            # Combine all text
            full_text = '\n\n'.join(parsed_sections.values())
            
            # Check minimum length
            if len(full_text) < self.min_length:
                return None
            
            return {
                'page_id': page_data['page_id'],
                'title': page_data['title'],
                'url': page_data.get('url', ''),
                'sections': parsed_sections,
                'full_text': full_text,
                'metadata': {
                    'timestamp': page_data.get('timestamp'),
                    'revid': page_data.get('revid'),
                    'categories': page_data.get('categories', []),
                    'length': len(full_text),
                    'num_sections': len(parsed_sections)
                }
            }
            
        except Exception as e:
            print(f"Error parsing page {page_data.get('title', 'unknown')}: {e}")
            return None
    
    def parse_all(self):
        """Parse all downloaded Wikipedia pages"""
        print("Parsing Wikipedia pages...")
        
        # Get all raw page files
        page_files = list(self.input_dir.glob("page_*.json"))
        print(f"Found {len(page_files)} pages to parse")
        
        parsed_count = 0
        skipped_count = 0
        
        for filepath in tqdm(page_files, desc="Parsing pages"):
            try:
                # Load raw page
                with open(filepath, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                
                # Parse page
                parsed = self.parse_page(page_data)
                
                if parsed:
                    # Save parsed page
                    output_file = self.output_dir / f"parsed_{page_data['page_id']}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)
                    parsed_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"\nError processing {filepath}: {e}")
                skipped_count += 1
        
        print(f"\nParsing complete!")
        print(f"Successfully parsed: {parsed_count} pages")
        print(f"Skipped: {skipped_count} pages")
        
        # Save summary
        summary = {
            'total_files': len(page_files),
            'parsed': parsed_count,
            'skipped': skipped_count,
            'output_dir': str(self.output_dir)
        }
        
        with open(self.output_dir / 'parsing_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Parse Wikipedia wikitext')
    parser.add_argument('--config', default='src/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    wiki_parser = WikiParser(args.config)
    wiki_parser.parse_all()


if __name__ == "__main__":
    main()


