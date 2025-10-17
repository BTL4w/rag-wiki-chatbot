"""
LLM-based answer generation
Supports multiple LLM providers (OpenAI, Gemini, Anthropic, local models)
"""

import os
from typing import List, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Generator:
    """Generate answers using LLM"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.provider = self.config['generation']['provider']
        self.model_name = self.config['generation']['model_name']
        self.temperature = self.config['generation']['temperature']
        self.max_tokens = self.config['generation'].get('max_tokens', 500)
        self.max_completion_tokens = self.config['generation'].get('max_completion_tokens', 500)
        self.top_p = self.config['generation'].get('top_p', 1.0)
        
        # Load prompts
        self.system_prompt = self.config['prompts']['system_prompt']
        self.context_template = self.config['prompts']['context_template']
        self.qa_template = self.config['prompts']['qa_template']
        
        # Initialize client
        self.client = self._init_client()
    
    def _init_client(self):
        """Initialize LLM client"""
        api_key = self.config['generation'].get('api_key')
        
        if self.provider == "openai":
            from openai import OpenAI
            key = api_key or os.getenv('OPENAI_API_KEY')
            if not key:
                raise ValueError("OpenAI API key not found")
            return OpenAI(api_key=key)
        
        elif self.provider == "gemini":
            import google.generativeai as genai
            key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not key:
                raise ValueError("Gemini API key not found")
            genai.configure(api_key=key)
            return genai.GenerativeModel(self.model_name)
        
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not key:
                raise ValueError("Anthropic API key not found")
            return Anthropic(api_key=key)
        
        elif self.provider == "local":
            # For local models (e.g., via Ollama or transformers)
            return None
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from chunks
        
        Args:
            chunks: List of retrieved chunk dictionaries
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context = self.context_template.format(
                index=i,
                title=chunk.get('title', 'Unknown'),
                section=chunk.get('section', 'Unknown'),
                text=chunk['text']
            )
            context_parts.append(context)
        
        return '\n\n'.join(context_parts)
    
    def build_prompt(self, question: str, context: str) -> str:
        """
        Build final prompt
        
        Args:
            question: User question
            context: Formatted context from chunks
        
        Returns:
            Complete prompt
        """
        return self.qa_template.format(
            context=context,
            question=question
        )
    
    def generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API"""
        # Use max_completion_tokens for newer models (gpt-4o, gpt-4o-mini)
        # Fall back to max_tokens for older models
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p
        }
        
        # For newer models (gpt-4o, gpt-4o-mini), use max_completion_tokens
        if "gpt-4o" in self.model_name.lower():
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
        
        response = self.client.chat.completions.create(**kwargs)
        
        return response.choices[0].message.content
    
    def generate_gemini(self, prompt: str) -> str:
        """Generate using Gemini API"""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        
        response = self.client.generate_content(
            full_prompt,
            generation_config={
                'temperature': self.temperature,
                'max_output_tokens': self.max_tokens,
                'top_p': self.top_p
            }
        )
        
        return response.text
    
    def generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic API"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    def generate(self, question: str, chunks: List[Dict]) -> Dict:
        """
        Generate answer from question and retrieved chunks
        
        Args:
            question: User question
            chunks: Retrieved chunks
        
        Returns:
            Dictionary with answer and metadata
        """
        # Build context and prompt
        context = self.build_context(chunks)
        prompt = self.build_prompt(question, context)
        
        # Generate answer
        if self.provider == "openai":
            answer = self.generate_openai(prompt)
        elif self.provider == "gemini":
            answer = self.generate_gemini(prompt)
        elif self.provider == "anthropic":
            answer = self.generate_anthropic(prompt)
        else:
            raise ValueError(f"Provider {self.provider} not implemented")
        
        return {
            'answer': answer,
            'question': question,
            'num_chunks': len(chunks),
            'sources': self._extract_sources(chunks)
        }
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks"""
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            sources.append({
                'index': i,
                'title': chunk.get('title', 'Unknown'),
                'section': chunk.get('section', 'Unknown'),
                'url': chunk.get('url', ''),
                'score': chunk.get('score', 0.0)
            })
        
        return sources


def main():
    """Test generator"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Test generator')
    parser.add_argument('--config', default='src/config.yaml', help='Path to config file')
    parser.add_argument('--question', required=True, help='Question to answer')
    parser.add_argument('--chunks', required=True, help='JSON file with chunks')
    args = parser.parse_args()
    
    # Load chunks
    with open(args.chunks, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Generate answer
    generator = Generator(args.config)
    result = generator.generate(args.question, chunks)
    
    print(f"Question: {result['question']}\n")
    print(f"Answer: {result['answer']}\n")
    print(f"Sources ({len(result['sources'])}):")
    for source in result['sources']:
        print(f"  [{source['index']}] {source['title']} - {source['section']}")


if __name__ == "__main__":
    main()


