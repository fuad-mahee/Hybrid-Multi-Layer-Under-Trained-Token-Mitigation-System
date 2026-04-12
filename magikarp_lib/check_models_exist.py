"""
Simplified version: Test undertrained tokens existence without requiring torch.
Uses the tokenizers library directly or requests from HuggingFace API.
"""

import json
import requests
from typing import Dict, List
from pathlib import Path


class SimpleTokenTester:
    """Test undertrained tokens using HuggingFace API (no model loading required)."""
    
    def __init__(self, dictionary_path: str = "undertrained_tokens_dictionary.json"):
        """Initialize with the undertrained tokens dictionary."""
        print(f"Loading undertrained tokens dictionary...")
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            self.token_dict = json.load(f)
        
        total_tokens = sum(len(tokens) for tokens in self.token_dict.values())
        print(f"Loaded {len(self.token_dict)} models with {total_tokens} total undertrained tokens\n")
    
    def map_to_hf_model_id(self, internal_name: str) -> str:
        """Convert internal model name to HuggingFace model ID."""
        mappings = {
            'openai_community': 'openai-community',
            'meta_llama': 'meta-llama',
            'mistralai': 'mistralai',
            'CohereForAI': 'CohereForAI',
            'EleutherAI': 'EleutherAI',
            'deepseek_ai': 'deepseek-ai',
            'google': 'google',
            'microsoft': 'microsoft',
            'allenai': 'allenai',
            'ai21labs': 'ai21labs',
            'bigcode': 'bigcode',
            'LGAI_EXAONE': 'LGAI-EXAONE',
            'Qwen': 'Qwen',
            'internlm': 'internlm',
            'facebook': 'facebook',
            'upstage': 'upstage',
            'Rakuten': 'Rakuten',
            'Skywork': 'Skywork',
            'tiiuae': 'tiiuae',
            'kyutai': 'kyutai',
            'HuggingFaceH4': 'HuggingFaceH4',
            'Nexusflow': 'Nexusflow',
            'stabilityai': 'stabilityai',
            'm_a_p': 'm-a-p',
            'chuxin_llm': 'chuxin-llm',
            'ibm_granite': 'ibm-granite',
            'utter_project': 'utter-project',
            'trillionlabs': 'trillionlabs',
            'PleIAs': 'PleIAs',
            'Fugaku_LLM': 'Fugaku-LLM',
            '01_ai': '01-ai',
            '152334H': '152334H',
            'benjamin': 'benjamin',
            'h2oai': 'h2oai',
            'UW': 'UW',
        }
        
        for internal_org, hf_org in mappings.items():
            if internal_name.startswith(internal_org + '_'):
                rest = internal_name[len(internal_org)+1:].replace('_', '-')
                return f"{hf_org}/{rest}"
        
        parts = internal_name.split('_', 1)
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1].replace('_', '-')}"
        return internal_name
    
    def check_model_on_hub(self, model_id: str) -> Dict:
        """Check if model exists on HuggingFace Hub."""
        url = f"https://huggingface.co/api/models/{model_id}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                return {
                    'exists': True,
                    'model_id': model_id,
                    'downloads': model_info.get('downloads', 'unknown'),
                    'likes': model_info.get('likes', 'unknown'),
                    'last_modified': model_info.get('lastModified', 'unknown')
                }
            else:
                return {
                    'exists': False,
                    'model_id': model_id,
                    'error': f'HTTP {response.status_code}'
                }
        except Exception as e:
            return {
                'exists': False,
                'model_id': model_id,
                'error': str(e)
            }
    
    def test_models_existence(
        self,
        models_to_test: List[str] = None,
        output_file: str = "model_existence_check.json"
    ):
        """Check which models still exist on HuggingFace Hub."""
        if models_to_test is None:
            models_to_test = list(self.token_dict.keys())
        
        print(f"{'='*80}")
        print(f"Checking existence of {len(models_to_test)} models on HuggingFace Hub")
        print(f"{'='*80}\n")
        
        results = {}
        
        for i, internal_name in enumerate(models_to_test, 1):
            hf_model_id = self.map_to_hf_model_id(internal_name)
            print(f"[{i}/{len(models_to_test)}] {internal_name} -> {hf_model_id}...")
            
            check_result = self.check_model_on_hub(hf_model_id)
            token_count = len(self.token_dict[internal_name])
            
            results[internal_name] = {
                'hf_model_id': hf_model_id,
                'undertrained_token_count': token_count,
                **check_result
            }
            
            if check_result['exists']:
                print(f"  ✓ Found (Downloads: {check_result.get('downloads', 'N/A')})")
            else:
                print(f"  ✗ Not found: {check_result.get('error', 'Unknown error')}")
        
        # Save results
        print(f"\n{'='*80}")
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print summary of results."""
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        
        total = len(results)
        found = sum(1 for r in results.values() if r.get('exists'))
        not_found = total - found
        
        print(f"Total models checked: {total}")
        print(f"Found on HuggingFace: {found} ({100*found/total:.1f}%)")
        print(f"Not found: {not_found} ({100*not_found/total:.1f}%)")
        
        if not_found > 0:
            print(f"\n{'='*80}")
            print("MODELS NOT FOUND")
            print(f"{'='*80}\n")
            for internal_name, result in results.items():
                if not result.get('exists'):
                    print(f"  - {internal_name} ({result['hf_model_id']})")
                    print(f"    Reason: {result.get('error', 'Unknown')}")
        
        # Most popular models
        popular = [(name, r) for name, r in results.items() 
                  if r.get('exists') and isinstance(r.get('downloads'), int)]
        if popular:
            popular.sort(key=lambda x: x[1]['downloads'], reverse=True)
            print(f"\n{'='*80}")
            print("TOP 10 MOST DOWNLOADED MODELS")
            print(f"{'='*80}\n")
            for name, r in popular[:10]:
                print(f"  {r['hf_model_id']}")
                print(f"    Downloads: {r['downloads']:,}, Likes: {r.get('likes', 'N/A')}")
                print(f"    Undertrained tokens: {r['undertrained_token_count']}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check if models from undertrained tokens dictionary still exist on HuggingFace"
    )
    parser.add_argument(
        '--dictionary',
        type=str,
        default='undertrained_tokens_dictionary.json',
        help='Path to undertrained tokens dictionary'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to check. If not provided, checks all.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_existence_check.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    tester = SimpleTokenTester(args.dictionary)
    results = tester.test_models_existence(
        models_to_test=args.models,
        output_file=args.output
    )
    
    print(f"\n✓ Check complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
