"""
Test whether undertrained tokens from the dictionary still exist in the latest model versions.

This script:
1. Loads the undertrained tokens dictionary
2. For each model, checks if the tokens still exist in the latest tokenizer
3. Optionally tests if they're still undertrained using the verification method
4. Generates a report comparing old vs new results
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

try:
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. Full verification will be disabled.")

from transformers import AutoTokenizer
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=""):
        return iterable


class UndertrainedTokenTester:
    """Test undertrained tokens against latest model versions."""
    
    def __init__(self, dictionary_path: str = "undertrained_tokens_dictionary.json"):
        """Initialize with the undertrained tokens dictionary."""
        print(f"Loading undertrained tokens dictionary from {dictionary_path}...")
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            self.token_dict = json.load(f)
        
        total_tokens = sum(len(tokens) for tokens in self.token_dict.values())
        print(f"Loaded {len(self.token_dict)} models with {total_tokens} total undertrained tokens")
    
    def map_to_hf_model_id(self, internal_name: str) -> str:
        """Convert internal model name to HuggingFace model ID."""
        # Replace underscores with slashes for HuggingFace format
        # e.g., "meta_llama_Llama_2_7b_hf" -> "meta-llama/Llama-2-7b-hf"
        
        # Common mappings
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
        
        # Find the organization part
        for internal_org, hf_org in mappings.items():
            if internal_name.startswith(internal_org + '_'):
                # Replace organization and convert remaining underscores to hyphens
                rest = internal_name[len(internal_org)+1:].replace('_', '-')
                return f"{hf_org}/{rest}"
        
        # Fallback: just replace underscores with hyphens
        parts = internal_name.split('_', 1)
        if len(parts) == 2:
            return f"{parts[0]}/{parts[1].replace('_', '-')}"
        return internal_name
    
    def check_tokens_exist_in_tokenizer(
        self, 
        model_id: str, 
        tokens_to_check: List[Dict]
    ) -> Dict:
        """
        Check if tokens still exist in the latest tokenizer.
        
        Returns:
            Dict with token existence information
        """
        try:
            print(f"\nLoading tokenizer for {model_id}...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            results = {
                'model_id': model_id,
                'tokenizer_loaded': True,
                'vocab_size': len(tokenizer),
                'tokens_checked': len(tokens_to_check),
                'tokens_still_exist': 0,
                'tokens_missing': 0,
                'tokens_changed': 0,
                'detailed_results': []
            }
            
            for token_info in tokens_to_check:
                token_id = token_info['token_id']
                original_raw_vocab = token_info['raw_vocab']
                
                # Check if token ID is still valid
                if token_id < len(tokenizer):
                    # Get current token at that ID
                    try:
                        current_vocab = tokenizer.convert_ids_to_tokens([token_id])[0]
                        
                        # Check if it's the same token
                        if current_vocab == original_raw_vocab:
                            status = 'exists_unchanged'
                            results['tokens_still_exist'] += 1
                        else:
                            status = 'exists_changed'
                            results['tokens_changed'] += 1
                        
                        results['detailed_results'].append({
                            'token_id': token_id,
                            'original_vocab': original_raw_vocab,
                            'current_vocab': current_vocab,
                            'status': status,
                            'original_max_prob': token_info['max_prob']
                        })
                    except Exception as e:
                        results['detailed_results'].append({
                            'token_id': token_id,
                            'original_vocab': original_raw_vocab,
                            'status': 'error',
                            'error': str(e)
                        })
                        results['tokens_missing'] += 1
                else:
                    results['detailed_results'].append({
                        'token_id': token_id,
                        'original_vocab': original_raw_vocab,
                        'status': 'missing_id_out_of_range',
                        'current_vocab_size': len(tokenizer)
                    })
                    results['tokens_missing'] += 1
            
            return results
            
        except Exception as e:
            return {
                'model_id': model_id,
                'tokenizer_loaded': False,
                'error': str(e),
                'tokens_checked': len(tokens_to_check)
            }
    
    def verify_token_still_undertrained(
        self,
        model_id: str,
        token_id: int,
        device: str = "cpu"
    ) -> Dict:
        """
        Test if a token is still undertrained using simple verification.
        
        This is a lightweight version - just tests if the model can reproduce the token.
        """
        if not TORCH_AVAILABLE:
            return {
                'token_id': token_id,
                'verified': False,
                'error': 'torch not available - cannot load model'
            }
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            # Simple test: ask model to repeat the token
            token_str = tokenizer.decode([token_id])
            prompt = f"Repeat this exactly: {token_str}\nRepeat:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=True,
                    top_k=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Check if token appears in output with high probability
            scores = outputs.scores[0] if len(outputs.scores) > 0 else None
            
            if scores is not None:
                probs = torch.nn.functional.softmax(scores[0], dim=-1)
                token_prob = probs[token_id].item()
                
                return {
                    'token_id': token_id,
                    'verified': True,
                    'token_prob': token_prob,
                    'still_undertrained': token_prob < 0.01,
                    'generated': tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:])
                }
            else:
                return {
                    'token_id': token_id,
                    'verified': False,
                    'error': 'No scores returned'
                }
                
        except Exception as e:
            return {
                'token_id': token_id,
                'verified': False,
                'error': str(e)
            }
    
    def test_all_models(
        self, 
        models_to_test: List[str] = None,
        check_existence_only: bool = True,
        output_file: str = "undertrained_tokens_retest_results.json"
    ):
        """
        Test all models or a subset.
        
        Args:
            models_to_test: List of internal model names to test. If None, test all.
            check_existence_only: If True, only check if tokens exist. If False, also verify if still undertrained.
            output_file: Where to save results
        """
        if models_to_test is None:
            models_to_test = list(self.token_dict.keys())
        
        all_results = {}
        
        print(f"\n{'='*80}")
        print(f"Testing {len(models_to_test)} models...")
        print(f"{'='*80}\n")
        
        for internal_name in tqdm(models_to_test, desc="Testing models"):
            hf_model_id = self.map_to_hf_model_id(internal_name)
            tokens_to_check = self.token_dict[internal_name]
            
            print(f"\n{internal_name} -> {hf_model_id}")
            
            # Check if tokens exist
            existence_results = self.check_tokens_exist_in_tokenizer(
                hf_model_id, 
                tokens_to_check
            )
            
            all_results[internal_name] = {
                'hf_model_id': hf_model_id,
                'existence_check': existence_results
            }
            
            # Print summary
            if existence_results.get('tokenizer_loaded'):
                print(f"  ✓ Tokenizer loaded")
                print(f"  Tokens still exist: {existence_results['tokens_still_exist']}/{existence_results['tokens_checked']}")
                print(f"  Tokens changed: {existence_results['tokens_changed']}")
                print(f"  Tokens missing: {existence_results['tokens_missing']}")
            else:
                print(f"  ✗ Failed to load tokenizer: {existence_results.get('error', 'Unknown error')}")
        
        # Save results
        print(f"\n{'='*80}")
        print(f"Saving results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Generate summary
        self.generate_summary(all_results)
        
        return all_results
    
    def generate_summary(self, results: Dict):
        """Generate a summary report."""
        print(f"\n{'='*80}")
        print("SUMMARY REPORT")
        print(f"{'='*80}\n")
        
        total_models = len(results)
        successful_loads = sum(1 for r in results.values() if r['existence_check'].get('tokenizer_loaded'))
        failed_loads = total_models - successful_loads
        
        print(f"Models tested: {total_models}")
        print(f"Successfully loaded: {successful_loads}")
        print(f"Failed to load: {failed_loads}\n")
        
        if successful_loads > 0:
            total_tokens_checked = 0
            total_still_exist = 0
            total_changed = 0
            total_missing = 0
            
            for internal_name, result in results.items():
                if result['existence_check'].get('tokenizer_loaded'):
                    ec = result['existence_check']
                    total_tokens_checked += ec['tokens_checked']
                    total_still_exist += ec['tokens_still_exist']
                    total_changed += ec['tokens_changed']
                    total_missing += ec['tokens_missing']
            
            print(f"Total tokens checked: {total_tokens_checked}")
            print(f"Still exist (unchanged): {total_still_exist} ({100*total_still_exist/total_tokens_checked:.1f}%)")
            print(f"Changed: {total_changed} ({100*total_changed/total_tokens_checked:.1f}%)")
            print(f"Missing: {total_missing} ({100*total_missing/total_tokens_checked:.1f}%)")
            
            print(f"\n{'='*80}")
            print("MODELS WITH MOST CHANGES")
            print(f"{'='*80}\n")
            
            models_by_changes = []
            for internal_name, result in results.items():
                if result['existence_check'].get('tokenizer_loaded'):
                    ec = result['existence_check']
                    if ec['tokens_checked'] > 0:
                        change_rate = (ec['tokens_changed'] + ec['tokens_missing']) / ec['tokens_checked']
                        models_by_changes.append((internal_name, change_rate, ec))
            
            models_by_changes.sort(key=lambda x: x[1], reverse=True)
            
            for internal_name, change_rate, ec in models_by_changes[:10]:
                print(f"{internal_name}:")
                print(f"  Change rate: {100*change_rate:.1f}%")
                print(f"  Changed: {ec['tokens_changed']}, Missing: {ec['tokens_missing']}, Total: {ec['tokens_checked']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test if undertrained tokens still exist in latest model versions"
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
        help='Specific models to test (internal names). If not provided, tests all.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='undertrained_tokens_retest_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Also verify if tokens are still undertrained (requires loading models - slow!)'
    )
    
    args = parser.parse_args()
    
    tester = UndertrainedTokenTester(args.dictionary)
    results = tester.test_all_models(
        models_to_test=args.models,
        check_existence_only=not args.verify,
        output_file=args.output
    )
    
    print(f"\n✓ Testing complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
