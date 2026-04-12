"""
Extract all undertrained tokens detected across all models.
This script creates a dictionary of all undertrained tokens found in the Magikarp paper.
"""

import gzip
import json
import os
from pathlib import Path
from collections import defaultdict

# Threshold for classifying tokens as undertrained (max_prob < 0.01)
VERIFICATION_THRESHOLD = 0.01

def load_verification_file(filepath):
    """Load a verification JSONL file (gzipped or not)."""
    try:
        # Convert Path to string
        filepath_str = str(filepath)
        
        # Try gzipped first
        if filepath_str.endswith('.gz'):
            with gzip.open(filepath_str, 'rt') as f:
                return [json.loads(line) for line in f]
        else:
            with open(filepath_str, 'r') as f:
                return [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_undertrained_tokens_from_model(verification_data, model_name):
    """
    Extract undertrained tokens from verification data.
    Returns tokens that have been verified as undertrained (max_prob < threshold).
    """
    undertrained = []
    
    for token_info in verification_data:
        # Check if token has verification results
        if 'verification' not in token_info:
            continue
        
        # Check if it's classified as undertrained
        magikarp_status = token_info.get('magikarp', '')
        max_prob = token_info.get('max_prob', 1.0)
        
        # Token is undertrained if:
        # 1. It's verified as strong or weak verified
        # 2. OR max_prob is below threshold
        if 'verified' in magikarp_status or max_prob < VERIFICATION_THRESHOLD:
            token_id = token_info.get('i')
            decoded = token_info.get('decoded', '')
            raw_vocab = token_info.get('raw_vocab', '')
            
            undertrained.append({
                'token_id': token_id,
                'decoded': decoded,
                'raw_vocab': raw_vocab,
                'max_prob': max_prob,
                'magikarp_status': magikarp_status,
                'model': model_name
            })
    
    return undertrained

def main():
    # Path to verifications folder
    verifications_dir = Path("results/verifications")
    
    if not verifications_dir.exists():
        print(f"Error: {verifications_dir} does not exist!")
        return
    
    # Dictionary to store all undertrained tokens by model
    all_undertrained_tokens = {}
    
    # Set to collect unique tokens across all models
    unique_tokens_set = set()
    
    # Process each verification file
    verification_files = sorted(verifications_dir.glob("*.jsonl.gz"))
    
    print(f"Processing {len(verification_files)} model verification files...")
    print()
    
    for filepath in verification_files:
        # Extract model name from filename
        model_name = filepath.stem.replace('.jsonl', '')
        
        # Load verification data
        verification_data = load_verification_file(filepath)
        
        if verification_data is None:
            continue
        
        # Extract undertrained tokens
        undertrained = extract_undertrained_tokens_from_model(verification_data, model_name)
        
        if undertrained:
            all_undertrained_tokens[model_name] = undertrained
            
            # Add to unique set
            for token in undertrained:
                unique_tokens_set.add(token['raw_vocab'])
            
            print(f"{model_name}: {len(undertrained)} undertrained tokens")
    
    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models analyzed: {len(all_undertrained_tokens)}")
    print(f"Total unique undertrained tokens (by raw_vocab): {len(unique_tokens_set)}")
    
    total_undertrained = sum(len(tokens) for tokens in all_undertrained_tokens.values())
    print(f"Total undertrained token instances across all models: {total_undertrained}")
    
    # Save to JSON file
    output_file = "undertrained_tokens_dictionary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_undertrained_tokens, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"Dictionary saved to: {output_file}")
    
    # Create a simplified version with just unique tokens
    unique_tokens_list = sorted(list(unique_tokens_set))
    simplified_output = "undertrained_tokens_unique.json"
    with open(simplified_output, 'w', encoding='utf-8') as f:
        json.dump(unique_tokens_list, f, indent=2, ensure_ascii=False)
    
    print(f"Unique tokens list saved to: {simplified_output}")
    
    # Create a frequency count of tokens across models
    token_frequency = defaultdict(list)
    for model_name, tokens in all_undertrained_tokens.items():
        for token in tokens:
            token_frequency[token['raw_vocab']].append(model_name)
    
    # Sort by frequency
    sorted_by_frequency = sorted(
        token_frequency.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )
    
    print()
    print("=" * 80)
    print("TOP 20 MOST COMMON UNDERTRAINED TOKENS (across models)")
    print("=" * 80)
    for i, (token, models) in enumerate(sorted_by_frequency[:20], 1):
        print(f"{i}. '{token}' - found in {len(models)} models")
    
    # Save frequency data
    frequency_output = "undertrained_tokens_frequency.json"
    frequency_data = {
        token: {
            'count': len(models),
            'models': models
        }
        for token, models in sorted_by_frequency
    }
    with open(frequency_output, 'w', encoding='utf-8') as f:
        json.dump(frequency_data, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"Frequency data saved to: {frequency_output}")
    
    return all_undertrained_tokens

if __name__ == "__main__":
    result = main()
