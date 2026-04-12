"""
Quick analysis of a new model's tokenizer without full verification.
This shows potential undertrained token candidates based on tokenizer structure.
"""

import argparse
import json
from collections import defaultdict


def analyze_model_tokenizer(model_id: str, output_file: str = None):
    """Analyze a model's tokenizer for potential undertrained tokens."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing tokenizer for: {model_id}")
    print(f"{'='*80}\n")
    
    try:
        # Try using tokenizers library (faster, no torch needed)
        from tokenizers import Tokenizer
        print("Attempting to load with tokenizers library...")
        try:
            tokenizer = Tokenizer.from_pretrained(model_id)
            vocab = tokenizer.get_vocab()
            print(f"✓ Loaded tokenizer using 'tokenizers' library")
        except:
            raise ImportError("Fall back to transformers")
            
    except (ImportError, Exception):
        # Fall back to transformers (requires torch)
        print("Loading with transformers library...")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            vocab = tokenizer.get_vocab()
            print(f"✓ Loaded tokenizer using 'transformers' library")
        except Exception as e:
            print(f"✗ Failed to load tokenizer: {e}")
            return None
    
    print(f"\nVocabulary size: {len(vocab):,}")
    
    # Analyze token patterns
    analysis = {
        'model_id': model_id,
        'vocab_size': len(vocab),
        'special_tokens': [],
        'byte_tokens': [],
        'unusual_tokens': [],
        'very_short_tokens': [],
        'unprintable_tokens': [],
        'statistics': {}
    }
    
    print("\nAnalyzing token patterns...")
    
    for token_str, token_id in vocab.items():
        # Check for special tokens
        if token_str.startswith('<') and token_str.endswith('>'):
            analysis['special_tokens'].append({
                'id': token_id,
                'token': token_str
            })
        
        # Check for byte tokens
        if token_str.startswith('<0x') and token_str.endswith('>'):
            analysis['byte_tokens'].append({
                'id': token_id,
                'token': token_str
            })
        
        # Check for very short tokens
        if len(token_str) == 1 and not token_str.isalnum():
            analysis['very_short_tokens'].append({
                'id': token_id,
                'token': token_str,
                'ord': ord(token_str) if len(token_str) == 1 else None
            })
        
        # Check for unprintable characters
        if any(not c.isprintable() and c not in [' ', '\n', '\t', '\r'] for c in token_str):
            analysis['unprintable_tokens'].append({
                'id': token_id,
                'token': repr(token_str)[1:-1]  # Remove quotes from repr
            })
        
        # Check for unusual patterns
        if any(pattern in token_str.lower() for pattern in ['unk', 'unused', 'mask', 'pad']):
            analysis['unusual_tokens'].append({
                'id': token_id,
                'token': token_str
            })
    
    # Statistics
    analysis['statistics'] = {
        'special_tokens_count': len(analysis['special_tokens']),
        'byte_tokens_count': len(analysis['byte_tokens']),
        'unusual_tokens_count': len(analysis['unusual_tokens']),
        'very_short_tokens_count': len(analysis['very_short_tokens']),
        'unprintable_tokens_count': len(analysis['unprintable_tokens'])
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print("ANALYSIS RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Total vocabulary size: {analysis['vocab_size']:,}")
    print(f"Special tokens: {analysis['statistics']['special_tokens_count']}")
    print(f"Byte tokens: {analysis['statistics']['byte_tokens_count']}")
    print(f"Unusual tokens: {analysis['statistics']['unusual_tokens_count']}")
    print(f"Very short tokens: {analysis['statistics']['very_short_tokens_count']}")
    print(f"Unprintable tokens: {analysis['statistics']['unprintable_tokens_count']}")
    
    if analysis['special_tokens']:
        print(f"\nSpecial tokens (first 20):")
        for token_info in analysis['special_tokens'][:20]:
            print(f"  [{token_info['id']}] {token_info['token']}")
    
    if analysis['unusual_tokens']:
        print(f"\nUnusual tokens:")
        for token_info in analysis['unusual_tokens']:
            print(f"  [{token_info['id']}] {token_info['token']}")
    
    if analysis['unprintable_tokens'][:10]:
        print(f"\nUnprintable tokens (first 10):")
        for token_info in analysis['unprintable_tokens'][:10]:
            print(f"  [{token_info['id']}] {token_info['token']}")
    
    # Save results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Full analysis saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("NOTE: This is a quick tokenizer analysis.")
    print("For full undertrained token detection, use:")
    print(f"  python -m magikarp.fishing {model_id}")
    print(f"{'='*80}\n")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Quick analysis of a model's tokenizer"
    )
    parser.add_argument(
        'model_id',
        type=str,
        help='HuggingFace model ID (e.g., meta-llama/Llama-3.2-3B-Instruct)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    analyze_model_tokenizer(args.model_id, args.output)


if __name__ == "__main__":
    main()
