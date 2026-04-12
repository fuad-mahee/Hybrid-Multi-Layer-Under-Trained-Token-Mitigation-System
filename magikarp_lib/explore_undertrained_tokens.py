"""
Helper script to explore the undertrained tokens dictionary.
Usage examples provided in the main function.
"""

import json

def load_data():
    """Load all undertrained token data files."""
    with open('undertrained_tokens_dictionary.json', 'r', encoding='utf-8') as f:
        all_tokens = json.load(f)
    
    with open('undertrained_tokens_unique.json', 'r', encoding='utf-8') as f:
        unique_tokens = json.load(f)
    
    with open('undertrained_tokens_frequency.json', 'r', encoding='utf-8') as f:
        frequency_data = json.load(f)
    
    return all_tokens, unique_tokens, frequency_data


def get_tokens_for_model(all_tokens, model_name):
    """Get all undertrained tokens for a specific model."""
    return all_tokens.get(model_name, [])


def get_models_with_token(frequency_data, token):
    """Get list of models that have a specific undertrained token."""
    if token in frequency_data:
        return frequency_data[token]['models']
    return []


def get_most_common_tokens(frequency_data, top_n=20):
    """Get the most common undertrained tokens across models."""
    sorted_tokens = sorted(
        frequency_data.items(), 
        key=lambda x: x[1]['count'], 
        reverse=True
    )
    return [(token, data['count'], data['models']) for token, data in sorted_tokens[:top_n]]


def search_tokens_by_pattern(all_tokens, pattern):
    """Search for tokens containing a specific pattern across all models."""
    results = {}
    for model_name, tokens in all_tokens.items():
        matching = [t for t in tokens if pattern.lower() in t['raw_vocab'].lower()]
        if matching:
            results[model_name] = matching
    return results


def get_statistics():
    """Get overall statistics about the dataset."""
    all_tokens, unique_tokens, frequency_data = load_data()
    
    total_models = len(all_tokens)
    total_unique = len(unique_tokens)
    total_instances = sum(len(tokens) for tokens in all_tokens.values())
    
    # Model with most tokens
    max_model = max(all_tokens.items(), key=lambda x: len(x[1]))
    min_model = min(all_tokens.items(), key=lambda x: len(x[1]))
    
    return {
        'total_models': total_models,
        'total_unique_tokens': total_unique,
        'total_instances': total_instances,
        'model_with_most': (max_model[0], len(max_model[1])),
        'model_with_least': (min_model[0], len(min_model[1])),
        'avg_tokens_per_model': total_instances / total_models
    }


def main():
    """Example usage of the helper functions."""
    print("Loading data...")
    all_tokens, unique_tokens, frequency_data = load_data()
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    stats = get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Get tokens for GPT-2")
    print("="*80)
    gpt2_tokens = get_tokens_for_model(all_tokens, 'openai_community_gpt2')
    print(f"GPT-2 has {len(gpt2_tokens)} undertrained tokens")
    print("\nFirst 5 tokens:")
    for token in gpt2_tokens[:5]:
        print(f"  ID: {token['token_id']}, Token: {token['raw_vocab']}, Max Prob: {token['max_prob']:.2e}")
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Find which models have the token 'webElementX'")
    print("="*80)
    models = get_models_with_token(frequency_data, 'webElementX')
    print(f"Token 'webElementX' appears in {len(models)} models:")
    for model in models[:10]:  # Show first 10
        print(f"  - {model}")
    if len(models) > 10:
        print(f"  ... and {len(models) - 10} more")
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Most common tokens (top 10)")
    print("="*80)
    top_tokens = get_most_common_tokens(frequency_data, top_n=10)
    for i, (token, count, models) in enumerate(top_tokens, 1):
        print(f"{i}. '{token}' - {count} models")
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Search for tokens containing 'Code'")
    print("="*80)
    results = search_tokens_by_pattern(all_tokens, 'Code')
    print(f"Found tokens with 'Code' in {len(results)} models")
    total = sum(len(tokens) for tokens in results.values())
    print(f"Total matching tokens: {total}")
    
    # Show a few examples
    print("\nExample matches from first 3 models:")
    for model_name in list(results.keys())[:3]:
        print(f"\n{model_name}:")
        for token in results[model_name][:3]:
            print(f"  - {token['raw_vocab']}")
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Get all models analyzed")
    print("="*80)
    print(f"Total models: {len(all_tokens)}")
    print("\nAll model names:")
    for model in sorted(all_tokens.keys()):
        print(f"  - {model}")


if __name__ == "__main__":
    main()
