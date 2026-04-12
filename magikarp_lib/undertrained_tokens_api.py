"""
Simple API to access the undertrained tokens dictionary.
This provides a clean interface to query and explore the undertrained tokens data.
"""

import json
from typing import List, Dict, Optional, Tuple


class UndertrainedTokensDB:
    """Database of undertrained tokens across LLMs."""
    
    def __init__(self):
        """Load the undertrained tokens data."""
        print("Loading undertrained tokens database...")
        
        with open('undertrained_tokens_dictionary.json', 'r', encoding='utf-8') as f:
            self.all_tokens = json.load(f)
        
        with open('undertrained_tokens_unique.json', 'r', encoding='utf-8') as f:
            self.unique_tokens = json.load(f)
        
        with open('undertrained_tokens_frequency.json', 'r', encoding='utf-8') as f:
            self.frequency_data = json.load(f)
        
        print(f"Loaded {len(self.all_tokens)} models with {len(self.unique_tokens)} unique tokens")
    
    def get_all_tokens_as_dict(self) -> Dict:
        """
        Get the complete dictionary of all undertrained tokens.
        
        Returns:
            Dictionary with model names as keys and lists of token info as values
        """
        return self.all_tokens
    
    def get_unique_tokens_list(self) -> List[str]:
        """
        Get list of all unique undertrained token strings.
        
        Returns:
            List of unique token strings
        """
        return self.unique_tokens
    
    def get_models(self) -> List[str]:
        """
        Get list of all analyzed models.
        
        Returns:
            List of model names
        """
        return list(self.all_tokens.keys())
    
    def get_tokens_for_model(self, model_name: str) -> List[Dict]:
        """
        Get all undertrained tokens for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'openai_community_gpt2')
        
        Returns:
            List of token dictionaries with keys: token_id, decoded, raw_vocab, 
            max_prob, magikarp_status, model
        """
        return self.all_tokens.get(model_name, [])
    
    def get_token_frequency(self, token: str) -> Tuple[int, List[str]]:
        """
        Get how many models have a specific token and which models.
        
        Args:
            token: The token string to search for
        
        Returns:
            Tuple of (count, list of model names)
        """
        if token in self.frequency_data:
            return (self.frequency_data[token]['count'], 
                   self.frequency_data[token]['models'])
        return (0, [])
    
    def get_most_common_tokens(self, top_n: int = 20) -> List[Tuple[str, int, List[str]]]:
        """
        Get the most common undertrained tokens across models.
        
        Args:
            top_n: Number of top tokens to return
        
        Returns:
            List of tuples (token, count, list of models)
        """
        sorted_tokens = sorted(
            self.frequency_data.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )
        return [(token, data['count'], data['models']) 
                for token, data in sorted_tokens[:top_n]]
    
    def search_tokens(self, pattern: str, case_sensitive: bool = False) -> Dict[str, List[Dict]]:
        """
        Search for tokens containing a specific pattern across all models.
        
        Args:
            pattern: String pattern to search for
            case_sensitive: Whether to do case-sensitive search
        
        Returns:
            Dictionary mapping model names to lists of matching tokens
        """
        results = {}
        pattern_to_search = pattern if case_sensitive else pattern.lower()
        
        for model_name, tokens in self.all_tokens.items():
            matching = []
            for t in tokens:
                token_str = t['raw_vocab'] if case_sensitive else t['raw_vocab'].lower()
                if pattern_to_search in token_str:
                    matching.append(t)
            if matching:
                results[model_name] = matching
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get overall statistics about the dataset.
        
        Returns:
            Dictionary with various statistics
        """
        total_instances = sum(len(tokens) for tokens in self.all_tokens.values())
        max_model = max(self.all_tokens.items(), key=lambda x: len(x[1]))
        min_model = min(self.all_tokens.items(), key=lambda x: len(x[1]))
        
        return {
            'total_models': len(self.all_tokens),
            'total_unique_tokens': len(self.unique_tokens),
            'total_token_instances': total_instances,
            'avg_tokens_per_model': total_instances / len(self.all_tokens),
            'model_with_most_tokens': max_model[0],
            'most_tokens_count': len(max_model[1]),
            'model_with_least_tokens': min_model[0],
            'least_tokens_count': len(min_model[1])
        }
    
    def filter_by_probability(self, max_prob_threshold: float) -> Dict[str, List[Dict]]:
        """
        Get tokens below a certain probability threshold across all models.
        
        Args:
            max_prob_threshold: Maximum probability threshold (e.g., 0.001 for very undertrained)
        
        Returns:
            Dictionary mapping model names to lists of tokens below threshold
        """
        results = {}
        for model_name, tokens in self.all_tokens.items():
            filtered = [t for t in tokens if t['max_prob'] < max_prob_threshold]
            if filtered:
                results[model_name] = filtered
        return results
    
    def get_cross_model_tokens(self, min_models: int = 5) -> List[Tuple[str, int, List[str]]]:
        """
        Get tokens that appear in multiple models.
        
        Args:
            min_models: Minimum number of models the token must appear in
        
        Returns:
            List of tuples (token, count, list of models)
        """
        cross_model = []
        for token, data in self.frequency_data.items():
            if data['count'] >= min_models:
                cross_model.append((token, data['count'], data['models']))
        return sorted(cross_model, key=lambda x: x[1], reverse=True)


# Example usage
if __name__ == "__main__":
    # Initialize the database
    db = UndertrainedTokensDB()
    
    # Get all tokens as a dictionary
    print("\n" + "="*80)
    print("DICTIONARY ACCESS EXAMPLES")
    print("="*80)
    
    # 1. Get the complete dictionary
    all_tokens_dict = db.get_all_tokens_as_dict()
    print(f"\n1. Complete dictionary has {len(all_tokens_dict)} models")
    
    # 2. Get tokens for a specific model
    gpt2_tokens = db.get_tokens_for_model('openai_community_gpt2')
    print(f"\n2. GPT-2 has {len(gpt2_tokens)} undertrained tokens")
    print(f"   First token: {gpt2_tokens[0]}")
    
    # 3. Get all unique tokens
    unique = db.get_unique_tokens_list()
    print(f"\n3. Total unique tokens: {len(unique)}")
    print(f"   First 5: {unique[:5]}")
    
    # 4. Get statistics
    stats = db.get_statistics()
    print(f"\n4. Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 5. Most common tokens
    print(f"\n5. Top 10 most common tokens:")
    for i, (token, count, models) in enumerate(db.get_most_common_tokens(10), 1):
        print(f"   {i}. '{token}' in {count} models")
    
    # 6. Search for specific pattern
    print(f"\n6. Searching for tokens with 'Unicode':")
    results = db.search_tokens('Unicode')
    print(f"   Found in {len(results)} models")
    
    # 7. Get cross-model tokens
    print(f"\n7. Tokens appearing in 10+ models:")
    cross_model = db.get_cross_model_tokens(min_models=10)
    print(f"   {len(cross_model)} tokens appear in 10+ models")
    
    # 8. Very undertrained tokens (max_prob < 0.001)
    very_undertrained = db.filter_by_probability(0.001)
    total_very = sum(len(tokens) for tokens in very_undertrained.values())
    print(f"\n8. Very undertrained tokens (max_prob < 0.001): {total_very} total")
    
    print("\n" + "="*80)
    print("Dictionary is ready to use! Access it via UndertrainedTokensDB()")
    print("="*80)
