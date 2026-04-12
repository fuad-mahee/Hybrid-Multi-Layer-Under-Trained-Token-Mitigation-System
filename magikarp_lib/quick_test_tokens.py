"""
Quick test script - tests a few models to verify the retest functionality works.
This is faster than testing all 81 models.
"""

from test_undertrained_tokens_latest import UndertrainedTokenTester

# Initialize tester
tester = UndertrainedTokenTester()

# Test just a few popular models
models_to_test = [
    'openai_community_gpt2',           # GPT-2
    'meta_llama_Llama_2_7b_hf',        # Llama 2 7B
    'mistralai_Mistral_7B_v0_1',       # Mistral 7B
]

print("\n" + "="*80)
print("QUICK TEST - Testing 3 Models")
print("="*80)
print("\nThis will check if undertrained tokens still exist in the latest tokenizers.")
print("Testing models:")
for m in models_to_test:
    print(f"  - {m}")
print()

# Run the test
results = tester.test_all_models(
    models_to_test=models_to_test,
    check_existence_only=True,
    output_file="quick_test_results.json"
)

print("\n" + "="*80)
print("Quick test complete! Check 'quick_test_results.json' for detailed results.")
print("="*80)
