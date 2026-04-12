import argparse
import os
from typing import List

# Import magikarp from the original codebase installed editable
try:
    import magikarp  # type: ignore
except ImportError as e:
    raise SystemExit(
        "Magikarp package not found. Ensure you ran 'pip install -e <path-to-magikarp-main>'."
    ) from e

# Depending on Magikarp's API, we try to mirror the original testing entrypoints.
# As we don't modify the original, we call its public functions. If names differ, adjust accordingly.


def run_tests(models: List[str], compare_baselines: bool = False, output_dir: str = "results") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Placeholder orchestration: adapt to Magikarp's API
    # Common pattern: for each model, run undertrained token detection and save outputs
    for model_id in models:
        print(f"Running Magikarp undertrained-token test for: {model_id}")
        # Hypothetical API based on typical structure; replace with actual magikarp calls
        # Example: magikarp.run_test(model_id=model_id, output_path=f"{output_dir}/{safe_name}.json")
        safe_name = (
            model_id.replace("/", "-").replace(":", "-").replace(" ", "-")
        )
        try:
            # Attempt to discover a likely entry point
            if hasattr(magikarp, "cli_main"):
                magikarp.cli_main(["--model", model_id, "--output", os.path.join(output_dir, f"{safe_name}.json")])
            elif hasattr(magikarp, "run"):
                magikarp.run(model_id=model_id, output_dir=output_dir)
            else:
                print(
                    "Magikarp entrypoint not found. Please update run_magikarp_tests.py to use the correct API."
                )
        except Exception as ex:
            print(f"Error running test for {model_id}: {ex}")

    if compare_baselines:
        baselines = [
            "meta-llama/Llama-3.1-8B-Instruct",  # baseline for Llama 3.2/3.3
            "Qwen/Qwen2.5-7B-Instruct",          # baseline for Qwen2.5-Turbo-7B
            "deepseek-ai/DeepSeek-V2-Lite",      # baseline for DeepSeek-V3
            "google/gemma-7b-it",                 # baseline for gemma-2-9b-it
        ]
        print("\nRunning baseline comparisons...")
        for base_id in baselines:
            safe_name = base_id.replace("/", "-")
            try:
                if hasattr(magikarp, "cli_main"):
                    magikarp.cli_main(["--model", base_id, "--output", os.path.join(output_dir, f"{safe_name}.json")])
                elif hasattr(magikarp, "run"):
                    magikarp.run(model_id=base_id, output_dir=output_dir)
                else:
                    print("Magikarp entrypoint not found for baselines.")
            except Exception as ex:
                print(f"Error running baseline {base_id}: {ex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Magikarp undertrained-token tests on specified models")
    parser.add_argument("--models", nargs="+", required=True, help="List of model IDs to test")
    parser.add_argument("--compare-baselines", action="store_true", help="Also run baseline models from the original study")
    parser.add_argument("--output-dir", default="results", help="Directory to store results")
    args = parser.parse_args()

    run_tests(args.models, compare_baselines=args.compare_baselines, output_dir=args.output_dir)
