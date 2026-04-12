# Magikarp Further Testing — HMUTTMS

Independent project implementing the **Hybrid Multi-Layer Under-Trained Token Mitigation System (HMUTTMS)** on top of the Magikarp undertrained-token detection framework.

## Project Structure

```
magikarp-further-testing/
├── magikarp_lib/          # Magikarp library (modified, local copy)
│   ├── magikarp/          # Core Python package
│   ├── pyproject.toml
│   └── *.py               # Detection and analysis scripts
├── data/                  # Pre-collected token data
│   ├── undertrained_tokens_dictionary.json   # 39,272 tokens from 81 LLMs
│   ├── undertrained_tokens_unique.json
│   ├── undertrained_tokens_frequency.json
│   ├── kimi_k2_magikarp_results.json         # Model B cross-validation data
│   └── magikarp_comparison.csv               # Model A (GPT-2) results
├── src/
│   └── run_magikarp_tests.py   # Test runner / HMUTTMS orchestration
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Prerequisites

- Windows 10/11
- Python 3.11 or 3.12
- Enough disk space to download models (if running live detection)

## Setup (PowerShell)

```powershell
# Navigate to the project
Set-Location "magikarp-further-testing"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install CPU-only Torch (replace with CUDA wheel if you have a GPU)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# Install base dependencies
pip install -r requirements.txt

# Install the local Magikarp library (editable)
pip install -e "./magikarp_lib"
```

## Run tests

```powershell
python .\src\run_magikarp_tests.py --models meta-llama/Llama-3.2-3B-Instruct Qwen/Qwen2.5-Turbo-7B --compare-baselines
```

## Data

The `data/` folder contains pre-collected undertrained token data from 81 LLMs — no re-running needed to start building the HMUTTMS layers.

| File | Description |
|------|-------------|
| `undertrained_tokens_dictionary.json` | Full token dictionary organised by model |
| `undertrained_tokens_unique.json` | 39,272 unique undertrained tokens |
| `undertrained_tokens_frequency.json` | How many models each token appears in |
| `kimi_k2_magikarp_results.json` | Cross-validation results using Kimi K2 as Model B |
| `magikarp_comparison.csv` | Detection results from custom-trained GPT-2 (Model A) |

## Model A (GPT-2) Checkpoints

The custom-trained GPT-2 model checkpoints are **not included** in this repository due to size. They are available on HuggingFace Hub (see project documentation) or can be re-trained using the scripts in `gpt2-engine-main/`.

## Notes

- If you have a GPU and proper CUDA installed, replace the Torch install line with the appropriate CUDA wheel from the [PyTorch website](https://pytorch.org).
- Some models may require a HuggingFace access token (e.g. Meta Llama). Set `HUGGINGFACE_TOKEN` in your environment if needed.
