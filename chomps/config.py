"""Paths, deployments and constants shared by every revision experiment.

Layout (research/llms-chomps):
    code/        this repository (public: github.com/sho-404/Chomps2025)
    datasets/    SHROOM-CAP splits, labels and source papers (not in the repo)
    results/     every artifact produced by the revision experiments
    cache/       LLM/embedding response cache and extracted paper text
    .env         AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY (never committed)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
for _d in (RESULTS, CACHE):
    _d.mkdir(exist_ok=True)

load_dotenv(ROOT / ".env")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
API_VERSION = "2025-04-01-preview"

LANGS = ("english", "spanish")
LANG_PREFIX = {"english": "en", "spanish": "es"}
SPLITS = ("train", "validation", "test")
SPLIT_FILE_TAG = {"train": "train", "validation": "valid", "test": "test"}
TRACKS = ("factuality", "fluency")
LABEL_KEY = {"factuality": "has_factual_mistakes", "fluency": "has_fluency_mistakes"}

# Base deployments on chomps-openai (North Central US). The model behind
# "gpt-5-mini" is gpt-5-mini-2025-08-07; GPT-4.1 mini/nano are 2025-04-14.
DEPLOYMENTS = {
    "gpt-5-mini": "gpt-5-mini",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "embedding": "text-embedding-3-large",
}
# Fine-tuned council members, keyed by (track, lang, base). Filled in as the
# jobs in code/finetuning/jobs.jsonl are deployed (see run_finetune.py deploy).
FT_DEPLOYMENTS = {
    ("factuality", "english", "gpt-4.1-mini"): "ft-fact-en-mini",
    ("factuality", "english", "gpt-4.1-nano"): "ft-fact-en-nano",
    ("factuality", "spanish", "gpt-4.1-mini"): "ft-fact-es-mini",
    ("factuality", "spanish", "gpt-4.1-nano"): "ft-fact-es-nano",
    ("fluency", "english", "gpt-4.1-mini"): "ft-flu-en-mini",
    ("fluency", "english", "gpt-4.1-nano"): "ft-flu-en-nano",
    ("fluency", "spanish", "gpt-4.1-mini"): "ft-flu-es-mini",
    ("fluency", "spanish", "gpt-4.1-nano"): "ft-flu-es-nano",
}

# Truncation limits inherited from the original scripts, kept so that the
# revision reproduces the competition protocol exactly.
MAX_PAPER_CHARS = 120_000   # retrieval stage: paper_text[:120000]
MAX_CONTEXT_CHARS = 60_000  # judge prompt: context[:60000]
JUDGE_MAX_COMPLETION_TOKENS = 16_384
RETRIEVAL_MAX_COMPLETION_TOKENS = 4_096

# Reasoning models (GPT-5 family) reject any temperature other than the
# default; the API returns HTTP 400 "Unsupported value". We therefore never
# send a temperature to them and document this in the paper.
REASONING_DEPLOYMENTS = {"gpt-5-mini"}

# Azure OpenAI list prices, USD per 1M tokens, Global Standard, October 2026
# (https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/).
# Update here if the price sheet changes; cost.py reads only this table.
PRICES_USD_PER_M = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "ft-gpt-4.1-mini": {"input": 0.80, "output": 3.20, "hosting_per_hour": 1.70},
    "ft-gpt-4.1-nano": {"input": 0.20, "output": 0.80, "hosting_per_hour": 1.70},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def price_key(deployment: str) -> str:
    if deployment.startswith("ft-"):
        return "ft-gpt-4.1-mini" if deployment.endswith("mini") else "ft-gpt-4.1-nano"
    return deployment
