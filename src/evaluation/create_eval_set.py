import json
import random
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE_PATH = Path("results/benchmark/base_predictions.json")
FINETUNED_PATH = Path("results/benchmark/finetuned_predictions.json")
OUTPUT_PATH = Path("results/benchmark/human_eval_set.json")

SEED = 123
N_EXAMPLES = 30


# ============================================================
# Load benchmark predictions
# ============================================================

with open(BASE_PATH, "r", encoding="utf-8") as f:
    base = json.load(f)["predictions"]

with open(FINETUNED_PATH, "r", encoding="utf-8") as f:
    finetuned = json.load(f)["predictions"]


# ============================================================
# Validate prediction alignment
# ============================================================

if len(base) != len(finetuned):
    raise ValueError("Base and NexaTune prediction counts do not match.")

for i, (b, f) in enumerate(zip(base, finetuned)):
    if b["instruction"] != f["instruction"]:
        raise ValueError(f"Example alignment mismatch at index {i}")


# ============================================================
# Deterministic selection from the existing benchmark
# ============================================================

rng = random.Random(SEED)
indices = sorted(rng.sample(range(len(base)), N_EXAMPLES))


# ============================================================
# Build evaluation set
# ============================================================

examples = []

for eval_id, index in enumerate(indices, start=1):
    examples.append({
        "eval_id": eval_id,
        "benchmark_index": index,
        "instruction": base[index]["instruction"],
        "reference": base[index]["reference"],
        "base_prediction": base[index]["prediction"],
        "finetuned_prediction": finetuned[index]["prediction"],
    })


# ============================================================
# Evaluation metadata
# ============================================================

result = {
    "evaluation_type": "AI-assisted LLM-as-judge evaluation",
    "selection_method": "Deterministic random sample from existing 100-example benchmark",
    "selection_seed": SEED,
    "n_examples": N_EXAMPLES,
    "examples": examples,
}


# ============================================================
# Save evaluation set
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)


# ============================================================
# Summary
# ============================================================

print("Evaluation set created successfully.")
print(f"Examples: {N_EXAMPLES}")
print(f"Seed: {SEED}")
print(f"Evaluation type: {result['evaluation_type']}")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Selected benchmark indices: {indices}")