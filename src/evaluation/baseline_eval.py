import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

DATASET_PATH = "data/processed/customer_support_sft"

OUTPUT_DIR = Path("results/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start small because we are running inference locally
NUM_EXAMPLES = 25

MAX_NEW_TOKENS = 256

SEED = 42


# ============================================================
# Header
# ============================================================

print("=" * 70)
print("NexaTune AI - Baseline Evaluation")
print("=" * 70)

print(f"\nModel        : {MODEL_NAME}")
print(f"Dataset      : {DATASET_PATH}")
print(f"Examples     : {NUM_EXAMPLES}")
print(f"Max new tok. : {MAX_NEW_TOKENS}")


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
    print(f"\nGPU          : {torch.cuda.get_device_name(0)}")
    print(f"CUDA         : {torch.version.cuda}")
else:
    device = "cpu"
    print("\nGPU          : Not available")

print(f"Device       : {device}")


# ============================================================
# Load tokenizer
# ============================================================

print("\n" + "-" * 70)
print("Loading tokenizer")
print("-" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# ============================================================
# Load 4-bit model
# ============================================================

print("\n" + "-" * 70)
print("Loading model")
print("-" * 70)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Load test dataset
# ============================================================

print("\n" + "-" * 70)
print("Loading test dataset")
print("-" * 70)

dataset = load_from_disk(DATASET_PATH)

test_dataset = dataset["test"]

print(f"Total test examples: {len(test_dataset)}")

# Reproducible sample
test_sample = test_dataset.shuffle(seed=SEED).select(
    range(min(NUM_EXAMPLES, len(test_dataset)))
)

print(f"Evaluation examples: {len(test_sample)}")


# ============================================================
# Generation function
# ============================================================

def generate_response(instruction):

    prompt = (
        "Customer:\n"
        f"{instruction}\n\n"
        "Assistant:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    start_time = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start_time

    # Only decode newly generated tokens
    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return response, elapsed


# ============================================================
# Run evaluation
# ============================================================

print("\n" + "-" * 70)
print("Running baseline evaluation")
print("-" * 70)

results = []

total_time = 0.0

for index, example in enumerate(test_sample):

    print(
        f"\n[{index + 1}/{len(test_sample)}] "
        f"Intent: {example['intent']}"
    )

    generated_response, elapsed = generate_response(
        example["instruction"]
    )

    total_time += elapsed

    result = {
        "index": index,
        "instruction": example["instruction"],
        "category": example["category"],
        "intent": example["intent"],
        "reference_response": example["response"],
        "generated_response": generated_response,
        "latency_seconds": round(elapsed, 4),
    }

    results.append(result)

    print(f"Latency: {elapsed:.2f}s")
    print(f"Generated: {generated_response[:300]}")


# ============================================================
# Save predictions
# ============================================================

predictions_path = OUTPUT_DIR / "predictions.json"

with open(predictions_path, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Calculate basic statistics
# ============================================================

average_latency = total_time / len(results)

generated_token_counts = []

for result in results:

    tokens = tokenizer(
        result["generated_response"],
        add_special_tokens=False,
    )

    generated_token_counts.append(
        len(tokens["input_ids"])
    )

average_generated_tokens = (
    sum(generated_token_counts)
    / len(generated_token_counts)
)


metrics = {
    "model": MODEL_NAME,
    "dataset": DATASET_PATH,
    "num_examples": len(results),
    "average_latency_seconds": round(
        average_latency,
        4,
    ),
    "total_inference_time_seconds": round(
        total_time,
        4,
    ),
    "average_generated_tokens": round(
        average_generated_tokens,
        2,
    ),
}


# ============================================================
# Save metrics
# ============================================================

metrics_path = OUTPUT_DIR / "metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(
        metrics,
        file,
        indent=2,
    )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("BASELINE EVALUATION COMPLETE")
print("=" * 70)

print(f"\nExamples evaluated      : {len(results)}")
print(
    f"Average latency         : "
    f"{average_latency:.4f} seconds"
)
print(
    f"Average generated tokens: "
    f"{average_generated_tokens:.2f}"
)

print(f"\nPredictions saved to:")
print(predictions_path)

print("\nMetrics saved to:")
print(metrics_path)

print("\n" + "=" * 70)