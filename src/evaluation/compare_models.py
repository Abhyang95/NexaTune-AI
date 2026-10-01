from pathlib import Path
import json
import time

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

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

DATASET_PATH = Path(
    "data/processed/customer_support_sft"
)

ADAPTER_PATH = Path(
    "models/final_adapter"
)

OUTPUT_PATH = Path(
    "results/model_comparison.json"
)

NUM_SAMPLES = 25
MAX_NEW_TOKENS = 256


# ============================================================
# Quantization
# ============================================================

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load dataset
# ============================================================

print("Loading evaluation dataset...")

dataset = load_from_disk(str(DATASET_PATH))

test_dataset = dataset["test"]

test_dataset = (
    test_dataset
    .shuffle(seed=42)
    .select(range(min(NUM_SAMPLES, len(test_dataset))))
)

print(f"Evaluation samples: {len(test_dataset)}")


# ============================================================
# Model loader
# ============================================================

def load_base_model():

    print("\nLoading BASE model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model.eval()

    return model


# ============================================================
# Generation
# ============================================================

def generate_response(model, instruction):

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
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.perf_counter() - start_time

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return response, latency


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(model, model_name):

    print(f"\nEvaluating: {model_name}")

    predictions = []

    total_latency = 0.0

    for index, example in enumerate(test_dataset):

        instruction = example["instruction"]

        reference = example["response"]

        prediction, latency = generate_response(
            model,
            instruction,
        )

        total_latency += latency

        predictions.append(
            {
                "instruction": instruction,
                "reference": reference,
                "prediction": prediction,
                "latency_seconds": latency,
            }
        )

        print(
            f"[{index + 1}/{len(test_dataset)}] "
            f"{latency:.2f}s"
        )

    average_latency = (
        total_latency / len(predictions)
    )

    return {
        "model": model_name,
        "samples": len(predictions),
        "average_latency_seconds": average_latency,
        "predictions": predictions,
    }


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Base model
    # ---------------------------------------------

    base_model = load_base_model()

    base_results = evaluate_model(
        base_model,
        "Qwen3-1.7B-Base",
    )

    del base_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------
    # Fine-tuned adapter
    # ---------------------------------------------

    if not ADAPTER_PATH.exists():

        print(
            "\n⚠️ Fine-tuned adapter not found:"
        )

        print(
            ADAPTER_PATH
        )

        print(
            "\nBase-model evaluation completed."
        )

        results = {
            "base_model": base_results,
            "fine_tuned_model": None,
            "status": "waiting_for_fine_tuned_adapter",
        }

    else:

        print(
            "\nFine-tuned adapter found."
        )

        print(
            "Fine-tuned evaluation will be added "
            "after adapter loading is verified."
        )

        results = {
            "base_model": base_results,
            "fine_tuned_model": None,
            "status": "adapter_found",
        }

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()