from pathlib import Path
import gc
import json
import statistics
import time

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

from rouge_score import rouge_scorer
from bert_score import score as bert_score


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

OUTPUT_DIR = Path(
    "results/benchmark"
)

NUM_SAMPLES = 100
SEED = 42
MAX_NEW_TOKENS = 128


# ============================================================
# Quantization
# ============================================================

QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# Helpers
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_tokenizer():
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_test_dataset():
    print("\nLoading evaluation dataset...")

    dataset = load_from_disk(
        str(DATASET_PATH)
    )

    test_dataset = dataset["test"]

    test_dataset = (
        test_dataset
        .shuffle(seed=SEED)
        .select(
            range(
                min(
                    NUM_SAMPLES,
                    len(test_dataset),
                )
            )
        )
    )

    print(
        f"Evaluation samples: "
        f"{len(test_dataset)}"
    )

    return test_dataset


# ============================================================
# Model Loading
# ============================================================

def load_base_model():

    print("\nLoading BASE model...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="auto",
        dtype=torch.float16,
    )

    model.eval()

    return model


def load_finetuned_model():

    print("\nLoading NEXATUNE model...")

    if not ADAPTER_PATH.exists():

        raise FileNotFoundError(
            f"\nNexaTune adapter not found:\n"
            f"{ADAPTER_PATH}\n\n"
            f"Expected the final adapter at:\n"
            f"models/final_adapter"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=QUANTIZATION_CONFIG,
        device_map="auto",
        dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
    )

    model.eval()

    return model


# ============================================================
# Generation
# ============================================================

def generate_response(
    model,
    tokenizer,
    instruction,
):

    prompt = (
        "Customer:\n"
        f"{instruction.strip()}\n\n"
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

    latency = (
        time.perf_counter()
        - start_time
    )

    generated_tokens = (
        outputs[0][
            inputs["input_ids"].shape[1]:
        ]
    )

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    generated_token_count = (
        generated_tokens.shape[0]
    )

    return (
        response,
        latency,
        generated_token_count,
    )


# ============================================================
# Evaluate One Model
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    dataset,
    model_name,
):

    print_header(
        f"Evaluating {model_name}"
    )

    predictions = []

    latencies = []

    generated_token_counts = []

    for index, example in enumerate(dataset):

        instruction = example["instruction"]
        reference = example["response"]

        (
            prediction,
            latency,
            generated_tokens,
        ) = generate_response(
            model,
            tokenizer,
            instruction,
        )

        predictions.append(
            {
                "instruction": instruction,
                "reference": reference,
                "prediction": prediction,
                "latency_seconds": latency,
                "generated_tokens": generated_tokens,
            }
        )

        latencies.append(latency)

        generated_token_counts.append(
            generated_tokens
        )

        print(
            f"[{index + 1:03d}/{len(dataset):03d}] "
            f"{latency:.2f}s"
        )

    average_latency = (
        sum(latencies)
        / len(latencies)
    )

    median_latency = (
        statistics.median(latencies)
    )

    average_generated_tokens = (
        sum(generated_token_counts)
        / len(generated_token_counts)
    )

    return {
        "model": model_name,
        "samples": len(predictions),
        "average_latency_seconds": average_latency,
        "median_latency_seconds": median_latency,
        "average_generated_tokens": average_generated_tokens,
        "predictions": predictions,
    }


# ============================================================
# Quality Metrics
# ============================================================

def calculate_metrics(results):

    references = [
        item["reference"]
        for item in results["predictions"]
    ]

    predictions = [
        item["prediction"]
        for item in results["predictions"]
    ]

    # --------------------------------------------------------
    # ROUGE-L
    # --------------------------------------------------------

    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    rouge_scores = []

    for reference, prediction in zip(
        references,
        predictions,
    ):

        result = rouge.score(
            reference,
            prediction,
        )

        rouge_scores.append(
            result["rougeL"].fmeasure
        )

    average_rouge_l = (
        sum(rouge_scores)
        / len(rouge_scores)
    )

    # --------------------------------------------------------
    # BERTScore
    # --------------------------------------------------------

    print(
        "\nCalculating BERTScore..."
    )

    precision, recall, f1 = bert_score(
        predictions,
        references,
        lang="en",
        verbose=True,
    )

    average_bert_precision = (
        precision.mean().item()
    )

    average_bert_recall = (
        recall.mean().item()
    )

    average_bert_f1 = (
        f1.mean().item()
    )

    return {
        "rougeL_f1": round(
            average_rouge_l,
            4,
        ),
        "bert_score_precision": round(
            average_bert_precision,
            4,
        ),
        "bert_score_recall": round(
            average_bert_recall,
            4,
        ),
        "bert_score_f1": round(
            average_bert_f1,
            4,
        ),
        "average_latency_seconds": round(
            results["average_latency_seconds"],
            4,
        ),
        "median_latency_seconds": round(
            results["median_latency_seconds"],
            4,
        ),
        "average_generated_tokens": round(
            results["average_generated_tokens"],
            2,
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print_header(
        "NexaTune AI - Reproducible Benchmark"
    )

    print(
        f"Base model       : {BASE_MODEL}"
    )

    print(
        f"Adapter           : {ADAPTER_PATH}"
    )

    print(
        f"Dataset           : {DATASET_PATH}"
    )

    print(
        f"Samples           : {NUM_SAMPLES}"
    )

    print(
        f"Seed              : {SEED}"
    )

    print(
        f"Max new tokens    : {MAX_NEW_TOKENS}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = load_tokenizer()

    test_dataset = load_test_dataset()

    # ========================================================
    # BASE MODEL
    # ========================================================

    base_model = load_base_model()

    base_results = evaluate_model(
        base_model,
        tokenizer,
        test_dataset,
        "Qwen3-1.7B-Base",
    )

    print("\nCalculating BASE metrics...")

    base_metrics = calculate_metrics(
        base_results
    )

    # Save base predictions immediately
    with open(
        OUTPUT_DIR / "base_predictions.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            base_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    del base_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # NEXATUNE MODEL
    # ========================================================

    finetuned_model = load_finetuned_model()

    finetuned_results = evaluate_model(
        finetuned_model,
        tokenizer,
        test_dataset,
        "NexaTune-Qwen3-1.7B",
    )

    print(
        "\nCalculating NEXATUNE metrics..."
    )

    finetuned_metrics = calculate_metrics(
        finetuned_results
    )

    # Save fine-tuned predictions
    with open(
        OUTPUT_DIR / "finetuned_predictions.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            finetuned_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    del finetuned_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # Final Comparison
    # ========================================================

    comparison = {
        "benchmark": {
            "base_model": BASE_MODEL,
            "adapter_path": str(ADAPTER_PATH),
            "dataset_path": str(DATASET_PATH),
            "num_examples": len(test_dataset),
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
        },
        "base_model": {
            "model": base_results["model"],
            **base_metrics,
        },
        "fine_tuned_model": {
            "model": finetuned_results["model"],
            **finetuned_metrics,
        },
    }

    with open(
        OUTPUT_DIR / "benchmark_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            comparison,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Console Summary
    # ========================================================

    print_header(
        "BENCHMARK COMPLETE"
    )

    print(
        "\n                     BASE       NEXATUNE"
    )

    print(
        f"ROUGE-L F1          "
        f"{base_metrics['rougeL_f1']:.4f}     "
        f"{finetuned_metrics['rougeL_f1']:.4f}"
    )

    print(
        f"BERTScore F1        "
        f"{base_metrics['bert_score_f1']:.4f}     "
        f"{finetuned_metrics['bert_score_f1']:.4f}"
    )

    print(
        f"Avg latency         "
        f"{base_metrics['average_latency_seconds']:.2f}s      "
        f"{finetuned_metrics['average_latency_seconds']:.2f}s"
    )

    print(
        f"Median latency      "
        f"{base_metrics['median_latency_seconds']:.2f}s      "
        f"{finetuned_metrics['median_latency_seconds']:.2f}s"
    )

    print(
        f"Avg generated tok.  "
        f"{base_metrics['average_generated_tokens']:.1f}      "
        f"{finetuned_metrics['average_generated_tokens']:.1f}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFiles:"
    )

    print(
        "  - base_predictions.json"
    )

    print(
        "  - finetuned_predictions.json"
    )

    print(
        "  - benchmark_metrics.json"
    )


if __name__ == "__main__":
    main()