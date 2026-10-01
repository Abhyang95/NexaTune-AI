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

from rouge_score import rouge_scorer
from bert_score import score as bert_score


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

DATASET_PATH = Path(
    "data/processed/customer_support_sft"
)

OUTPUT_DIR = Path(
    "results/benchmark"
)

NUM_SAMPLES = 100
SEED = 42
MAX_NEW_TOKENS = 128


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a professional customer support assistant.

Answer the customer's question clearly and concisely.

Do not invent order numbers, tracking numbers, refunds, account changes,
or actions you cannot actually perform.

If information is missing, ask the customer for the required details.

Do not claim that you performed an action unless the information explicitly
confirms that the action was completed.
""".strip()


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


# ============================================================
# Generation
# ============================================================

def generate_response(
    model,
    tokenizer,
    instruction,
):

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
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
# Evaluation
# ============================================================

def evaluate_model(
    model,
    tokenizer,
    dataset,
):

    print_header(
        "Evaluating Prompted Base Qwen3-1.7B"
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
        "model": "Qwen3-1.7B-Base-Prompted",
        "system_prompt": SYSTEM_PROMPT,
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

    return {
        "rougeL_f1": round(
            average_rouge_l,
            4,
        ),
        "bert_score_precision": round(
            precision.mean().item(),
            4,
        ),
        "bert_score_recall": round(
            recall.mean().item(),
            4,
        ),
        "bert_score_f1": round(
            f1.mean().item(),
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
        "NexaTune AI - Prompted Baseline Benchmark"
    )

    print(
        f"Base model       : {BASE_MODEL}"
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

    print(
        "\nPrompting strategy:"
    )

    print(
        SYSTEM_PROMPT
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer = load_tokenizer()

    test_dataset = load_test_dataset()

    # ========================================================
    # PROMPTED BASE MODEL
    # ========================================================

    model = load_base_model()

    results = evaluate_model(
        model,
        tokenizer,
        test_dataset,
    )

    print(
        "\nCalculating prompted-base metrics..."
    )

    metrics = calculate_metrics(
        results
    )

    # ========================================================
    # Save predictions
    # ========================================================

    predictions_path = (
        OUTPUT_DIR
        / "prompted_base_predictions.json"
    )

    with open(
        predictions_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Save metrics
    # ========================================================

    metrics_output = {
        "benchmark": {
            "base_model": BASE_MODEL,
            "dataset_path": str(DATASET_PATH),
            "num_examples": len(test_dataset),
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
        },
        "prompted_base_model": {
            "model": results["model"],
            "system_prompt": SYSTEM_PROMPT,
            **metrics,
        },
    }

    metrics_path = (
        OUTPUT_DIR
        / "prompted_base_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Cleanup
    # ========================================================

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # Console Summary
    # ========================================================

    print_header(
        "PROMPTED BASELINE COMPLETE"
    )

    print(
        f"\nROUGE-L F1          "
        f"{metrics['rougeL_f1']:.4f}"
    )

    print(
        f"BERTScore F1        "
        f"{metrics['bert_score_f1']:.4f}"
    )

    print(
        f"Avg latency         "
        f"{metrics['average_latency_seconds']:.2f}s"
    )

    print(
        f"Median latency      "
        f"{metrics['median_latency_seconds']:.2f}s"
    )

    print(
        f"Avg generated tok.  "
        f"{metrics['average_generated_tokens']:.1f}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {predictions_path}"
    )

    print(
        f"  {metrics_path}"
    )


if __name__ == "__main__":
    main()