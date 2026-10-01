import json
from pathlib import Path

from rouge_score import rouge_scorer
from bert_score import score


PREDICTIONS_PATH = Path(
    "results/baseline/predictions.json"
)

OUTPUT_PATH = Path(
    "results/baseline/quality_metrics.json"
)


print("=" * 70)
print("NexaTune AI - Baseline Quality Metrics")
print("=" * 70)


# ============================================================
# Load predictions
# ============================================================

print("\nLoading predictions...")

with open(PREDICTIONS_PATH, "r", encoding="utf-8") as file:
    predictions = json.load(file)

print(f"Loaded {len(predictions)} predictions.")


references = [
    item["reference_response"]
    for item in predictions
]

generated = [
    item["generated_response"]
    for item in predictions
]


# ============================================================
# ROUGE-L
# ============================================================

print("\n" + "-" * 70)
print("Calculating ROUGE-L")
print("-" * 70)

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True,
)

rouge_scores = []

for reference, prediction in zip(
    references,
    generated,
):
    score_result = rouge.score(
        reference,
        prediction,
    )

    rouge_scores.append(
        score_result["rougeL"].fmeasure
    )


average_rouge_l = (
    sum(rouge_scores) / len(rouge_scores)
)

print(
    f"Average ROUGE-L F1: "
    f"{average_rouge_l:.4f}"
)


# ============================================================
# BERTScore
# ============================================================

print("\n" + "-" * 70)
print("Calculating BERTScore")
print("-" * 70)

print(
    "BERTScore may download a scoring model "
    "from Hugging Face on the first run."
)

precision, recall, f1 = score(
    generated,
    references,
    lang="en",
    verbose=True,
)

average_bert_precision = precision.mean().item()
average_bert_recall = recall.mean().item()
average_bert_f1 = f1.mean().item()


print(
    f"\nBERTScore Precision: "
    f"{average_bert_precision:.4f}"
)

print(
    f"BERTScore Recall   : "
    f"{average_bert_recall:.4f}"
)

print(
    f"BERTScore F1       : "
    f"{average_bert_f1:.4f}"
)


# ============================================================
# Latency
# ============================================================

latencies = [
    item["latency_seconds"]
    for item in predictions
]

average_latency = (
    sum(latencies) / len(latencies)
)


# ============================================================
# Final metrics
# ============================================================

metrics = {
    "model": "Qwen/Qwen3-1.7B-Base",
    "num_examples": len(predictions),
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
        average_latency,
        4,
    ),
}


# ============================================================
# Save
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=2,
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("QUALITY METRICS COMPLETE")
print("=" * 70)

print(
    f"\nROUGE-L F1       : "
    f"{average_rouge_l:.4f}"
)

print(
    f"BERTScore F1     : "
    f"{average_bert_f1:.4f}"
)

print(
    f"Average latency  : "
    f"{average_latency:.4f}s"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_PATH}"
)

print("\n" + "=" * 70)