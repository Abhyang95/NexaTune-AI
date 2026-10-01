import json
from pathlib import Path

import numpy as np
from bert_score import score
from rouge_score import rouge_scorer


# ============================================================
# Configuration
# ============================================================

BASE_PATH = Path(
    "results/benchmark/base_predictions.json"
)

PROMPTED_PATH = Path(
    "results/benchmark/prompted_base_predictions.json"
)

FINETUNED_PATH = Path(
    "results/benchmark/finetuned_predictions.json"
)

OUTPUT_PATH = Path(
    "results/benchmark/bootstrap_ci.json"
)

N_BOOTSTRAP = 10_000
SEED = 42


# ============================================================
# Helpers
# ============================================================

def load_predictions(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)["predictions"]


def validate_alignment(
    reference_predictions,
    other_predictions,
    model_name,
):

    if len(reference_predictions) != len(
        other_predictions
    ):

        raise ValueError(
            f"Prediction count mismatch: "
            f"Reference={len(reference_predictions)}, "
            f"{model_name}={len(other_predictions)}"
        )

    for i, (
        reference_item,
        other_item,
    ) in enumerate(
        zip(
            reference_predictions,
            other_predictions,
        )
    ):

        if (
            reference_item["instruction"]
            != other_item["instruction"]
        ):

            raise ValueError(
                f"Instruction mismatch at index {i} "
                f"for {model_name}"
            )


def paired_bootstrap(
    first_scores,
    second_scores,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):

    differences = (
        second_scores
        - first_scores
    )

    observed_difference = np.mean(
        differences
    )

    rng = np.random.default_rng(seed)

    bootstrap_diffs = np.empty(
        n_bootstrap
    )

    for i in range(n_bootstrap):

        indices = rng.integers(
            0,
            len(differences),
            size=len(differences),
        )

        bootstrap_diffs[i] = np.mean(
            differences[indices]
        )

    ci_low, ci_high = np.percentile(
        bootstrap_diffs,
        [2.5, 97.5],
    )

    probability_positive = np.mean(
        bootstrap_diffs > 0
    )

    return {
        "first_mean": float(
            np.mean(first_scores)
        ),
        "second_mean": float(
            np.mean(second_scores)
        ),
        "observed_difference": float(
            observed_difference
        ),
        "confidence_level": 0.95,
        "confidence_interval": {
            "lower": float(ci_low),
            "upper": float(ci_high),
        },
        "bootstrap_probability_difference_positive": float(
            probability_positive
        ),
    }


# ============================================================
# Load Predictions
# ============================================================

print("=" * 70)
print("NexaTune AI - Paired Bootstrap Comparison")
print("=" * 70)

print("\nLoading prediction files...")

base = load_predictions(
    BASE_PATH
)

prompted = load_predictions(
    PROMPTED_PATH
)

finetuned = load_predictions(
    FINETUNED_PATH
)

print(
    f"Base examples       : {len(base)}"
)

print(
    f"Prompted examples   : {len(prompted)}"
)

print(
    f"NexaTune examples   : {len(finetuned)}"
)


# ============================================================
# Validate Alignment
# ============================================================

validate_alignment(
    base,
    prompted,
    "Prompted Base",
)

validate_alignment(
    base,
    finetuned,
    "NexaTune",
)

validate_alignment(
    prompted,
    finetuned,
    "NexaTune",
)

print(
    "\nPrediction alignment: PASSED"
)


# ============================================================
# References
# ============================================================

references = [
    item["reference"]
    for item in base
]

base_predictions = [
    item["prediction"]
    for item in base
]

prompted_predictions = [
    item["prediction"]
    for item in prompted
]

finetuned_predictions = [
    item["prediction"]
    for item in finetuned
]


# ============================================================
# ROUGE-L
# ============================================================

print(
    "\nCalculating per-example ROUGE-L..."
)

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True,
)


def rouge_scores(
    predictions,
    references,
):

    scores = []

    for prediction, reference in zip(
        predictions,
        references,
    ):

        result = rouge.score(
            reference,
            prediction,
        )

        scores.append(
            result["rougeL"].fmeasure
        )

    return np.array(
        scores,
        dtype=np.float64,
    )


base_rouge = rouge_scores(
    base_predictions,
    references,
)

prompted_rouge = rouge_scores(
    prompted_predictions,
    references,
)

finetuned_rouge = rouge_scores(
    finetuned_predictions,
    references,
)


# ============================================================
# BERTScore
# ============================================================

print(
    "\nCalculating per-example BERTScore..."
)

print("\nBase BERTScore...")

_, _, base_bert = score(
    base_predictions,
    references,
    lang="en",
    verbose=True,
)

print("\nPrompted Base BERTScore...")

_, _, prompted_bert = score(
    prompted_predictions,
    references,
    lang="en",
    verbose=True,
)

print("\nNexaTune BERTScore...")

_, _, finetuned_bert = score(
    finetuned_predictions,
    references,
    lang="en",
    verbose=True,
)

base_bert = (
    base_bert
    .cpu()
    .numpy()
)

prompted_bert = (
    prompted_bert
    .cpu()
    .numpy()
)

finetuned_bert = (
    finetuned_bert
    .cpu()
    .numpy()
)


# ============================================================
# Comparison Function
# ============================================================

def compare_models(
    first_name,
    first_rouge,
    first_bert,
    second_name,
    second_rouge,
    second_bert,
):

    rouge_result = paired_bootstrap(
        first_rouge,
        second_rouge,
    )

    bert_result = paired_bootstrap(
        first_bert,
        second_bert,
    )

    return {
        "comparison": (
            f"{first_name} vs {second_name}"
        ),
        "n_examples": len(first_rouge),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "rougeL": rouge_result,
        "bertscore": bert_result,
    }


# ============================================================
# Three Pairwise Comparisons
# ============================================================

print(
    "\nRunning paired bootstrap comparisons..."
)

base_vs_prompted = compare_models(
    "Base",
    base_rouge,
    base_bert,
    "Prompted Base",
    prompted_rouge,
    prompted_bert,
)

base_vs_finetuned = compare_models(
    "Base",
    base_rouge,
    base_bert,
    "NexaTune",
    finetuned_rouge,
    finetuned_bert,
)

prompted_vs_finetuned = compare_models(
    "Prompted Base",
    prompted_rouge,
    prompted_bert,
    "NexaTune",
    finetuned_rouge,
    finetuned_bert,
)


# ============================================================
# Output
# ============================================================

output = {
    "benchmark": {
        "n_examples": len(references),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "confidence_level": 0.95,
    },
    "comparisons": {
        "base_vs_prompted": base_vs_prompted,
        "base_vs_finetuned": base_vs_finetuned,
        "prompted_vs_finetuned": prompted_vs_finetuned,
    },
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
    )


# ============================================================
# Console Summary
# ============================================================

def print_comparison(result):

    print("\n" + "-" * 70)

    print(
        result["comparison"]
    )

    print("-" * 70)

    rouge_result = result["rougeL"]
    bert_result = result["bertscore"]

    print(
        f"ROUGE-L first mean : "
        f"{rouge_result['first_mean']:.4f}"
    )

    print(
        f"ROUGE-L second mean: "
        f"{rouge_result['second_mean']:.4f}"
    )

    print(
        f"ROUGE-L difference : "
        f"{rouge_result['observed_difference']:+.4f}"
    )

    print(
        f"ROUGE-L 95% CI     : "
        f"["
        f"{rouge_result['confidence_interval']['lower']:+.4f}, "
        f"{rouge_result['confidence_interval']['upper']:+.4f}"
        f"]"
    )

    print(
        f"ROUGE-L P(diff > 0): "
        f"{rouge_result['bootstrap_probability_difference_positive']:.4f}"
    )

    print()

    print(
        f"BERTScore first mean : "
        f"{bert_result['first_mean']:.4f}"
    )

    print(
        f"BERTScore second mean: "
        f"{bert_result['second_mean']:.4f}"
    )

    print(
        f"BERTScore difference : "
        f"{bert_result['observed_difference']:+.4f}"
    )

    print(
        f"BERTScore 95% CI     : "
        f"["
        f"{bert_result['confidence_interval']['lower']:+.4f}, "
        f"{bert_result['confidence_interval']['upper']:+.4f}"
        f"]"
    )

    print(
        f"BERTScore P(diff > 0): "
        f"{bert_result['bootstrap_probability_difference_positive']:.4f}"
    )


print_header = lambda title: print(
    "\n" + "=" * 70 + "\n"
    + title
    + "\n" + "=" * 70
)

print_header(
    "BOOTSTRAP RESULTS"
)

print_comparison(
    base_vs_prompted
)

print_comparison(
    base_vs_finetuned
)

print_comparison(
    prompted_vs_finetuned
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_PATH
)