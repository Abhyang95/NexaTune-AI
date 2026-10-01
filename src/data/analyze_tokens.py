from datasets import load_from_disk
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
DATASET_PATH = "data/processed/customer_support_sft"


print("=" * 70)
print("NexaTune AI - Token Length Analysis")
print("=" * 70)


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

dataset = load_from_disk(DATASET_PATH)

print("\nDataset:")
print(dataset)


# ---------------------------------------------------------
# Analyze token lengths
# ---------------------------------------------------------

for split_name in dataset:

    print("\n" + "-" * 70)
    print(f"{split_name.upper()} TOKEN STATISTICS")
    print("-" * 70)

    lengths = []

    for text in dataset[split_name]["text"]:

        tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        lengths.append(
            len(tokens["input_ids"])
        )

    lengths.sort()

    total = len(lengths)

    def percentile(values, p):
        index = int(len(values) * p)
        index = min(index, len(values) - 1)
        return values[index]

    print(f"Examples : {total}")
    print(f"Minimum  : {min(lengths)} tokens")
    print(f"Maximum  : {max(lengths)} tokens")
    print(f"Average  : {sum(lengths) / total:.2f} tokens")
    print(f"Median   : {lengths[total // 2]} tokens")
    print(f"P90      : {percentile(lengths, 0.90)} tokens")
    print(f"P95      : {percentile(lengths, 0.95)} tokens")
    print(f"P99      : {percentile(lengths, 0.99)} tokens")


print("\n" + "=" * 70)
print("Token analysis completed.")
print("=" * 70)