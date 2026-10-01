from datasets import load_from_disk


DATASET_PATH = "data/processed/customer_support"
OUTPUT_PATH = "data/processed/customer_support_sft"


print("=" * 70)
print("NexaTune AI - SFT Dataset Formatting")
print("=" * 70)


# ---------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------

print("\nLoading processed dataset...")

dataset = load_from_disk(DATASET_PATH)

print(dataset)


# ---------------------------------------------------------
# Format each example
# ---------------------------------------------------------

def format_example(example):

    text = (
        "Customer:\n"
        f"{example['instruction']}\n\n"
        "Assistant:\n"
        f"{example['response']}"
    )

    return {
        "text": text
    }


print("\nFormatting examples...")

formatted_dataset = dataset.map(
    format_example
)


# ---------------------------------------------------------
# Keep useful metadata + training text
# ---------------------------------------------------------

def select_columns(dataset_split):

    columns = [
        "instruction",
        "category",
        "intent",
        "response",
        "text",
    ]

    existing_columns = [
        column
        for column in columns
        if column in dataset_split.column_names
    ]

    return dataset_split.select_columns(existing_columns)


for split_name in formatted_dataset:
    formatted_dataset[split_name] = select_columns(
        formatted_dataset[split_name]
    )


# ---------------------------------------------------------
# Show examples
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("FORMATTED EXAMPLES")
print("-" * 70)

for i in range(3):

    example = formatted_dataset["train"][i]

    print(f"\nExample {i + 1}")
    print("-" * 50)
    print(example["text"])


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("Saving SFT dataset")
print("-" * 70)

formatted_dataset.save_to_disk(
    OUTPUT_PATH
)

print(f"\nSaved to: {OUTPUT_PATH}")

print("\n" + "=" * 70)
print("SFT formatting completed.")
print("=" * 70)