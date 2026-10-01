from datasets import load_dataset, DatasetDict


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

RANDOM_SEED = 42


print("=" * 70)
print("NexaTune AI - Dataset Preparation")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load raw dataset
# ---------------------------------------------------------

print("\nLoading dataset...")

dataset = load_dataset(DATASET_NAME)

data = dataset["train"]

print(f"Raw examples: {len(data)}")


# ---------------------------------------------------------
# 2. Remove unnecessary metadata
# ---------------------------------------------------------

print("\nRemoving unnecessary columns...")

columns_to_remove = ["flags"]

data = data.remove_columns(columns_to_remove)

print("Remaining columns:")
print(data.column_names)


# ---------------------------------------------------------
# 3. Validate required fields
# ---------------------------------------------------------

required_columns = [
    "instruction",
    "category",
    "intent",
    "response",
]

for column in required_columns:
    if column not in data.column_names:
        raise ValueError(f"Missing required column: {column}")


# ---------------------------------------------------------
# 4. Remove empty records
# ---------------------------------------------------------

print("\nChecking empty records...")

before = len(data)

data = data.filter(
    lambda x:
        x["instruction"] is not None
        and x["response"] is not None
        and x["instruction"].strip() != ""
        and x["response"].strip() != ""
)

after = len(data)

print(f"Removed empty records: {before - after}")
print(f"Remaining examples: {after}")


# ---------------------------------------------------------
# 5. Normalize whitespace
# ---------------------------------------------------------

print("\nNormalizing whitespace...")


def clean_text(example):
    example["instruction"] = " ".join(
        example["instruction"].split()
    )

    example["response"] = " ".join(
        example["response"].split()
    )

    example["category"] = example["category"].strip()
    example["intent"] = example["intent"].strip()

    return example


data = data.map(clean_text)


# ---------------------------------------------------------
# 6. Remove exact instruction-response duplicates
# ---------------------------------------------------------

print("\nChecking exact instruction-response duplicates...")

before = len(data)

data = data.to_pandas()

data = data.drop_duplicates(
    subset=["instruction", "response"]
)

after = len(data)

print(f"Exact pairs removed: {before - after}")

print(f"Examples after deduplication: {after}")


# Convert back to Hugging Face Dataset
from datasets import Dataset

data = Dataset.from_pandas(
    data,
    preserve_index=False
)


# ---------------------------------------------------------
# 7. Create grouped split by instruction
# ---------------------------------------------------------

print("\nCreating leakage-aware split...")


# Get unique instructions
unique_instructions = list(
    set(data["instruction"])
)

import random

random.seed(RANDOM_SEED)

random.shuffle(unique_instructions)


total_instructions = len(unique_instructions)

train_cutoff = int(total_instructions * 0.80)
val_cutoff = int(total_instructions * 0.90)

train_instructions = set(
    unique_instructions[:train_cutoff]
)

val_instructions = set(
    unique_instructions[train_cutoff:val_cutoff]
)

test_instructions = set(
    unique_instructions[val_cutoff:]
)


train_data = data.filter(
    lambda x: x["instruction"] in train_instructions
)

val_data = data.filter(
    lambda x: x["instruction"] in val_instructions
)

test_data = data.filter(
    lambda x: x["instruction"] in test_instructions
)


# ---------------------------------------------------------
# 8. Create DatasetDict
# ---------------------------------------------------------

final_dataset = DatasetDict({
    "train": train_data,
    "validation": val_data,
    "test": test_data,
})


# ---------------------------------------------------------
# 9. Display statistics
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("FINAL DATASET")
print("-" * 70)

for split_name, split in final_dataset.items():
    print(f"{split_name:<12}: {len(split):>6} examples")


# ---------------------------------------------------------
# 10. Save dataset
# ---------------------------------------------------------

output_path = "data/processed/customer_support"

print(f"\nSaving processed dataset to:")
print(output_path)

final_dataset.save_to_disk(output_path)


print("\n" + "=" * 70)
print("Dataset preparation completed successfully.")
print("=" * 70)