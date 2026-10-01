from datasets import load_dataset

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

print("=" * 70)
print("NexaTune AI - Dataset Download")
print("=" * 70)

print(f"\nLoading dataset: {DATASET_NAME}")

dataset = load_dataset(DATASET_NAME)

print("\nDataset loaded successfully.")

print("\nDataset structure:")
print(dataset)

for split in dataset:
    print(f"\n{split}: {len(dataset[split])} examples")

print("\nColumns:")
print(dataset["train"].column_names)

print("\nFirst example:")
print(dataset["train"][0])

print("\n" + "=" * 70)