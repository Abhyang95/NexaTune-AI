from datasets import load_dataset
from collections import Counter


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


print("=" * 70)
print("NexaTune AI - Dataset Inspection")
print("=" * 70)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

dataset = load_dataset(DATASET_NAME)

train = dataset["train"]

print(f"\nTotal examples: {len(train)}")

print("\nColumns:")
for column in train.column_names:
    print(f"  - {column}")


# ---------------------------------------------------------
# Category analysis
# ---------------------------------------------------------

categories = Counter(train["category"])

print("\n" + "-" * 70)
print("CATEGORY DISTRIBUTION")
print("-" * 70)

for category, count in categories.most_common():
    percentage = count / len(train) * 100
    print(f"{category:<30} {count:>6} ({percentage:>6.2f}%)")


# ---------------------------------------------------------
# Intent analysis
# ---------------------------------------------------------

intents = Counter(train["intent"])

print("\n" + "-" * 70)
print("INTENT DISTRIBUTION")
print("-" * 70)

print(f"Unique intents: {len(intents)}")

for intent, count in intents.most_common():
    percentage = count / len(train) * 100
    print(f"{intent:<40} {count:>6} ({percentage:>6.2f}%)")


# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

for column in train.column_names:
    missing = sum(
        1
        for value in train[column]
        if value is None or str(value).strip() == ""
    )

    print(f"{column:<15}: {missing}")


# ---------------------------------------------------------
# Duplicate analysis
# ---------------------------------------------------------

instructions = train["instruction"]
responses = train["response"]

instruction_duplicates = len(instructions) - len(set(instructions))

instruction_response_pairs = [
    (instruction, response)
    for instruction, response in zip(instructions, responses)
]

pair_duplicates = (
    len(instruction_response_pairs)
    - len(set(instruction_response_pairs))
)

print("\n" + "-" * 70)
print("DUPLICATE ANALYSIS")
print("-" * 70)

print(f"Duplicate instructions: {instruction_duplicates}")
print(f"Duplicate instruction-response pairs: {pair_duplicates}")


# ---------------------------------------------------------
# Text length statistics
# ---------------------------------------------------------

instruction_lengths = [
    len(str(x).split())
    for x in instructions
]

response_lengths = [
    len(str(x).split())
    for x in responses
]


def statistics(values):
    values = sorted(values)

    return {
        "min": min(values),
        "max": max(values),
        "average": sum(values) / len(values),
        "median": values[len(values) // 2],
    }


instruction_stats = statistics(instruction_lengths)
response_stats = statistics(response_lengths)


print("\n" + "-" * 70)
print("TEXT LENGTH STATISTICS")
print("-" * 70)

print("\nInstruction:")
print(f"  Minimum : {instruction_stats['min']} words")
print(f"  Maximum : {instruction_stats['max']} words")
print(f"  Average : {instruction_stats['average']:.2f} words")
print(f"  Median  : {instruction_stats['median']} words")

print("\nResponse:")
print(f"  Minimum : {response_stats['min']} words")
print(f"  Maximum : {response_stats['max']} words")
print(f"  Average : {response_stats['average']:.2f} words")
print(f"  Median  : {response_stats['median']} words")


# ---------------------------------------------------------
# Sample records
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("SAMPLE RECORDS")
print("-" * 70)

for i in range(min(5, len(train))):
    example = train[i]

    print(f"\nExample {i + 1}")
    print(f"Category   : {example['category']}")
    print(f"Intent     : {example['intent']}")
    print(f"Instruction: {example['instruction']}")
    print(f"Response   : {example['response']}")


print("\n" + "=" * 70)
print("Inspection completed.")
print("=" * 70)