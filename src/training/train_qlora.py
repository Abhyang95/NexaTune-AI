
import sys
from pathlib import Path

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

import torch

from datasets import load_from_disk
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from src.training.config import (
    MODEL_NAME,
    DATASET_PATH,
    OUTPUT_DIR,
    MAX_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    NUM_EPOCHS,
    MAX_STEPS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LOGGING_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    SEED,
)


# ============================================================
# Configuration display
# ============================================================

def print_training_configuration():
    """Print the active NexaTune training configuration."""

    print("=" * 70)
    print("NexaTune AI - QLoRA Training Configuration")
    print("=" * 70)

    print(f"\nModel        : {MODEL_NAME}")
    print(f"Dataset      : {DATASET_PATH}")
    print(f"Max length   : {MAX_LENGTH}")
    print(f"Epochs       : {NUM_EPOCHS}")
    print(f"Max steps    : {MAX_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")

    print("\nLoRA configuration:")
    print(f"  Rank       : {LORA_R}")
    print(f"  Alpha      : {LORA_ALPHA}")
    print(f"  Dropout    : {LORA_DROPOUT}")
    print(f"  Targets    : {LORA_TARGET_MODULES}")


# ============================================================
# Device check
# ============================================================

def check_cuda():
    """Verify that a CUDA GPU is available for QLoRA training."""

    print("\n" + "-" * 70)
    print("Hardware")
    print("-" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for QLoRA training."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_properties = torch.cuda.get_device_properties(0)

    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(
        f"VRAM: "
        f"{gpu_properties.total_memory / (1024 ** 3):.2f} GB"
    )


# ============================================================
# Load dataset
# ============================================================

def load_training_dataset():
    """Load the prepared train and validation datasets."""

    print("\n" + "-" * 70)
    print("Loading dataset")
    print("-" * 70)

    dataset = load_from_disk(DATASET_PATH)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(f"Train examples     : {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


# ============================================================
# Load tokenizer
# ============================================================

def load_training_tokenizer():
    """Load and configure the base-model tokenizer."""

    print("\n" + "-" * 70)
    print("Loading tokenizer")
    print("-" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")

    return tokenizer


# ============================================================
# Quantization configuration
# ============================================================

def create_quantization_config():
    """Create the 4-bit NF4 QLoRA quantization configuration."""

    print("\n" + "-" * 70)
    print("Configuring 4-bit quantization")
    print("-" * 70)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("4-bit NF4 configuration ready.")

    return quantization_config


# ============================================================
# Load base model
# ============================================================

def load_base_model(quantization_config):
    """Load the Qwen base model using 4-bit quantization."""

    print("\n" + "-" * 70)
    print("Loading base model")
    print("-" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.config.use_cache = False

    print("Base model loaded.")

    return model


# ============================================================
# LoRA configuration
# ============================================================

def create_lora_config():
    """Create the PEFT LoRA configuration."""

    print("\n" + "-" * 70)
    print("Creating LoRA configuration")
    print("-" * 70)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False,
    )

    print("LoRA configuration ready.")

    return peft_config


# ============================================================
# SFT configuration
# ============================================================

def create_training_arguments():
    """Create the TRL SFT training configuration."""

    print("\n" + "-" * 70)
    print("Creating SFT configuration")
    print("-" * 70)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        dataset_text_field="text",
        max_length=MAX_LENGTH,

        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,

        # Precision
        fp16=False,
        bf16=False,

        # Keep gradient clipping disabled because the tested
        # training environment uses this configuration successfully.
        max_grad_norm=0.0,

        # Evaluation
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        # Saving
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,

        # Logging
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,

        # Reproducibility
        seed=SEED,

        # No external experiment tracker
        report_to="none",

        # Keep packing disabled for this experiment
        packing=False,

        # Don't automatically push to Hugging Face
        push_to_hub=False,

        # Model behavior
        use_cache=False,
    )

    print("SFT configuration ready.")

    return training_args


# ============================================================
# Create trainer
# ============================================================

def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    peft_config,
    training_args,
):
    """Create the NexaTune SFTTrainer."""

    print("\n" + "-" * 70)
    print("Creating SFTTrainer")
    print("-" * 70)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("SFTTrainer created successfully.")

    return trainer


# ============================================================
# Build trainer
# ============================================================

def build_trainer():
    """
    Build and return the complete NexaTune SFTTrainer.

    IMPORTANT:
    This function only prepares the training pipeline.
    It does NOT start training.
    """

    print_training_configuration()

    check_cuda()

    train_dataset, eval_dataset = load_training_dataset()

    tokenizer = load_training_tokenizer()

    quantization_config = create_quantization_config()

    model = load_base_model(quantization_config)

    peft_config = create_lora_config()

    training_args = create_training_arguments()

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        training_args=training_args,
    )

    # ========================================================
    # Trainable parameter summary
    # ========================================================

    print("\n" + "-" * 70)
    print("Trainable parameter summary")
    print("-" * 70)

    trainer.model.print_trainable_parameters()

    print("\n" + "=" * 70)
    print("TRAINER READY")
    print("=" * 70)

    print("\nTraining has NOT started.")
    print("Call trainer.train() from the training entry point.")

    return trainer, tokenizer


# ============================================================
# Import-safe module behavior
# ============================================================

if __name__ == "__main__":
    print("\nThis module builds the trainer but does not execute training directly.")
    print("Use the canonical training command instead:")
    print("\n    python -m src.training.run_training\n")

