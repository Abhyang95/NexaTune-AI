from pathlib import Path


# ============================================================
# Model
# ============================================================

MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


# ============================================================
# Dataset
# ============================================================

DATASET_PATH = "data/processed/customer_support_sft"


# ============================================================
# Output
# ============================================================

OUTPUT_DIR = Path("outputs/nexatune-qlora")


# ============================================================
# Sequence configuration
# ============================================================

MAX_LENGTH = 512


# ============================================================
# QLoRA configuration
# ============================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


# ============================================================
# Training configuration
# ============================================================

NUM_EPOCHS = 1
MAX_STEPS = -1

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10
EVAL_STEPS = 250
SAVE_STEPS = 250

SEED = 42


# ============================================================
# Quantization
# ============================================================

LOAD_IN_4BIT = True

BNB_4BIT_QUANT_TYPE = "nf4"

BNB_4BIT_USE_DOUBLE_QUANT = True

BNB_4BIT_COMPUTE_DTYPE = "float16"