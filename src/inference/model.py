from pathlib import Path
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

ADAPTER_PATH = Path(
    os.getenv("NEXATUNE_ADAPTER_PATH", "models/final_adapter")
)


def load_model():
    print("=" * 70)
    print("Loading NexaTune inference model")
    print("=" * 70)

    print(f"Base model : {BASE_MODEL}")
    print(f"Adapter    : {ADAPTER_PATH}")

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Adapter not found:\n{ADAPTER_PATH}\n\n"
            "Place the NexaTune final adapter in models/final_adapter "
            "or set NEXATUNE_ADAPTER_PATH."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
    )

    model.eval()

    print("\nModel loaded successfully.")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    customer_message: str,
    max_new_tokens: int = 256,
):
    prompt = (
        "Customer:\n"
        f"{customer_message.strip()}\n\n"
        "Assistant:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return response