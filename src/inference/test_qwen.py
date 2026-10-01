import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen3-1.7B-Base"


print("=" * 70)
print("NexaTune AI - Qwen3-1.7B Base Model Test")
print("=" * 70)

print(f"Model: {MODEL_NAME}")
print(f"GPU available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected.")

print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------
# 1. 4-bit quantization configuration
# ---------------------------------------------------------

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ---------------------------------------------------------
# 2. Load tokenizer
# ---------------------------------------------------------

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded.")


# ---------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------

print("\nLoading model in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
)

print("Model loaded successfully.")


# ---------------------------------------------------------
# 4. Display device information
# ---------------------------------------------------------

print("\nModel device:")
print(model.device)


if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    print(f"GPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory reserved:   {reserved:.2f} GB")


# ---------------------------------------------------------
# 5. Test prompt
# ---------------------------------------------------------

prompt = """Customer: I was charged twice for the same order.

Assistant:"""

print("\nPrompt:")
print(prompt)


inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to("cuda")


# ---------------------------------------------------------
# 6. Generate
# ---------------------------------------------------------

print("Generating response...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


# ---------------------------------------------------------
# 7. Decode
# ---------------------------------------------------------

generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("=" * 70)
print("MODEL OUTPUT")
print("=" * 70)

print(generated_text)

print("=" * 70)

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)

    print(f"Final GPU memory allocated: {allocated:.2f} GB")
    print(f"Final GPU memory reserved:   {reserved:.2f} GB")

print("\nNexaTune AI model test completed.")