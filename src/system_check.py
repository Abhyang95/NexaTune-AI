import torch
import transformers
import datasets
import peft
import trl
import bitsandbytes


print("=" * 60)
print("NexaTune AI - Environment Check")
print("=" * 60)

print(f"Python environment: OK")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Datasets: {datasets.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"TRL: {trl.__version__}")
print(f"bitsandbytes: {bitsandbytes.__version__}")

print("-" * 60)

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    props = torch.cuda.get_device_properties(0)

    print(
        f"VRAM: "
        f"{props.total_memory / (1024 ** 3):.2f} GB"
    )

print("=" * 60)