import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download

# ============================================================
# Hugging Face ZeroGPU compatibility
# ============================================================

try:
    import spaces
except ImportError:

    class _LocalSpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    spaces = _LocalSpaces()


import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel

from api.logging_config import logger
from api.capability_guardrail import (
    get_safe_response,
    scan_for_unsupported_claims,
)


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

ADAPTER_REPO = os.getenv(
    "NEXATUNE_ADAPTER_REPO",
    "abhyang95/nexatune-qwen3-1.7b-adapter",
)

LOCAL_ADAPTER_PATH = Path(
    os.getenv(
        "NEXATUNE_ADAPTER_PATH",
        "models/final_adapter",
    )
)

USE_HF_ADAPTER = (
    os.getenv(
        "NEXATUNE_USE_HF_ADAPTER",
        "false",
    ).lower()
    == "true"
)


# ============================================================
# Adapter resolution
# ============================================================

def resolve_adapter_path() -> Path:
    """
    Resolve the NexaTune LoRA adapter.

    Local development:
        models/final_adapter

    Hugging Face Space:
        Downloads the adapter from the Hugging Face Hub.
    """

    if USE_HF_ADAPTER:
        logger.info(
            "Downloading NexaTune adapter from Hugging Face Hub: %s",
            ADAPTER_REPO,
        )

        downloaded_path = snapshot_download(
            repo_id=ADAPTER_REPO,
            repo_type="model",
        )

        logger.info(
            "NexaTune adapter downloaded to: %s",
            downloaded_path,
        )

        return Path(downloaded_path)

    return LOCAL_ADAPTER_PATH


# ============================================================
# Inference Service
# ============================================================

class NexaTuneInference:
    """
    NexaTune QLoRA inference service.

    Supports:

    1. Local FastAPI inference
    2. Hugging Face ZeroGPU Gradio inference

    The actual generation method is decorated with
    @spaces.GPU. On a normal local machine this decorator
    becomes a no-op.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    @property
    def device(self):
        """
        Determine the device used for inference.

        ZeroGPU uses CUDA during GPU allocation.
        Local development uses CUDA when available.
        """

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    # --------------------------------------------------------
    # Model loading
    # --------------------------------------------------------

    def load(self):
        """
        Load the Qwen base model and NexaTune LoRA adapter.

        The model is loaded once and reused for subsequent
        generation requests.
        """

        if self.loaded:
            logger.info(
                "NexaTune model already loaded; skipping reload"
            )
            return

        # ----------------------------------------------------
        # Resolve adapter
        # ----------------------------------------------------

        adapter_path = resolve_adapter_path()

        logger.info(
            "Loading NexaTune inference model"
        )

        logger.info(
            "Base model: %s",
            BASE_MODEL,
        )

        logger.info(
            "Adapter path: %s",
            adapter_path,
        )

        if not adapter_path.exists():
            logger.error(
                "NexaTune adapter not found at: %s",
                adapter_path,
            )

            raise FileNotFoundError(
                f"NexaTune adapter not found at:\n"
                f"{adapter_path}\n\n"
                "For local development, place the adapter "
                "inside models/final_adapter.\n"
                "For Hugging Face Spaces, set "
                "NEXATUNE_USE_HF_ADAPTER=true."
            )

        logger.info(
            "CUDA available: %s",
            torch.cuda.is_available(),
        )

        if torch.cuda.is_available():
            logger.info(
                "CUDA device: %s",
                torch.cuda.get_device_name(
                    torch.cuda.current_device()
                ),
            )

        # ----------------------------------------------------
        # Quantization
        # ----------------------------------------------------

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        # ----------------------------------------------------
        # Base model
        # ----------------------------------------------------

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16,
        )

        # ----------------------------------------------------
        # LoRA adapter
        # ----------------------------------------------------

        self.model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
        )

        self.model.eval()

        self.loaded = True

        logger.info(
            "NexaTune model loaded successfully"
        )

        logger.info(
            "Inference device: %s",
            self.model.device,
        )

    # --------------------------------------------------------
    # GPU generation
    # --------------------------------------------------------

    @spaces.GPU(duration=120)
    def _generate_on_gpu(
        self,
        prompt: str,
        max_new_tokens: int,
    ):
        """
        Perform model generation.

        On Hugging Face ZeroGPU, this function receives
        temporary GPU allocation.

        Locally, @spaces.GPU is a no-op.
        """

        if self.model is None:
            raise RuntimeError(
                "NexaTune model is not loaded."
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return response, len(generated_tokens)

    # --------------------------------------------------------
    # Public generation API
    # --------------------------------------------------------

    def generate(
        self,
        customer_message: str,
        max_new_tokens: int = 256,
    ):
        """
        Generate a customer-support response and apply
        the NexaTune capability guardrail.
        """

        if not self.loaded:
            logger.error(
                "Generation requested while model is not loaded"
            )

            raise RuntimeError(
                "NexaTune model is not loaded."
            )

        start_time = time.perf_counter()

        prompt = (
            "Customer:\n"
            f"{customer_message.strip()}\n\n"
            "Assistant:\n"
        )

        # ----------------------------------------------------
        # Model generation
        # ----------------------------------------------------

        response, generated_token_count = (
            self._generate_on_gpu(
                prompt,
                max_new_tokens,
            )
        )

        latency = (
            time.perf_counter() - start_time
        )

        logger.info(
            "Generation completed | latency=%.2fs | "
            "generated_tokens=%d",
            latency,
            generated_token_count,
        )

        # ----------------------------------------------------
        # Capability guardrail
        # ----------------------------------------------------

        guardrail_result = (
            scan_for_unsupported_claims(
                response
            )
        )

        logger.info(
            "Guardrail scan | flagged=%s | categories=%s",
            guardrail_result.flagged,
            guardrail_result.categories,
        )

        # ----------------------------------------------------
        # Guardrail remediation
        # ----------------------------------------------------

        if guardrail_result.flagged:

            logger.warning(
                "Guardrail remediation triggered | "
                "categories=%s",
                guardrail_result.categories,
            )

            safe_response = get_safe_response(
                guardrail_result.matches,
                guardrail_result.categories,
            )

            return {
                "response": safe_response,
                "guardrail_flagged": True,
                "guardrail_matches": (
                    guardrail_result.matches
                ),
                "guardrail_categories": (
                    guardrail_result.categories
                ),
                "original_response": response,
            }

        # ----------------------------------------------------
        # Normal response
        # ----------------------------------------------------

        return {
            "response": response,
            "guardrail_flagged": False,
            "guardrail_matches": [],
            "guardrail_categories": [],
            "original_response": None,
        }


# ============================================================
# Global inference service
# ============================================================

inference_service = NexaTuneInference()