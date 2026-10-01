import os
import time
from pathlib import Path

# Hugging Face ZeroGPU compatibility
#
# `spaces` is available on Hugging Face Spaces.
# Locally it may not be installed, so we provide a no-op fallback.
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

ADAPTER_PATH = Path(
    os.getenv(
        "NEXATUNE_ADAPTER_PATH",
        "models/final_adapter",
    )
)


# ============================================================
# Inference Service
# ============================================================

class NexaTuneInference:
    """
    NexaTune QLoRA inference service.

    Designed to support:

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

        ZeroGPU requires CUDA.
        Local development uses CUDA when available and
        otherwise falls back to CPU.
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

        logger.info(
            "Loading NexaTune inference model"
        )

        logger.info(
            "Base model: %s",
            BASE_MODEL,
        )

        logger.info(
            "Adapter path: %s",
            ADAPTER_PATH,
        )

        if not ADAPTER_PATH.exists():
            logger.error(
                "NexaTune adapter not found at: %s",
                ADAPTER_PATH,
            )

            raise FileNotFoundError(
                f"NexaTune adapter not found at:\n"
                f"{ADAPTER_PATH}\n\n"
                "Set NEXATUNE_ADAPTER_PATH or place the adapter "
                "inside models/final_adapter."
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

        #
        # ZeroGPU:
        #   The current HF ZeroGPU runtime supports loading
        #   CUDA-backed models during startup and manages the
        #   actual GPU allocation for @spaces.GPU calls.
        #
        # Local:
        #   device_map="auto" allows Accelerate to place the
        #   quantized model appropriately.
        #

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
            str(ADAPTER_PATH),
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

        Locally, @spaces.GPU is a no-op and generation
        runs normally on the local machine.
        """

        if self.model is None:
            raise RuntimeError(
                "NexaTune model is not loaded."
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        # Move inputs to the same device used by the model.
        #
        # For ZeroGPU this resolves to CUDA when the
        # decorated function is executing.
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
        Generate a customer-support response and apply the
        NexaTune capability guardrail.

        Returns:
            dict containing:

            - response
            - guardrail_flagged
            - guardrail_matches
            - guardrail_categories
            - original_response
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