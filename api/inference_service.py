
from pathlib import Path
import os
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from api.logging_config import logger
from api.capability_guardrail import (
    scan_for_unsupported_claims,
    get_safe_response,
)


BASE_MODEL = "Qwen/Qwen3-1.7B-Base"

ADAPTER_PATH = Path(
    os.getenv(
        "NEXATUNE_ADAPTER_PATH",
        "models/final_adapter",
    )
)


class NexaTuneInference:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self):
        """
        Load the Qwen base model and NexaTune LoRA adapter.
        """

        logger.info("Loading NexaTune inference model")
        logger.info("Base model: %s", BASE_MODEL)
        logger.info("Adapter path: %s", ADAPTER_PATH)

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

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_PATH),
        )

        self.model.eval()
        self.loaded = True

        logger.info(
            "NexaTune model loaded successfully"
        )

    def generate(
        self,
        customer_message: str,
        max_new_tokens: int = 256,
    ):
        """
        Generate a response and run the capability guardrail.

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

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        latency = time.perf_counter() - start_time
        generated_token_count = len(generated_tokens)

        logger.info(
            "Generation completed | latency=%.2fs | generated_tokens=%d",
            latency,
            generated_token_count,
        )

        # Capability guardrail
        guardrail_result = scan_for_unsupported_claims(
            response
        )

        logger.info(
            "Guardrail scan | flagged=%s | categories=%s",
            guardrail_result.flagged,
            guardrail_result.categories,
        )

        # Remediate unsupported capability claims
        if guardrail_result.flagged:

            logger.warning(
                "Guardrail remediation triggered | categories=%s",
                guardrail_result.categories,
            )

            safe_response = get_safe_response(
                guardrail_result.matches,
                guardrail_result.categories,
            )

            return {
                "response": safe_response,
                "guardrail_flagged": True,
                "guardrail_matches": guardrail_result.matches,
                "guardrail_categories": (
                    guardrail_result.categories
                ),
                "original_response": response,
            }

        return {
            "response": response,
            "guardrail_flagged": False,
            "guardrail_matches": [],
            "guardrail_categories": [],
            "original_response": None,
        }


# Global inference service
inference_service = NexaTuneInference()

