# NexaTune AI — Model Card

## Model Overview

**Model:** NexaTune-Qwen3-1.7B  
**Base Model:** Qwen/Qwen3-1.7B-Base  
**Model Type:** Causal Language Model adapted for customer-support response generation  
**Fine-Tuning Method:** QLoRA + LoRA + Supervised Fine-Tuning (SFT)  
**Parameter-Efficient Fine-Tuning:** Yes  
**Primary Domain:** Customer Support  
**Adapter Format:** PEFT LoRA adapter  
**Quantization:** 4-bit NF4  
**Deployment:** FastAPI inference API + Gradio interface

NexaTune AI is a domain-adapted language model created by fine-tuning
Qwen3-1.7B-Base on a customer-support instruction/response dataset.

The project demonstrates an end-to-end model customization workflow:

```text
Customer Support Dataset
        ↓
Data Preparation + SFT Formatting
        ↓
QLoRA / LoRA Fine-Tuning
        ↓
Evaluation
        ↓
LoRA Adapter
        ↓
Inference API
        ↓
Gradio Application
```
## Intended Use

NexaTune is intended for experimentation and research involving:

Customer-support response generation
Domain adaptation of small language models
Parameter-efficient fine-tuning
Evaluation of fine-tuned language models
AI/ML engineering demonstrations
Educational demonstrations of QLoRA and LoRA
Local inference on consumer GPUs

The model is particularly useful as a demonstration of how a general-purpose
language model can be adapted toward a specialized customer-support domain.

## Out-of-Scope Uses

NexaTune should not be treated as:

A production customer-service agent without additional safeguards
A source of authoritative financial, legal, medical, or regulatory advice
A system capable of accessing real customer accounts
A system capable of verifying orders, payments, refunds, or shipments
A replacement for human customer-support personnel
A factual database of real-world order or account information

The model does not have access to external customer systems unless such tools
are explicitly integrated into the surrounding application.

## Base Model

NexaTune is built on:

Qwen/Qwen3-1.7B-Base

The base model provides the general language-generation capabilities while
fine-tuning adapts the model toward the customer-support domain.

The NexaTune project does not retrain all base-model parameters.

Instead, parameter-efficient fine-tuning is used to learn a relatively small
LoRA adapter.

## Fine-Tuning Method
QLoRA

NexaTune uses a QLoRA-based training setup.

The base model is loaded using:

4-bit quantization
NF4 quantization
Double quantization
FP16 computation

This significantly reduces the memory required to fine-tune the model.

LoRA Configuration

The adapter configuration uses:

Parameter	Value
LoRA rank (r)	16
LoRA alpha	32
LoRA dropout	0.05
Target modules	q_proj, k_proj, v_proj, o_proj
Trainable parameters	6,422,528
Total parameters	1,726,997,504
Trainable percentage	0.3719%

Only a small fraction of the total model parameters are updated during
fine-tuning.

## Training Configuration
Parameter	Value
Training method	SFT
Epochs	1
Learning rate	2e-4
Weight decay	0.01
Warmup steps	100
Per-device batch size	1
Gradient accumulation	8
Gradient checkpointing	Enabled
Maximum sequence length	512
Compute precision	FP16

Training was performed using a cloud T4 GPU environment because the local
GTX 1650 has limited VRAM for full training.

The final LoRA adapter was subsequently downloaded and used for local
inference.

## Dataset

NexaTune was trained on:

bitext/Bitext-customer-support-llm-chatbot-training-dataset

The original dataset contains 26,872 examples.

Relevant fields include:

instruction
category
intent
response

The dataset covers common customer-support scenarios such as:

Accounts
Orders
Refunds
Invoices
Payments
Delivery
Shipping
Subscriptions
Cancellation
Feedback
Contact/support requests
## Data Preparation

The dataset was inspected before training.

Important findings included:

No missing values were found.
There were duplicate instructions.
Exact instruction-response pairs were not duplicated.
The same instruction can have multiple valid responses.

Because of this, exact instruction-response pairs were retained rather than
removing examples solely because their instructions were identical.

The final processed dataset was split into:

Split	Examples
Train	21,500
Validation	2,713
Test	2,659
Total	26,872

The test split was kept separate from training.

## SFT Format

Training examples were converted into a customer/assistant conversational
format:

Customer:
{instruction}

Assistant:
{response}

This format is also used during evaluation and inference.

## Sequence Length Analysis

Token-length analysis was performed using the Qwen tokenizer.

Observed training-token statistics:

Statistic	Tokens
Mean	140.95
Median	119
95th percentile	275
99th percentile	380
Maximum	502

A maximum sequence length of 512 tokens was therefore selected for SFT.

## Evaluation Methodology

NexaTune was evaluated against the original base model:

Qwen/Qwen3-1.7B-Base

The comparison uses the same held-out test examples for both models.

Benchmark Configuration
Setting	Value
Evaluation examples	100
Dataset split	Test
Selection seed	42
Decoding	Deterministic
Sampling	Disabled
Maximum new tokens	128
Prompt format	Customer / Assistant
Base model	Qwen3-1.7B-Base
Fine-tuned model	Base + NexaTune LoRA

The benchmark can be reproduced with:

python -m src.evaluation.run_benchmark

Results are saved under:

results/benchmark/
## Quantitative Evaluation

The reproducible benchmark produced the following results:

Metric	Base Model	NexaTune
ROUGE-L F1	0.2120	0.4082
BERTScore F1	0.8574	0.9160
Average latency	9.22 s	15.45 s
Median latency	10.95 s	15.22 s
Average generated tokens	100.1	99.9

These measurements come from a 100-example held-out benchmark and should
not be interpreted as universal performance guarantees.

The benchmark demonstrates improved similarity to the reference responses
under the selected evaluation metrics, while inference latency was higher
for the fine-tuned adapter in this local evaluation environment.

## Evaluation Interpretation

The evaluation suggests that domain fine-tuning changed the model's response
behavior toward the customer-support examples represented in the training
data.

The improvement in ROUGE-L and BERTScore indicates greater similarity between
NexaTune responses and the reference responses on this benchmark.

However, automated text-similarity metrics do not fully measure:

Factual correctness
Helpfulness
Safety
Customer satisfaction
Appropriate escalation
Hallucination rate
Tool-use correctness
Policy compliance

Additional human or LLM-based evaluation is therefore planned.

## Qualitative Observations

Manual inspection of benchmark examples showed several useful behavioral
changes.

Examples included:

Article-related request

The base model sometimes produced conversational artifacts or refusal-like
behavior.

NexaTune more consistently asked for the relevant article details.

Invoice request

For an invoice-download request, the base model could produce an invented
link or unrelated artifact.

NexaTune more consistently directed the user toward the billing/invoice
workflow without inventing a real external resource.

Shipping-address issue

The base model sometimes asserted that an issue had already been resolved.

NexaTune more frequently asked for additional information needed to address
the request.

Tracking/order requests

The base model sometimes generated unsupported tracking information.

NexaTune more frequently requested the order or tracking identifier instead
of inventing a status.

These observations are qualitative and are not intended as statistically
validated behavioral claims.

## Known Limitations
1. Hallucination

NexaTune can still generate information that is not supported by the input.

Fine-tuning does not guarantee factual correctness.

2. Unsupported Capability Claims

In some examples, NexaTune can produce language suggesting that it is
connecting to or waiting for a human support agent even though no such tool
or workflow is actually connected.

For example, a deployed application should not allow the model to imply that
an external action has occurred unless that action has actually been
performed.

This should be addressed through application-level guardrails and tool
verification.

3. Limited Evaluation Size

The primary benchmark contains 100 examples.

This is useful for reproducibility and model comparison but is not sufficient
to establish broad production-level performance.

4. Reference-Based Metrics

ROUGE-L and BERTScore compare generated responses with reference responses.
A valid customer-support response may differ substantially in wording from
the reference while still being correct.

5. Latency

The fine-tuned model showed higher latency than the base model in the local
100-example benchmark.

Latency is hardware- and environment-dependent and should be re-measured
under the intended deployment configuration.

6. No External Customer-System Access

The model itself cannot:

Look up real orders
Process refunds
Verify account information
Change shipping addresses
Access payment systems
Contact human agents

Such capabilities require explicit external tools and application logic.

## Safety Considerations

NexaTune is a model adaptation experiment rather than an autonomous
transaction-processing system.

A production deployment should implement:

Input validation
Output filtering
Tool authorization
Authentication
Rate limiting
Logging and monitoring
Prompt-injection defenses
Sensitive-data handling
Human escalation
Verification before external actions

The application should distinguish between:

Model-generated text

and:

Verified external actions

The model should never be allowed to claim that an external operation
occurred solely because it generated text describing that operation.

## Deployment

NexaTune currently supports local inference through:

Gradio UI
      ↓
FastAPI
      ↓
Qwen3-1.7B + LoRA Adapter

FastAPI endpoints include:

GET  /health
GET  /model-info
POST /generate

The Gradio interface provides a customer-support playground and exposes
model/evaluation information.

## Hardware
Training

Training was performed using a cloud NVIDIA T4 environment with QLoRA.

Local Inference

Local inference was tested on:

GPU: NVIDIA GeForce GTX 1650
VRAM: 4 GB

The 4-bit quantized model allows inference on this constrained GPU.

## Adapter Artifacts

The final adapter contains:

adapter_config.json
adapter_model.safetensors
chat_template.jinja
README.md
tokenizer.json
tokenizer_config.json
training_args.bin

The primary adapter weights are stored in:

models/final_adapter/

The adapter uses approximately 25 MB of model weights.

## Reproducibility

The project provides scripts for:

Dataset preparation
        ↓
Token analysis
        ↓
        ↓
QLoRA training
        ↓
Inference
        ↓
Benchmark evaluation

The current reproducible benchmark command is:

python -m src.evaluation.run_benchmark

Benchmark artifacts are written to:

results/benchmark/
## Evaluation Status

The current evaluation workflow includes:

- Reproducible benchmark evaluation
- ROUGE-L and BERTScore
- Latency and generated-token analysis
- Paired bootstrap confidence intervals
- AI-assisted LLM-as-judge evaluation
- Qualitative failure analysis

The current reproducible benchmark command is:

```text
python -m src.evaluation.run_benchmark

## Project Scope

NexaTune AI is primarily an AI engineering and model-customization
project demonstrating:

Dataset engineering
Instruction formatting
Token analysis
Parameter-efficient fine-tuning
QLoRA
LoRA
Supervised fine-tuning
Model evaluation
Reproducible benchmarking
Local inference
FastAPI model serving
Gradio application development

The project deliberately focuses on the model customization lifecycle rather
than combining unrelated AI technologies.

## Responsible Use

Users deploying NexaTune should independently validate model outputs before
using them for consequential customer interactions.

The model should be treated as a probabilistic text-generation system and
not as a source of guaranteed truth.

Any production deployment should include appropriate monitoring, validation,
access control, and human oversight.

## Project Status

Current status:

- [x] Dataset preparation
- [x] Dataset analysis
- [x] SFT formatting
- [x] QLoRA fine-tuning
- [x] Final LoRA adapter
- [x] Local inference
- [x] FastAPI API
- [x] Gradio interface
- [x] 100-example benchmark
- [x] Reproducible benchmark command
- [ ] Bootstrap confidence intervals
- [ ] Human/LLM evaluation
- [ ] Prompted baseline comparison
- [ ] Production deployment

## Summary

NexaTune AI demonstrates how a small open-source language model can be
adapted to a specialized customer-support domain using parameter-efficient
fine-tuning.

The project compares the original Qwen3-1.7B base model with a QLoRA/LoRA
adapted version using a fixed 100-example held-out benchmark.

The current results show improved reference-response similarity under the
selected ROUGE-L and BERTScore metrics, alongside higher local inference
latency for the fine-tuned model.

The project therefore treats fine-tuning as a measurable engineering
experiment rather than assuming that model adaptation automatically improves
every aspect of system performance.


