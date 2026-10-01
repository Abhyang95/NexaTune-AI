# NexaTune AI

## Domain-Adaptive LLM Fine-Tuning & Evaluation Platform

NexaTune AI is an end-to-end LLM fine-tuning, evaluation, and inference platform that adapts an open-source language model to customer-support conversations using **QLoRA, LoRA, PEFT, and supervised fine-tuning (SFT)**.

The project focuses on an important practical question:

> Can parameter-efficient fine-tuning improve the behavior of a small open-source language model on a specialized customer-support task, and can that improvement be measured reproducibly?

NexaTune covers the complete model-customization lifecycle:

```text
Dataset
   ↓
Data Quality
   ↓
Leakage-Aware Splitting
   ↓
SFT Formatting
   ↓
Token Analysis
   ↓
Base Model Evaluation
   ↓
QLoRA + SFT
   ↓
Checkpoint Recovery
   ↓
LoRA Adapter
   ↓
Reproducible Benchmark
   ↓
Statistical Validation
   ↓
AI-Assisted LLM-as-Judge Evaluation
   ↓
Failure Analysis
   ↓
FastAPI Serving
   ↓
Capability Guardrails
   ↓
Gradio Product UI
```

---

# Project Highlights

* Fine-tuned **Qwen/Qwen3-1.7B-Base**
* Used **QLoRA + LoRA + PEFT + SFT**
* Trained on **26,872 customer-support examples**
* Used instruction-level grouped splitting to reduce evaluation leakage
* Used **4-bit NF4 quantization**
* Updated only approximately **0.372% of total model parameters**
* Used **LoRA rank 16** with attention projection targets
* Recovered training after a cloud-session interruption
* Produced a final LoRA adapter of approximately **24.5 MB**
* Built a deterministic **100-example held-out benchmark**
* Compared **Base**, **Prompted Base**, and **NexaTune**
* Added paired **10,000-resample bootstrap confidence intervals**
* Added **AI-assisted LLM-as-judge evaluation**
* Performed qualitative failure analysis
* Built a **FastAPI inference service**
* Added capability-aware post-generation guardrails
* Built a product-style **Gradio interface**
* Added structured inference logging and model metadata

---

# Table of Contents

* [Problem](#problem)
* [Why Fine-Tuning](#why-fine-tuning)
* [Architecture](#architecture)
* [End-to-End Workflow](#end-to-end-workflow)
* [Dataset](#dataset)
* [Data Preparation](#data-preparation)
* [Leakage Prevention](#leakage-prevention)
* [SFT Formatting](#sft-formatting)
* [Token Analysis](#token-analysis)
* [Fine-Tuning](#fine-tuning)
* [Training Configuration](#training-configuration)
* [Checkpoint Recovery](#checkpoint-recovery)
* [Evaluation Methodology](#evaluation-methodology)
* [Benchmark Results](#benchmark-results)
* [Statistical Validation](#statistical-validation)
* [AI-Assisted LLM-as-Judge Evaluation](#ai-assisted-llm-as-judge-evaluation)
* [Qualitative Failure Analysis](#qualitative-failure-analysis)
* [Capability Guardrails](#capability-guardrails)
* [FastAPI Inference API](#fastapi-inference-api)
* [Gradio Interface](#gradio-interface)
* [Project Structure](#project-structure)
* [Local Setup](#local-setup)
* [Running the Application](#running-the-application)
* [Reproducible Benchmark](#reproducible-benchmark)
* [Training Entry Point](#training-entry-point)
* [Model Card](#model-card)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [Interview Talking Points](#interview-talking-points)
* [Project Status](#project-status)

---

# Problem

General-purpose language models can generate useful text, but they may not consistently follow the style, terminology, response structure, and behavioral patterns required by a specialized customer-support domain.

NexaTune investigates whether parameter-efficient fine-tuning can adapt a relatively small open-source model to customer-support conversations while keeping the trainable parameter footprint small.

The goal is not simply to generate more text or use a larger model.

The goal is to:

1. establish a reproducible baseline,
2. adapt the model,
3. measure the change,
4. investigate failure cases,
5. validate the results statistically,
6. expose the model through an API,
7. add safeguards around unsupported capabilities.

---

# Why Fine-Tuning?

## Fine-Tuning

Fine-tuning changes model behavior through learned parameter updates.

It can be useful for:

* domain-specific response patterns
* response style
* instruction following
* task-specific formatting
* consistent behavioral patterns

NexaTune uses **LoRA adapters** rather than updating the entire base model.

## RAG

Retrieval-Augmented Generation provides external information at inference time.

RAG is particularly useful for:

* frequently changing information
* private documents
* knowledge retrieval
* citations
* grounding responses in external sources

NexaTune intentionally focuses on **model adaptation rather than retrieval**.

This gives the project a different role from a separate RAG/agentic project in the portfolio.

A production customer-support system could combine both:

```text
Fine-Tuned Model
      +
RAG / Knowledge Base
      +
Tools / APIs
      +
Guardrails
```

---

# Architecture

```text
┌───────────────────────────────────────────────┐
│ Customer Support Dataset                      │
│ 26,872 examples                               │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│ Data Validation                               │
│ Missing values / duplicates / quality checks  │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│ Leakage-Aware Splitting                       │
│ Instruction-grouped train / validation / test│
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│ SFT Formatting                                │
│ Customer → Assistant                          │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│ Qwen/Qwen3-1.7B-Base                          │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│ QLoRA + PEFT + SFT                            │
│ 4-bit NF4                                    │
│ LoRA r=16, alpha=32                           │
└──────────────────────┬────────────────────────┘
                       ↓
┌───────────────────────────────────────────────┐
│ LoRA Adapter                                  │
│ ~24.5 MB                                      │
└──────────────────────┬────────────────────────┘
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
┌───────────────────────┐   ┌───────────────────────┐
│ Deterministic         │   │ Qualitative           │
│ Benchmark             │   │ Failure Analysis      │
└───────────┬───────────┘   └───────────┬───────────┘
            ↓                           ↓
┌───────────────────────┐   ┌───────────────────────┐
│ Bootstrap CI          │   │ AI-Assisted           │
│ 10,000 Resamples      │   │ LLM-as-Judge          │
└───────────┬───────────┘   └───────────┬───────────┘
            └──────────────┬─────────────┘
                           ↓
                ┌───────────────────────┐
                │ FastAPI Inference API │
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ Capability Guardrail  │
                │ Detect → Replace      │
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ Gradio Product UI     │
                └───────────────────────┘
```

---

# End-to-End Workflow

```text
Dataset
  ↓
Data inspection
  ↓
Cleaning + exact-pair deduplication
  ↓
Instruction-grouped train / validation / test split
  ↓
SFT formatting
  ↓
Token-length analysis
  ↓
Base-model benchmark
  ↓
QLoRA configuration
  ↓
Cloud GPU training
  ↓
Checkpoint persistence
  ↓
Checkpoint recovery
  ↓
Final LoRA adapter
  ↓
Deterministic benchmark
  ↓
Bootstrap confidence intervals
  ↓
Prompted baseline
  ↓
AI-assisted LLM-as-judge evaluation
  ↓
Qualitative failure analysis
  ↓
FastAPI inference
  ↓
Capability guardrails
  ↓
Gradio UI
```

---

# Dataset

**Dataset:**

```text
bitext/Bitext-customer-support-llm-chatbot-training-dataset
```

Original dataset size:

```text
26,872 examples
```

Columns:

```text
flags
instruction
category
intent
response
```

The dataset contains customer-support categories including:

* ACCOUNT
* ORDER
* REFUND
* INVOICE
* CONTACT
* PAYMENT
* FEEDBACK
* DELIVERY
* SHIPPING
* SUBSCRIPTION
* CANCEL

The dataset contains 27 customer-support intents.

---

# Data Preparation

## Data Quality

| Check                                | Result |
| ------------------------------------ | -----: |
| Total examples                       | 26,872 |
| Missing values                       |      0 |
| Duplicate instructions               |  2,237 |
| Duplicate instruction-response pairs |      0 |

The dataset contains repeated customer instructions with different valid responses.

Therefore, deduplication was performed using the **exact instruction-response pair** rather than removing every repeated instruction.

---

# Leakage Prevention

A major evaluation concern was preventing identical customer instructions from appearing across different splits.

The dataset was split at the **instruction-group level**.

This means that all examples belonging to the same instruction group remain within the same split.

Final split:

| Split      |   Examples |
| ---------- | ---------: |
| Train      |     21,500 |
| Validation |      2,713 |
| Test       |      2,659 |
| **Total**  | **26,872** |

This prevents an identical customer instruction from appearing in both training and evaluation data.

---

# SFT Formatting

Examples were converted into a consistent customer-support conversation format:

```text
Customer:
{instruction}

Assistant:
{response}
```

This format is used during supervised fine-tuning and benchmark generation.

---

# Token Analysis

Token-length analysis was performed before selecting the training sequence length.

Observed training distribution:

| Statistic | Tokens |
| --------- | -----: |
| Mean      | 140.95 |
| Median    |    119 |
| P90       |    233 |
| P95       |    275 |
| P99       |    380 |
| Maximum   |    502 |

Based on this analysis, a maximum sequence length of:

```text
512 tokens
```

was selected.

This covers the observed maximum sequence length while avoiding an unnecessarily large context allocation.

---

# Fine-Tuning

NexaTune uses:

```text
Qwen/Qwen3-1.7B-Base
        +
4-bit Quantization
        +
LoRA
        +
Supervised Fine-Tuning
```

The base model is loaded using 4-bit NF4 quantization.

LoRA adapters are applied to:

```text
q_proj
k_proj
v_proj
o_proj
```

Only the adapter parameters are trained.

Trainable parameters:

```text
6,422,528
```

Total parameters:

```text
1,726,997,504
```

Trainable percentage:

```text
0.3719%
```

This significantly reduces the number of parameters updated during training compared with full-model fine-tuning.

---

# Training Configuration

| Parameter               |                Value |
| ----------------------- | -------------------: |
| Base model              | Qwen/Qwen3-1.7B-Base |
| Epochs                  |                    1 |
| Max steps               |                   -1 |
| Learning rate           |                 2e-4 |
| Weight decay            |                 0.01 |
| Warmup steps            |                  100 |
| Training batch size     |                    1 |
| Evaluation batch size   |                    1 |
| Gradient accumulation   |                    8 |
| Maximum sequence length |                  512 |
| LoRA rank               |                   16 |
| LoRA alpha              |                   32 |
| LoRA dropout            |                 0.05 |
| LoRA targets            |  q/k/v/o projections |
| Quantization            |            4-bit NF4 |
| Double quantization     |              Enabled |
| Compute dtype           |                 FP16 |
| Gradient checkpointing  |              Enabled |
| Seed                    |                   42 |

The final cloud training run was performed using T4 GPU infrastructure.

The local development machine contains a GTX 1650 with 4 GB VRAM. Local trainer construction and actual training execution were also validated, but the local hardware is substantially slower for a full 21,500-example run.

---

# Checkpoint Recovery

Training was initially performed using cloud GPU infrastructure.

The training process was interrupted by a cloud-session disconnect after a checkpoint had been persisted.

Instead of restarting from zero, the checkpoint was transferred to another cloud environment and training was resumed.

The recovery flow was:

```text
Google Colab
     ↓
checkpoint-1250
     ↓
Kaggle Dataset
     ↓
Kaggle T4 GPU
     ↓
Resume Training
     ↓
Final LoRA Adapter
```

The final adapter was subsequently downloaded for local inference and evaluation.

Final adapter size:

```text
~24.5 MB
```

---

# Evaluation Methodology

The canonical benchmark uses a fixed held-out subset of the test set.

Configuration:

```text
Examples:       100
Selection:      test_dataset.shuffle(seed=42).select(range(100))
Decoding:       deterministic
do_sample:      False
max_new_tokens: 128
```

The same examples and generation configuration are used across the compared systems.

The benchmark evaluates:

* Base Qwen3-1.7B
* Prompted Base Qwen3-1.7B
* NexaTune Qwen3-1.7B + LoRA

Metrics include:

* ROUGE-L F1
* BERTScore F1
* average latency
* median latency
* generated token count
* per-example results
* qualitative failure analysis

The benchmark is deterministic under the documented generation configuration.

---

# Benchmark Results

## Base vs NexaTune

| Metric                   |    Base |   NexaTune |     Change |
| ------------------------ | ------: | ---------: | ---------: |
| ROUGE-L F1               |  0.2120 | **0.4082** |    +0.1962 |
| BERTScore F1             |  0.8574 | **0.9160** |    +0.0586 |
| Average latency          |  9.22 s |    15.45 s |     +67.6% |
| Median latency           | 10.95 s |    15.22 s |     +38.9% |
| Average generated tokens |   100.1 |       99.9 | ~unchanged |

Relative ROUGE-L improvement:

```text
~92.5%
```

Relative BERTScore F1 improvement:

```text
~6.8%
```

The benchmark shows substantially higher text-similarity metrics for NexaTune on this 100-example evaluation set.

NexaTune also has higher measured inference latency in this local benchmark.

These results should be interpreted as **benchmark-specific measurements**, not universal performance claims.

---

# Prompted Baseline

A prompted base-model baseline was also evaluated using the same 100 examples and deterministic generation configuration.

The system prompt instructs the base model to:

* behave as a professional customer-support assistant,
* answer clearly and concisely,
* avoid inventing order/tracking/refund information,
* request missing information,
* avoid claiming actions that were not confirmed.

This provides a more informative comparison than using only an unprompted base model.

The benchmark therefore distinguishes:

```text
Base
   ↓
Prompt Engineering
   ↓
Fine-Tuning
```

This helps separate gains attributable to simple prompting from gains associated with learned model adaptation.

---

# Statistical Validation

Paired bootstrap resampling was performed over the same benchmark examples.

Configuration:

```text
Resamples: 10,000
Seed:      42
Interval: 95%
Method:    Paired percentile bootstrap
```

For each metric, the per-example difference between the compared systems was calculated before bootstrap resampling.

## Base vs NexaTune

### ROUGE-L

```text
Mean difference: +0.1962
95% CI:          [+0.1751, +0.2185]
```

### BERTScore F1

```text
Mean difference: +0.0585
95% CI:          [+0.0535, +0.0638]
```

## Prompted Base vs NexaTune

### ROUGE-L

```text
Mean difference: +0.1854
95% CI:          [+0.1639, +0.2074]
```

### BERTScore F1

```text
Mean difference: +0.0485
95% CI:          [+0.0436, +0.0535]
```

The benchmark artifact also reports the bootstrap probability that the paired difference is greater than zero.

This is reported as:

```text
Bootstrap probability(diff > 0)
```

rather than being described as a conventional p-value.

Full results are stored in:

```text
results/benchmark/bootstrap_ci.json
```

---

# AI-Assisted LLM-as-Judge Evaluation

Automatic text metrics do not fully capture:

* correctness
* relevance
* helpfulness
* professionalism
* factuality
* unsupported claims

A separate 30-example AI-assisted LLM-as-judge evaluation was therefore performed using examples sampled from the benchmark.

This was **AI-assisted evaluation using an LLM judge**.

It should not be described as completed human-rater evaluation.

## Results

| Criterion       | Base | NexaTune |     Delta |
| --------------- | ---: | -------: | --------: |
| Correctness     | 3.50 |     3.83 |     +0.33 |
| Relevance       | 4.10 |     4.57 |     +0.47 |
| Helpfulness     | 3.43 |     3.97 |     +0.53 |
| Professionalism | 4.30 |     4.60 |     +0.30 |
| Factuality      | 4.37 |     3.33 | **-1.03** |
| Overall         | 3.94 |     4.06 |     +0.12 |

Scores use a 1–5 rubric.

Overall preference counts:

```text
NexaTune: 13
Base:     12
Tie:       5
```

Percentages:

```text
NexaTune: 43.3%
Base:     40.0%
Tie:       16.7%
```

The mean overall score margin was:

```text
+0.12
```

with a bootstrap 95% confidence interval of approximately:

```text
[-0.21, +0.49]
```

The interval crosses zero, so the small aggregate overall-score difference should not be presented as a statistically established overall preference.

An important finding is the **factuality regression**.

Although NexaTune scored higher on correctness, relevance, helpfulness, and professionalism in this AI-assisted evaluation, its factuality score was lower.

This finding directly informed the addition of capability-aware guardrails to the serving layer.

---

# Qualitative Failure Analysis

Several recurring base-model behaviors were observed during qualitative analysis.

## Example: Missing article information

For a request such as:

```text
I need to buy an aricle
```

the base model produced conversational artifacts/refusal-like behavior.

NexaTune more consistently treated the request as a customer-support interaction and asked for the missing article details.

---

## Example: Invoice workflow

For:

```text
where to download invoice #00108?
```

the base model generated an invented sandbox-style payment link.

NexaTune more closely followed the expected billing/invoice workflow without inventing an external link.

---

## Example: Shipping-address issue

For shipping-address problems, the base model sometimes stated that an issue had already been resolved despite having no evidence that such an action had occurred.

NexaTune more often requested the information necessary to continue the support interaction.

---

## Example: Package arrival

The base model sometimes invented:

* tracking numbers
* dates
* shipment status

when those details were not supplied.

NexaTune more often requested a tracking or order number.

---

# Known Model Limitation

Fine-tuning improved domain behavior but did not completely eliminate unsupported capability claims.

For example, NexaTune can sometimes generate language implying that it can:

* connect a customer to a human agent,
* perform a backend action,
* initiate an operational workflow,

even though the deployed system does not provide those capabilities.

This is an important distinction:

```text
Model behavior ≠ Actual system capability
```

The serving layer therefore adds an explicit capability guardrail.

---

# Capability Guardrails

NexaTune includes a post-generation capability guardrail.

The serving flow is:

```text
Customer Request
      ↓
Model Generation
      ↓
Capability Detection
      ↓
Guardrail Match?
   ↙          ↘
 Yes           No
 ↓             ↓
Safe          Original
Response      Response
 ↓             ↓
      API Response
```

The guardrail currently checks categories including:

```text
HUMAN_HANDOFF
REFUND_ACTION
ORDER_CANCELLATION
BACKEND_ACTION
UNSUPPORTED_COMMITMENT
```

The system prompt also explicitly informs the model that it cannot:

* connect the customer to a human agent,
* transfer a conversation,
* place or cancel orders,
* issue or initiate refunds,
* modify account/order information,
* confirm backend actions that were not actually performed.

The API retains the original generated response for traceability when a guardrail is triggered.

This creates a practical separation:

```text
LLM
 ↓
Generate language
 ↓
Guardrail
 ↓
Enforce application capabilities
```

Guardrails are not treated as a substitute for proper authorization or transactional backend validation.

---

# FastAPI Inference API

The model is exposed through FastAPI.

Architecture:

```text
POST /generate
      ↓
Pydantic Validation
      ↓
Inference Service
      ↓
Qwen3 Base + LoRA Adapter
      ↓
Capability Guardrail
      ↓
Structured API Response
```

---

## GET `/health`

Example:

```json
{
  "status": "healthy",
  "service": "NexaTune AI",
  "model_loaded": true
}
```

---

## GET `/model-info`

Example:

```json
{
  "base_model": "Qwen/Qwen3-1.7B-Base",
  "fine_tuning": "QLoRA",
  "lora_rank": 16,
  "quantization": "4-bit NF4",
  "domain": "Customer Support"
}
```

---

## POST `/generate`

Request:

```json
{
  "message": "I was charged twice for the same order."
}
```

The API validates that the customer message is between 1 and 4000 characters.

The response contains the generated answer together with guardrail traceability fields.

Conceptually:

```json
{
  "response": "...",
  "model": "NexaTune-Qwen3-1.7B",
  "guardrail_flagged": false,
  "guardrail_matches": [],
  "guardrail_categories": [],
  "original_response": "..."
}
```

When a capability violation is detected:

```text
Generated Response
       ↓
Guardrail Match
       ↓
Safe Replacement Response
       ↓
guardrail_flagged = true
```

The original response remains available internally through the API response structure for traceability.

---

# Gradio Interface

The Gradio interface provides a product-style frontend for NexaTune.

Current capabilities include:

* NexaTune branding
* API status monitoring
* live customer-support playground
* response generation
* quick test scenarios
* benchmark metrics
* evaluation results
* training pipeline presentation
* model configuration
* guardrail information
* error handling
* clear/reset functionality
* responsive layout

Architecture:

```text
Gradio :7860
     ↓
FastAPI :8000
     ↓
NexaTune Inference Service
     ↓
Qwen3 + LoRA Adapter
     ↓
Capability Guardrail
```

---

# Project Structure

```text
NexaTune-AI/
│
├── api/
│   ├── __init__.py
│   ├── capability_guardrail.py
│   ├── inference_service.py
│   ├── logging_config.py
│   ├── main.py
│   └── schemas.py
│
├── app/
│   └── gradio_app.py
│
├── configs/
│   └── llm_judge_rubric.json
│
├── data/
│   └── processed/
│       ├── customer_support/
│       └── customer_support_sft/
│
├── models/
│   └── final_adapter/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── chat_template.jinja
│       ├── README.md
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
│
├── notebooks/
│
├── outputs/
│
├── results/
│   ├── baseline/
│   └── benchmark/
│       ├── base_predictions.json
│       ├── benchmark_metrics.json
│       ├── bootstrap_ci.json
│       ├── finetuned_predictions.json
│       ├── human_eval_set.json
│       ├── human_evaluation.xlsx
│       ├── prompted_base_metrics.json
│       └── prompted_base_predictions.json
│
├── src/
│   ├── data/
│   │   ├── analyze_tokens.py
│   │   ├── download_dataset.py
│   │   ├── format_sft.py
│   │   ├── inspect_dataset.py
│   │   └── prepare_dataset.py
│   │
│   ├── evaluation/
│   │   ├── baseline_eval.py
│   │   ├── bootstrap_ci.py
│   │   ├── compare_models.py
│   │   ├── create_eval_set.py
│   │   ├── create_human_eval_sheet.py
│   │   ├── metrics.py
│   │   ├── prompted_baseline.py
│   │   └── run_benchmark.py
│   │
│   ├── inference/
│   │   ├── generate_response.py
│   │   ├── model.py
│   │   └── test_qwen.py
│   │
│   ├── training/
│   │   ├── config.py
│   │   ├── run_training.py
│   │   └── train_qlora.py
│   │
│   └── system_check.py
│
├── .gitignore
├── README.md
└── requirements.txt
```



---

# Local Setup

## 1. Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY>
cd NexaTune-AI
```

## 2. Create a virtual environment

```powershell
python -m venv .venv
```

## 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. Install dependencies

```powershell
pip install -r requirements.txt
```

## 5. Verify the adapter

The local inference service expects the final LoRA adapter at:

```text
models/final_adapter/
```

The adapter path can also be configured through:

```text
NEXATUNE_ADAPTER_PATH
```

The adapter weights are intentionally not committed to Git because model binaries are excluded through `.gitignore`.

---

# Running the Application

## Terminal 1 — FastAPI

```powershell
uvicorn api.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

---

## Terminal 2 — Gradio

```powershell
python app\gradio_app.py
```

UI:

```text
http://127.0.0.1:7860
```

Both processes should be running for the complete local application.

---

# Reproducible Benchmark

The canonical benchmark can be executed with:

```powershell
python -m src.evaluation.run_benchmark
```

The benchmark uses:

```text
Held-out test data
       ↓
shuffle(seed=42)
       ↓
100 examples
       ↓
Same prompt format
       ↓
Deterministic decoding
       ↓
Base + NexaTune evaluation
       ↓
Metrics
```

Primary artifacts:

```text
results/benchmark/
├── base_predictions.json
├── finetuned_predictions.json
└── benchmark_metrics.json
```

Additional evaluation artifacts include:

```text
results/benchmark/
├── bootstrap_ci.json
├── prompted_base_predictions.json
├── prompted_base_metrics.json
├── human_eval_set.json
└── human_evaluation.xlsx
```

The benchmark configuration uses:

```text
Seed:           42
Examples:       100
do_sample:      False
max_new_tokens: 128
```

---

# Training Entry Point

The canonical training entry point is:

```powershell
python -m src.training.run_training
```

The training architecture was explicitly separated into:

```text
train_qlora.py
    ↓
build_trainer()
    ↓
returns SFTTrainer
```

and:

```text
run_training.py
    ↓
build_trainer()
    ↓
trainer.train()
    ↓
save final adapter
```

Importing the training modules does **not** automatically start training.

Training is started only through the explicit entry point.

The final adapter is saved to:

```text
outputs/nexatune-qlora/final_adapter/
```

The validated production/demo adapter used by the current local application is stored separately under:

```text
models/final_adapter/
```

---

# Model Card

## Model

```text
Qwen/Qwen3-1.7B-Base
        +
NexaTune LoRA Adapter
```

## Fine-Tuning Method

```text
QLoRA
LoRA
PEFT
Supervised Fine-Tuning
```

## Quantization

```text
4-bit NF4
Double Quantization
FP16 Compute
```

## Domain

```text
Customer Support
```

## Intended Use

NexaTune is intended for experimentation, research, and portfolio demonstration of:

* parameter-efficient fine-tuning
* domain adaptation
* LLM evaluation
* model serving
* customer-support response generation
* model behavior analysis
* capability-aware inference

## Out of Scope

The model should not be treated as:

* an authoritative account system
* a payment processor
* an order-management system
* a refund processing system
* a human-agent connection service
* a source of real customer account information
* an autonomous system authorized to perform transactional actions

The model does not inherently have access to customer accounts, orders, payments, or external support systems.

---

# Limitations

## Unsupported Capability Claims

The model can still generate language implying capabilities that are not connected to actual tools.

The serving layer mitigates several known patterns using explicit capability guardrails.

---

## Factuality

AI-assisted evaluation showed a lower factuality score for NexaTune than the base model on the 30-example evaluation set.

This is an important limitation and demonstrates why automatic text-similarity improvements should not be treated as complete evidence of quality.

---

## Latency

The 100-example local benchmark measured higher latency for NexaTune:

```text
Base:     9.22 s average
NexaTune: 15.45 s average
```

The benchmark environment is local hardware and should not be treated as a universal production-latency measurement.

---

## Automatic Metrics

ROUGE-L and BERTScore measure similarity and semantic alignment but do not fully capture:

* factuality
* unsupported claims
* helpfulness
* safety
* capability boundaries
* real-world task success

This is why NexaTune combines automatic metrics, statistical validation, qualitative analysis, and AI-assisted evaluation.

---

## Benchmark Size

The headline benchmark contains:

```text
100 held-out examples
```

The AI-assisted LLM-as-judge evaluation contains:

```text
30 examples
```

These are useful evaluation sets but are not sufficient to establish universal model performance.

---

## Dataset Limitations

The fine-tuned model inherits limitations, biases, and response patterns from its training dataset.

---

# Future Work

## Engineering Hardening

* [ ] Unit tests for API and inference components
* [ ] API integration tests
* [ ] Stronger environment-based configuration
* [ ] Production authentication
* [ ] Rate limiting
* [ ] Expanded guardrail test coverage
* [ ] Better observability
* [ ] Production deployment

## Evaluation Extensions

* [ ] Larger held-out benchmark
* [ ] Independent human-rater study
* [ ] Multiple LLM judges
* [ ] Judge-model reproducibility study
* [ ] Error taxonomy by intent/category
* [ ] Calibration and hallucination analysis
* [ ] More detailed latency profiling

## Research Extensions

* [ ] Compare LoRA ranks
* [ ] Compare learning rates
* [ ] Compare different base models
* [ ] Compare QLoRA configurations
* [ ] Larger training dataset
* [ ] Tool-aware customer-support agent
* [ ] RAG + fine-tuning hybrid architecture
* [ ] Inference optimization
* [ ] Quantization/performance comparison

---

# Interview Talking Points

## Why QLoRA?

QLoRA allows parameter-efficient fine-tuning while keeping the base model quantized.

This reduces the memory requirements compared with full-parameter fine-tuning and makes adaptation of larger models more accessible on constrained hardware.

---

## Why LoRA?

LoRA learns low-rank adapter matrices instead of updating the entire base model.

In NexaTune:

```text
Total parameters:      1.727B
Trainable parameters:  6.42M
Trainable percentage:  0.3719%
```

---

## Why rank 16?

LoRA rank controls the capacity of the adapter.

Rank 16 was selected as a practical configuration balancing adaptation capacity and parameter efficiency for the available compute budget.

---

## Why q/k/v/o projections?

The attention projection layers are important components of transformer attention behavior and are commonly targeted during LoRA adaptation.

NexaTune targets:

```text
q_proj
k_proj
v_proj
o_proj
```

---

## Why 4-bit NF4?

The base model is quantized to reduce memory requirements during QLoRA training.

NF4 is designed for representing normally distributed neural-network weights efficiently in low precision.

---

## How did you prevent evaluation leakage?

The dataset was split using instruction-level grouping so identical customer instructions could not cross the train, validation, and test boundaries.

This is important because ordinary random row splitting could place repeated instructions in both training and evaluation data.

---

## Why evaluate against the base model?

Without a baseline, a fine-tuned score has limited meaning.

NexaTune therefore compares the adapted model against the original base model using the same held-out examples and generation configuration.

---

## Why add a prompted baseline?

A prompted baseline helps distinguish improvements caused by simple instruction prompting from improvements associated with fine-tuning.

The evaluation therefore considers:

```text
Base
Prompted Base
NexaTune
```

---

## Why report latency?

A model can improve output quality while becoming slower.

Production evaluation therefore needs to consider:

* quality
* latency
* memory
* cost
* reliability
* safety

---

## Why use bootstrap confidence intervals?

A single average metric does not show how stable the observed difference is.

Paired bootstrap resampling estimates uncertainty around the per-example metric difference while preserving the pairing between systems.

NexaTune uses:

```text
10,000 resamples
95% percentile confidence interval
seed 42
```

---

## Why use LLM-as-judge evaluation?

Text-similarity metrics cannot fully measure qualities such as helpfulness, relevance, professionalism, and factuality.

The AI-assisted evaluation adds a structured 1–5 rubric across those dimensions.

However, because it uses an LLM judge rather than independent human raters, the results should be treated as an additional evaluation signal rather than definitive human preference evidence.

---

## What did the evaluation reveal?

The automatic benchmark showed higher ROUGE-L and BERTScore for NexaTune.

The AI-assisted evaluation showed higher scores for:

* correctness
* relevance
* helpfulness
* professionalism

but a lower factuality score.

This discrepancy is important because it demonstrates that improving text similarity does not automatically eliminate unsupported claims.

---

## Why add guardrails?

The model can generate language describing capabilities that the application does not actually provide.

The guardrail layer therefore separates:

```text
What the model says
```

from:

```text
What the application is actually authorized to do
```

The guardrail detects known unsupported capability patterns and replaces unsafe/unsupported responses with a capability-safe response.

---

## Fine-Tuning vs RAG?

Fine-tuning changes learned model behavior.

RAG supplies external knowledge at inference time.

A practical production architecture could combine:

```text
Fine-Tuned Model
       +
RAG
       +
Tools / APIs
       +
Guardrails
       +
Evaluation
```

NexaTune intentionally focuses on the fine-tuning component.

---

## What would you improve next?

The next improvements would focus on:

1. larger evaluation sets,
2. independent human evaluation,
3. stronger factuality measurement,
4. expanded guardrail coverage,
5. production tool integration,
6. RAG + fine-tuning experimentation,
7. inference optimization,
8. production deployment and observability.

---

# Project Status

## Implemented

* Dataset inspection
* Data quality analysis
* Exact instruction-response pair deduplication
* Instruction-grouped splitting
* SFT formatting
* Token-length analysis
* QLoRA configuration
* LoRA fine-tuning
* Cloud checkpoint recovery
* Final adapter generation
* Deterministic 100-example benchmark
* Prompted base-model baseline
* ROUGE-L evaluation
* BERTScore evaluation
* Latency evaluation
* Generated-token analysis
* Paired bootstrap confidence intervals
* AI-assisted LLM-as-judge evaluation
* Qualitative failure analysis
* FastAPI inference API
* Structured logging
* Capability guardrails
* Model metadata endpoint
* Gradio product interface

## In Progress / Cleanup

* Repository cleanup
* Removal of legacy development files
* Expanded automated testing
* Documentation refinement
* Production hardening

## Future

* Larger evaluation study
* Independent human evaluation
* Tool-aware support workflows
* RAG integration
* Production deployment
* Inference optimization
* Expanded observability

---

# Key Result

NexaTune demonstrates a complete **model customization and evaluation workflow** rather than only a fine-tuning script.

The project connects:

```text
Data Engineering
      ↓
Leakage Prevention
      ↓
LLM Fine-Tuning
      ↓
Evaluation
      ↓
Statistical Validation
      ↓
Failure Analysis
      ↓
Safety / Guardrails
      ↓
API Serving
      ↓
Product Interface
```

The central result is not simply that the fine-tuned model produced a higher benchmark score.

The more important engineering result is that the project establishes a measurable workflow for answering:

> **Did fine-tuning change model behavior, where did it help, where did it fail, and how should those failures be handled in an application?**
