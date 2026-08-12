AI-ML Engineer - Assignment

**Scenario**

MindMap Digital wants a small, cost-efficient open-weight model
fine-tuned to extract structured metadata from enterprise documents
(invoices, purchase orders, or similar) into strict JSON matching a
fixed schema, supporting both English and Hindi source documents. The
model needs to run affordably --- no multi-GPU cluster --- integrate
into an existing Django/FastAPI stack, and pass automated quality gates
before anything reaches production.

**Target schema (example):**

json

Plain Text

{\"invoice_number\": \"string\",\"vendor_name\":
\"string\",\"invoice_date\": \"YYYY-MM-DD\",\"line_items\":
\[{\"description\": \"string\", \"quantity\": \"number\",
\"unit_price\": \"number\"}\],\"subtotal\": \"number\",\"tax_amount\":
\"number\",\"total_amount\": \"number\",\"currency\": \"string (ISO
4217)\"}

**What You\'ll Receive**

- A synthetic training set (\~300--500 document → JSON pairs, mixed
  English/Hindi)

- A held-out golden test set (\~50 examples) --- do not train on this

- A small general-capability benchmark subset, for
  catastrophic-forgetting checks

- A base repo skeleton with requirements.txt and empty scripts/, tests/,
  docs/ folders

- Suggested (not mandatory) base models: Qwen2.5-1.5B-Instruct,
  Llama-3.2-3B-Instruct, Gemma-2-2b-it --- propose a different one and
  justify it if you prefer, including anything newer than this brief

------------------------------------------------------------------------

**Day 1 --- Model Selection & Fine-Tuning**

**1.1 Base model memo** *(\~30--45 min)*Compare two candidate base
models on: license terms, parameter count vs. hardware budget,
multilingual/Hindi capability, and tokenizer/chat-template quirks. One
page, written *before* you start training.

**1.2 Fine-tuning runs** *(\~4--6 hrs)*Fine-tune using LoRA and at least
one of QLoRA/DoRA via Transformers/PEFT/TRL (Unsloth or Axolotl welcome,
not required). For every run, log:

  -----------------------------------------------------------------------------------------------------------
  **Run   **Method**   **rank   **alpha**   **target    **LR /       **seq   **batch   **grad    **epochs**
  ID**                 (r)**                modules**   schedule**   len**   size**    accum**   
  ------- ------------ -------- ----------- ----------- ------------ ------- --------- --------- ------------

  -----------------------------------------------------------------------------------------------------------

If you hit a training instability, tokenization mismatch, or
structured-output failure, don\'t just fix it --- document what you saw,
how you diagnosed it, and what changed.

**Day 1 deliverables:** training scripts/notebooks, hyperparameter log,
experiment-tracking export (MLflow or equivalent), a short training
diary.

------------------------------------------------------------------------

**Day 2 --- Evaluation, Certification & Forgetting Detection**

**2.1 Acceptance criteria --- write these first** *(\~30 min)*Before
evaluating anything, commit a short acceptance_criteria.md stating your
target thresholds (e.g., ≥90% JSON schema validity, ≥0.85 field-level
F1). Commit this *before* you run evaluation.

**2.2 Automated evaluation pipeline** *(\~3--4 hrs)*Score both your
fine-tuned model and the untouched base model on:

- JSON schema validity rate

- Field-level Exact Match, Precision, Recall, F1

- A general-capability benchmark subset (base vs. fine-tuned) --- your
  catastrophic-forgetting check

**2.3 Human review** *(\~1 hr)*Manually review \~20 outputs against a
short rubric you define (correctness, faithfulness, formatting). Note
anywhere your automated metrics and your own judgment disagree, and why.

**2.4 CI quality gate** *(\~1 hr)*A script or CI config that fails the
build if any metric falls below your 2.1 thresholds.

**2.5 Model card** *(\~1 hr)*Capabilities, known limitations/edge cases,
target usage, exact validation numbers, and a clear go/no-go
recommendation with evidence.

**Day 2 deliverables:** acceptance_criteria.md, eval scripts, results
table (base vs. fine-tuned vs. general benchmark), CI config,
human-review notes, model card.

------------------------------------------------------------------------

**Day 3 --- Optimization, Serving & Integration**

**3.1 Quantization & benchmarking** *(\~2--3 hrs)*Quantize your merged
model (GGUF, AWQ, GPTQ, or bitsandbytes --- your choice) and serve it
with vLLM, SGLang, or llama.cpp. Benchmark TTFT, throughput, and memory
footprint, quantized vs. unquantized, on a fixed prompt set. State your
methodology, not just the numbers.

**3.2 API service** *(\~2--3 hrs)*Wrap your model in a FastAPI endpoint
that\'s OpenAI-compatible (/v1/chat/completions-style) with token
streaming. Wrap the quantization step as an async task (Celery + Redis,
or a documented stand-in) rather than a blocking call.

**3.3 Integration design (write-up only)** *(\~1--1.5 hrs)*A short
design doc: one MCP-style connector interface (pick any target system
from the JD\'s list) and a small proof-of-concept for hybrid
sparse+dense retrieval over a tiny sample corpus. Full implementation
not required --- we\'re evaluating the design thinking, particularly how
you separate deterministic business rules from probabilistic model
output.

**Day 3 deliverables:** quantization script + benchmark table, FastAPI
service + README, integration design doc.

------------------------------------------------------------------------

**Code Quality Bar (applies throughout --- not a separate day)**

- Type-annotated Python, clean under mypy

- ruff + black clean, bandit with no unaddressed findings

- pytest coverage for your evaluation logic and API endpoint, at minimum

- Real commit history with conventional commit messages --- please
  don\'t squash before submitting

**Submission**

A Git repository containing everything above, plus:

- README.md --- how to run each piece

- SUMMARY.md --- trade-offs you made, what you\'d do differently with
  another week, and anything you know is a weak spot in your submission
