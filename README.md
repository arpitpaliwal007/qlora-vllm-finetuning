# QLoRA fine-tuning → evaluation → release gate → quantized vLLM serving

A reproducible, single-GPU project that fine-tunes **Qwen2.5-1.5B-Instruct** with **QLoRA** for
grounded question answering (SQuAD v1.1), evaluates the base / fine-tuned / quantized models on the
same held-out set, **blocks or approves a release** with a fail-closed policy gate, and serves the
approved **W4A16** checkpoint locally with **vLLM**.

## Status: what has and has not been executed

| Stage | Executed? | Where / evidence |
|---|---|---|
| Data pipeline (validate, normalize, dedup, grouped split, manifest) | ✅ on a synthetic fixture (CPU) | `tests/test_prepare_data.py` |
| Chat templating + assistant-only loss masking + truncation policy | ✅ on a tiny tokenizer (CPU) | `tests/test_dataset.py` |
| `train.py` → resume → `merge_adapter.py` → `evaluate` → `compare` | ✅ end-to-end on a **tiny random model** (CPU, no NF4) | `tests/test_train_pipeline_cpu.py` |
| Metrics (EM/F1 best-over-references, paired bootstrap) | ✅ unit tested | `tests/test_metrics.py` |
| Release gate + promotion/rollback | ✅ on **synthetic** reports (CPU) | `tests/test_release_gate.py` |
| SQuAD download, real QLoRA run, NF4 loading on GPU | ❌ **not run** (no GPU, no Hub access in the dev sandbox) | run `scripts/run_gpu_pipeline.sh smoke` |
| llm-compressor W4A16 quantization, vLLM serving, endpoint eval/bench | ❌ **not run** | same script |
| Promotion of a real artifact / on-prem deployment | ❌ **not run** | same script |

**There are no measured model results in this repository.** Every number in the tests is a labelled
synthetic fixture. The GPU steps are implemented and documented but still need to be executed on a
machine with an NVIDIA GPU (see [GPU setup](#gpu-setup-and-execution)).

## Task and architecture

**Task.** Grounded extractive QA: given a Wikipedia paragraph and a question, answer with the shortest
span from the paragraph. Dataset: [`rajpurkar/squad`](https://huggingface.co/datasets/rajpurkar/squad)
(SQuAD v1.1, config `plain_text`, Hub revision `7b6d24c440a36b6815f21b70d25016731768db1f`,
licence CC BY-SA 4.0; 87,599 train / 10,570 validation records with fields `id, title, context,
question, answers{text[], answer_start[]}`). Metrics: SQuAD-normalized exact match and token F1, best
over all reference answers.

**Model.** `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0), pinned to Hub revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Inspection found no incompatibility with the task, so the
original default is retained.

```
SQuAD (HF Hub, pinned revision)
   │  python -m qlora_vllm.prepare_data            validate → NFKC/whitespace normalize → exact + near dedup
   ▼                                              → title/near-dup-context grouped split → contamination check
data/squad_qa/{train,validation,test,calibration}.jsonl + manifest.json (hashes, rejections, checks)
   │  scripts/train.py --config configs/qlora.yaml  NF4 base + LoRA, assistant-only loss, fp16 on T4
   ▼
artifacts/qlora-adapter/   (adapter + training_metadata.json + artifact_manifest.json)
   │  scripts/merge_adapter.py
   ▼
artifacts/merged-*/        (fp16 merged model + manifest, parent = adapter id)
   │  scripts/quantize_w4a16.py   llm-compressor GPTQ W4A16, calibration = train subset
   ▼
artifacts/quantized-w4a16-*/   (compressed-tensors checkpoint + manifest, parent = merged id)
   │  scripts/serve_vllm.sh start <dir>   vLLM, --served-model-name <artifact_id>
   ▼
candidate endpoint ──► evaluate (openai backend) + bench ──► release_gate ──► promote ──► deploy/approved/current
                                                                   ▲
base + fine-tuned evaluated with the HF backend on the same test.jsonl / prompts / decoding ┘
```

Every model directory carries an `artifact_manifest.json` whose `artifact_id` is a SHA-256 over its
weights and configs, plus its parent artifact id. Reports reference these ids, the test-file hash and an
`eval_config_hash`; the gate and the promotion command refuse anything that does not line up.

## What existed and what was added

The original repository was a small starter: `scripts/train.py` (TRL `SFTTrainer`), `merge_adapter.py`,
`quantize_gguf.py`, a YAML config loader, a two-function `dataset.py` and one trivial test.

Checked against the current libraries (Transformers 5.22 / TRL 1.17 / PEFT 0.21 / bitsandbytes 0.50 were the latest on
PyPI as of this change), the training script no longer ran:

* `SFTTrainer(dataset_text_field=..., max_seq_length=...)` — both kwargs were removed from TRL's trainer
  (they moved to `SFTConfig.max_length`).
* `TrainingArguments(warmup_ratio=...)` — removed in Transformers 5 (`warmup_steps` now accepts a ratio).
* `from_pretrained(torch_dtype=...)` — renamed to `dtype` in Transformers 5.
* The config assumed `bf16: true`, which a T4 cannot do.

Changes:

| Component | Before | Now |
|---|---|---|
| Training | TRL SFTTrainer, full-sequence loss, silent truncation | `transformers.Trainer` + PEFT with explicit assistant-only labels, `drop`/`error` policy for over-long examples, hardware-aware precision, metadata + manifest, resume via `--resume` |
| Data | `cp data/sample.jsonl data/train.jsonl` | `qlora_vllm.prepare_data` (SQuAD → chat JSONL, dedup, grouped splits, manifest) |
| Evaluation | none | `qlora_vllm.evaluate` (HF + OpenAI-compatible backends, per-example scores, bootstrap CIs), `qlora_vllm.bench` |
| Release gate | none | `qlora_vllm.release_gate` + `configs/release_policy.yaml`, `qlora_vllm.promote` with rollback |
| Quantization | GGUF helper only | `scripts/quantize_w4a16.py` (llm-compressor → compressed-tensors, served by vLLM); GGUF helper kept as an unverified alternative for llama.cpp/Ollama |
| Serving | documented `vllm serve` | `scripts/serve_vllm.sh` (artifact-id model name, health wait, load-time record), `scripts/smoke_test_api.py` |
| Tests/CI | 1 test | 35 CPU tests, `ci.yml`; manual self-hosted `gpu-pipeline.yml` |

Why `Trainer` instead of TRL: TRL's `assistant_only_loss` requires `{% generation %}` markers in the chat
template, which Qwen2.5's template does not have, and TRL truncates over-long sequences from the right,
which would delete the supervised answer. Doing the tokenization in ~40 lines (`dataset.encode_example`)
makes the mask exact and testable, and removes a fast-moving dependency.

## Setup

CPU (tests, data tooling, gate):

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
pip install -r requirements-cpu.txt && pip install -e .
pytest -q            # 35 tests, ~40 s on 2 CPU cores
ruff check src scripts tests
```

### GPU setup and execution

Hardware: one NVIDIA GPU with ≥ 15 GB VRAM (the configs target a **T4**, compute capability 7.5,
**FP16** — BF16 is neither available nor assumed), ~40 GB free disk, an NVIDIA driver compatible with the
CUDA runtime of the pinned wheels, outbound access to huggingface.co and PyPI.

Two virtualenvs are required because vLLM 0.32.0 pins `compressed-tensors==0.17.0` while
llm-compressor 0.15.0 pins `compressed-tensors==0.20.1`:

```bash
python -m venv .venv-train && .venv-train/bin/pip install -r requirements-gpu-train.txt -e .
python -m venv .venv-serve && .venv-serve/bin/pip install -r requirements-gpu-serve.txt -e .
```

Run everything (smoke first — ~20 optimizer steps on 256 examples, validates the whole path):

```bash
scripts/run_gpu_pipeline.sh smoke
scripts/run_gpu_pipeline.sh full      # 20k examples × 2 epochs: several hours on a T4
```

The script is idempotent per step (delete a step's output directory to redo it) and runs, in order,
the commands below.

## Step-by-step commands

### 1. Dataset preparation

```bash
python -m qlora_vllm.prepare_data --source hf --out data/squad_qa --profile full    # or --profile smoke
# offline alternative with the official JSON files:
python -m qlora_vllm.prepare_data --source local --train-json train-v1.1.json --test-json dev-v1.1.json --out data/squad_qa
```

Methodology (all recorded in `data/squad_qa/manifest.json`):

* **Validation** – every record needs non-empty `id/title/context/question`, at least one non-empty answer,
  and each `answer_start` span must match the context. Rejection reasons are counted and listed per id
  in `rejections.json`.
* **Normalization** – Unicode NFKC + whitespace collapse applied identically to title, context, question
  and answers; each normalized answer is re-verified as a substring of the normalized context (otherwise
  the record is rejected, so no answer is damaged silently). All reference answers are kept.
* **Deduplication** – exact duplicates on normalized `(context, question)`; near-duplicate questions on
  the same context (token Jaccard ≥ 0.9) are rejected; near-duplicate contexts (MinHash/LSH, 3-gram
  Jaccard ≥ 0.8) are *not* dropped but merged into one split group.
* **Splits** – units are **whole groups** (Wikipedia article title ∪ near-duplicate-context clusters),
  shuffled with the seed, so no context or article appears in two splits. The **test** split comes from
  SQuAD's official validation split (multiple reference answers per question) after removing any record
  sharing a title or context with the train pool; **validation** and **train** come from the SQuAD train
  split. `calibration.jsonl` is a seeded sample of **train** only. The manifest records a contamination
  check (title / context / group / question-context overlap for every pair of splits) and the command
  aborts if it fails.
* Profiles: `smoke` = 256/64/64 (+32 calibration), `full` = 20,000/1,000/1,000 (+256). Sizes are
  overridable with `--train-size` etc.
* Output records: `{id, title, context, question, answers[], messages[]}`; `messages` is the chat
  conversation (system prompt in `qlora_vllm/prompting.py`, user = context + question, assistant = first
  reference answer). Evaluation reuses the `messages` prompt and scores against all `answers`.

### 2. QLoRA training

```bash
python scripts/train.py --config configs/qlora_smoke.yaml      # 20 steps
python scripts/train.py --config configs/qlora.yaml             # full run
python scripts/train.py --config configs/qlora.yaml --resume    # continue from the last checkpoint
```

* NF4 double-quant base (`bitsandbytes`), LoRA r=32/α=64 on all attention and MLP projections, gradient
  checkpointing (non-reentrant), paged AdamW 32-bit, cosine schedule, seeded (`seed`, `data_seed`,
  `set_seed`).
* `precision: auto` → fp16 on a T4, bf16 only where every visible GPU supports it; an explicit `bf16`
  on unsupported hardware is an error, not a silent downgrade.
* Loss is computed only on the assistant turn: the prompt is rendered with `add_generation_prompt=True`
  and tokenized separately from the completion, so the label boundary is exact.
* `max_seq_length` / `on_too_long`: an example longer than the limit is **dropped and counted**
  (`drop`) or aborts the run (`error`); it is never truncated. SQuAD prompts are < 600 tokens, so with
  `max_seq_length: 1024` the expected drop count is 0 — check `training_metadata.json → encoding`.
* Outputs: adapter + tokenizer, `checkpoint-N/` dirs, `training_metadata.json` (model + resolved
  revision, dataset hash, full config, package versions, hardware, encoding stats, metrics, resume
  source) and `artifact_manifest.json`.

### 3. Merge

```bash
python scripts/merge_adapter.py --adapter-dir artifacts/qlora-adapter --output-dir artifacts/merged-full
```

Merges into the fp16 base (base model/revision taken from the adapter manifest). Merging an adapter
trained against NF4 weights into fp16 weights is standard but not bit-identical to the training-time
model, which is why the merged model is evaluated as its own variant.

### 4. Evaluation

```bash
python -m qlora_vllm.evaluate run --backend hf --variant base --model Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --test data/squad_qa/test.jsonl --out reports/base
python -m qlora_vllm.evaluate run --backend hf --variant finetuned --model artifacts/merged-full \
    --test data/squad_qa/test.jsonl --out reports/finetuned
python -m qlora_vllm.evaluate run --backend openai --variant quantized --base-url http://localhost:8001/v1 \
    --candidate-dir artifacts/quantized-w4a16-full --test data/squad_qa/test.jsonl --out reports/quantized
python -m qlora_vllm.evaluate compare --report base=reports/base/results.json \
    --report finetuned=reports/finetuned/results.json --report quantized=reports/quantized/results.json --out reports/comparison
```

Same examples (file sha256 + `--limit` are part of `eval_config_hash`), same prompt messages, greedy
decoding, `max_new_tokens 48`. Each run writes `predictions.jsonl` (id, prediction, references, EM, F1,
error, output tokens, latency) and `results.json` (metrics incl. empty-response and error rates, model
identity, dataset hashes, eval config, timing, hardware, package versions). `compare` refuses reports with
different test files or eval configs and writes `comparison.md/json` with paired bootstrap 95% CIs
(10,000 resamples) for `finetuned−base` and `quantized−finetuned`.

Note the engine confound: base/fine-tuned are scored through `transformers.generate`, the quantized
candidate through vLLM. To remove it, serve the merged fp16 model with `scripts/serve_vllm.sh` and
evaluate it with `--backend openai` too (`--candidate-dir artifacts/merged-full`).

### 5. Quantization and serving

```bash
python scripts/quantize_w4a16.py --model-dir artifacts/merged-full \
    --calibration data/squad_qa/calibration.jsonl --output-dir artifacts/quantized-w4a16-full
scripts/serve_vllm.sh start artifacts/quantized-w4a16-full 8001        # waits for /health, writes serve_status.json
python scripts/smoke_test_api.py --base-url http://localhost:8001/v1 --candidate-dir artifacts/quantized-w4a16-full
python -m qlora_vllm.bench --base-url http://localhost:8001/v1 --candidate-dir artifacts/quantized-w4a16-full \
    --prompts data/squad_qa/test.jsonl --requests 128 --concurrency 4 --warmup 8 --out reports/bench.json
scripts/serve_vllm.sh stop artifacts/quantized-w4a16-full
```

Route: **llm-compressor GPTQ W4A16 (group 128) → compressed-tensors → vLLM**. vLLM's hardware table
lists GPTQ/AWQ INT4 and bitsandbytes as supported on Turing (T4) and FP8 as unsupported; llm-compressor
is the vLLM project's own quantizer, so the checkpoint format is the engine's native one. QLoRA's NF4
weights are a *training-time* representation and are not exported — the serving checkpoint is produced
from the merged fp16 model. Calibration uses `calibration.jsonl` (seeded train subset; file hash and
sample count are written into the quantized artifact's manifest). The server is started with
`--served-model-name <artifact_id>`, so the endpoint itself identifies the artifact it serves and the
evaluation/benchmark reports record whether that matched (`identity_verified`).

The benchmark distinguishes model-load time (from `serve_status.json`, measured by the serve script),
request latency and time-to-first-token (streaming; p50/p90/p95/p99), aggregate requests/s and output
tokens/s at the given concurrency, prompt/output token lengths (from `usage`), and GPU memory
(`nvidia-smi`, before/after). Anything that cannot be measured is reported as `null` with a note.

`scripts/quantize_gguf.py` (llama.cpp GGUF) is retained for CPU/Ollama deployments but is outside the
gated path and has not been exercised.

### 6. Release gate and promotion

```bash
python -m qlora_vllm.release_gate --policy configs/release_policy.yaml \
    --candidate reports/quantized/results.json --candidate-dir artifacts/quantized-w4a16-full \
    --finetuned reports/finetuned/results.json --bench reports/bench.json \
    [--baseline deploy/approved/current/release/results.json] --out reports/release_decision.json --verify-weights
echo $?    # 0 = PASS, 1 = FAIL
python -m qlora_vllm.promote --decision reports/release_decision.json \
    --candidate-dir artifacts/quantized-w4a16-full --approved-dir deploy/approved
python -m qlora_vllm.promote --rollback --approved-dir deploy/approved
scripts/serve_vllm.sh start deploy/approved/current 8000
```

Checks (see `configs/release_policy.yaml`; **the shipped thresholds are illustrative and must be
calibrated against measured results**): minimum F1/EM; max F1/EM regression vs the approved baseline;
max F1/EM degradation of the quantized candidate vs the unquantized fine-tuned model; max empty-response
and error rates; benchmark p95 latency, p95 TTFT, output tokens/s, request count and error rate (when
`performance.required`). Evidence checks fail closed: missing/malformed reports, reports older than
`max_report_age_hours` or older than the artifact they describe, a candidate report or benchmark whose
artifact id ≠ the candidate manifest (or whose endpoint identity was not verified), a fine-tuned report
that is not the candidate's recorded parent, reports on different test files or eval configs, and
weights on disk that no longer hash to their manifest (`--verify-weights`).

`reports/release_decision.json` records PASS/FAIL, every check with observed value, threshold and
reason, candidate identity (artifact id, kind, parent), dataset hash, eval config and report references.
`promote` requires a PASS decision younger than 72 h whose artifact id matches the re-hashed candidate,
copies the artifact and its evidence to `deploy/approved/current/`, keeps the previous approval under
`previous-<timestamp>/` (default: 2 kept) and logs to `history.json`. A candidate server may run for
evaluation at any time; only `promote` touches the approved slot.

## Tests and CI

`pytest -q` runs 35 CPU tests: data validation and rejection accounting, Unicode-safe normalization,
exact/near dedup, grouped-split determinism and leakage checks, official-dev-as-test handling, loss
masking and truncation policy on a tiny ChatML tokenizer, metrics and bootstrap, artifact hashing and
tamper detection, precision selection (CPU/T4/A100), and a real `train.py → --resume → merge → evaluate
→ compare` run on a tiny random Qwen2 model, plus the endpoint evaluator, streaming benchmark parser and API
smoke test against a mock OpenAI-compatible server. The release-gate suite proves a consistent candidate passes
and that quality floors, baseline regressions, quantization degradation, empty/error rates, missing or
malformed evidence, stale reports, mismatched artifacts/lineage/test sets, failed benchmarks and
tampered weights are rejected, and that promotion/rollback behave.

Fixtures (`tests/fixtures/squad_mini.json`, the tiny model in `tests/conftest.py`, and the invented
reports in `tests/test_release_gate.py`) are labelled synthetic. **They are not GPU validation and not
release evidence.** `.github/workflows/ci.yml` runs lint + tests on a CPU runner;
`.github/workflows/gpu-pipeline.yml` is a manual job for a self-hosted GPU runner.

## Reproducibility

Pinned: dataset Hub revision, model Hub revision, package versions (`requirements-*.txt`; recorded again
in every manifest/report), seeds for splitting, calibration sampling, training and bootstrap. Dataset
identity = `manifest.json → dataset_hash` (sha256 over the split files); model identity = artifact ids;
evaluation identity = `eval_config_hash`. CUDA kernels (bitsandbytes, vLLM) are not bit-deterministic
across hardware, so reruns reproduce the *procedure* and should match within bootstrap CIs, not to the bit.

## Known limitations and open steps

* **No GPU run has happened yet.** The author's environment had no GPU and no huggingface.co access,
  so real training, quantization, serving, benchmark numbers and a real release decision are all pending
  `scripts/run_gpu_pipeline.sh smoke` followed by `full`.
* The CPU suite was run on the exact train-environment pins (`torch 2.13.0`, `transformers 5.19.0`,
  `peft 0.21.2`, `datasets 5.1.0`, `accelerate 1.15.0`) and also on `transformers 5.22.1`; the
  CUDA-only pieces (`bitsandbytes` NF4 loading, `llmcompressor`, `vllm`) import-checked at most and
  are exercised only by the GPU script.
* The compressed-tensors checkpoint is written by compressed-tensors 0.20.1 and read by vLLM's 0.17.0;
  the format is designed to be backward-readable but this specific pair is unverified. If `vllm serve`
  rejects the checkpoint, quantize in a separate env with `llmcompressor==0.12.0` (compressed-tensors
  0.17.1, needs `transformers<=5.10.1`, `torch<=2.12`) or upgrade vLLM.
* vLLM on a T4 cannot use FlashAttention-2 (Ampere+); it falls back to another attention backend. The
  `--dtype float16`, `--max-model-len 2048`, `--gpu-memory-utilization 0.85` defaults in
  `serve_vllm.sh` are starting points.
* The quantized candidate is scored through vLLM while base/fine-tuned are scored through
  `transformers.generate`; see the engine-confound note above for the apples-to-apples variant.
* Near-duplicate detection is MinHash on word 3-grams; semantically paraphrased questions on different
  contexts are not detected (and are not leakage in this task, since contexts differ).
* `quantize_gguf.py` and LoRA-adapter serving in vLLM are not part of the gated path and are unverified.
* Release thresholds in `configs/release_policy.yaml` are examples. Calibrate them from the measured
  base vs fine-tuned gap and the bootstrap CIs before treating the gate as meaningful.

## Layout

```text
configs/            qlora.yaml (full, T4), qlora_smoke.yaml, release_policy.yaml
scripts/            train.py, merge_adapter.py, quantize_w4a16.py, quantize_gguf.py,
                    serve_vllm.sh, smoke_test_api.py, run_gpu_pipeline.sh
src/qlora_vllm/     prepare_data, dataset (masking), config, hardware, artifacts, prompting,
                    metrics, evaluate, bench, release_gate, promote
tests/              CPU tests + synthetic fixtures
data/               prepared datasets (gitignored), sample.jsonl
artifacts/ reports/ deploy/   generated (gitignored)
```
