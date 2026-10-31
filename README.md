# QLoRA fine-tuning → evaluation → release gate → quantized vLLM serving

A reproducible, single-GPU project that fine-tunes **Qwen2.5-1.5B-Instruct** with **QLoRA** for
grounded question answering (SQuAD v1.1), evaluates the base / fine-tuned / quantized models on the
same held-out set, **blocks or approves a release** with a fail-closed policy gate, and serves the
approved **W4A16** checkpoint locally with **vLLM**.

## Status: what has and has not been executed

| Stage | Executed? | Where / evidence |
|---|---|---|
| Data pipeline (validate, normalize, dedup, grouped split, manifest) | ✅ synthetic fixture (CPU tests) and real SQuAD (Colab T4 smoke run) | `tests/test_prepare_data.py`, smoke manifest |
| Chat templating + assistant-only loss masking + truncation policy | ✅ tiny tokenizer (CPU) and Qwen2.5 (smoke run: 0 of 320 examples dropped) | `tests/test_dataset.py`, `training_metadata.json` |
| QLoRA training with NF4 on a GPU | ✅ full profile: 20,000 examples, 2 epochs / 2,500 steps, Colab T4, FP16 | [Full-run evidence](evidence/full-20261031/) |
| `train.py` → resume → `merge_adapter.py` → `evaluate` → `compare` | ✅ full GPU run resumed from checkpoint 2250 and completed | [Full-run evidence](evidence/full-20261031/) |
| llm-compressor W4A16 quantization → vLLM serving → endpoint eval + bench | ✅ smoke and full runs; full candidate evaluated on 1,000 questions | [Measured results](#measured-results) |
| Release gate + promotion/rollback | ✅ full gate **PASS** and local promotion; smoke gate **FAIL**; rollback covered by CPU tests | [Full-run evidence](evidence/full-20261031/), `tests/test_release_gate.py` |
| Full-profile training and promotion of a real artifact | ✅ completed 2026-10-31; policy thresholds remain illustrative, calibration pending | [Full-run evidence](evidence/full-20261031/) |

The smoke run is a pipeline validation, not a model result: 20 steps on 256 examples, scored on 64
held-out questions. Numbers from it are reported below with that caveat. The completed full run uses
1,000 held-out questions and is reported separately.

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
| Tests/CI | 1 test | 36 CPU tests, `ci.yml`; manual self-hosted `gpu-pipeline.yml` |

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
pytest -q            # 36 tests, ~40-90 s on 2 CPU cores
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
  shuffled with the seed, so no context or article appears in two splits. At most `ceil(size / 8)`
  records of a group are used (`--min-groups-per-split`), so a split spans several articles; the unused
  remainder of a group is discarded, never handed to another split. The **test** split comes from
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
  (`drop`) or aborts the run (`error`); it is never truncated. The full run's longest encoded training
  example was 721 tokens; with `max_seq_length: 1024`, no training or validation examples were dropped.
  Check `training_metadata.json → encoding` for each run.
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

`pytest -q` runs 36 CPU tests: data validation and rejection accounting, Unicode-safe normalization,
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

## Measured results

### Smoke run — Colab, Tesla T4 (15 GB), 2026-10-31

Pipeline validation only (`--profile smoke`: 256 train / 64 validation / 64 test / 32 calibration,
`max_steps: 20`, effective batch 4). Environment: torch 2.13.0 (cu129 wheels), transformers 5.19.0,
peft 0.21.2, bitsandbytes 0.50.2, llmcompressor 0.15.0, vllm 0.32.0; dataset hash `53f7cd97…`.
Note: this run trained and merged in **BF16** because of a detection bug (`torch.cuda.is_bf16_supported()`
reports emulated BF16 on Turing); it is fixed in `hardware.py` (compute capability ≥ 8.0 required), so
later runs use FP16 on a T4 as intended.

| variant | backend | EM | F1 | empty | errors | artifact |
|---|---|---|---|---|---|---|
| base Qwen2.5-1.5B-Instruct | HF | 0.6875 | 0.7529 | 0 | 0 | `884234f9…` |
| fine-tuned (merged) | HF | 0.8125 | 0.8641 | 0 | 0 | `b5d6ecf3…` |
| quantized W4A16 | vLLM | 0.7344 | 0.8523 | 0 | 0 | `ff473c1c…` |

Paired bootstrap (n = 64, 10,000 resamples, 95% CI): fine-tuned − base EM **+0.125** [+0.047, +0.219],
F1 **+0.111** [+0.047, +0.187]; quantized − fine-tuned EM −0.078 [−0.188, +0.016], F1 −0.012
[−0.067, +0.047].

Training: 174 s for 20 steps (train loss 0.35, eval loss 0.26). Quantization: 79 s. vLLM start to
`/health`: 168 s. Endpoint benchmark (32 requests, concurrency 4, 8 warm-up, max 48 tokens):
p50/p95/p99 latency 0.073/0.227/0.348 s, TTFT p50/p95 0.044/0.060 s, 41.2 req/s, 272.7 output tok/s.

Release gate with the shipped example policy: **FAIL** on one check — `exact_match_quantization_degradation`
(observed 0.078 > 0.03); all 20 other checks passed, including artifact identity, lineage, freshness,
comparability and every benchmark limit. On 64 examples a 0.078 EM gap is five questions and its CI
spans zero, so this is the example threshold meeting a tiny sample, not evidence of real degradation.
Promotion was correctly blocked. The full-run results below provide evidence for future threshold
calibration; the shipped thresholds have not been changed.

### Full run — Colab, Tesla T4 (15 GB), 2026-10-31

Completed the full profile: **20,000 train / 1,000 validation / 1,000 test / 256 calibration**,
2 epochs (2,500 optimizer steps), NF4 QLoRA with FP16 compute. After earlier runtime interruptions,
the final session resumed from `checkpoint-2250` (epoch 1.8) and reached step 2,500 / epoch 2.0.
No training or validation examples were dropped for length. The resumed training invocation reported
1,597 s, including evaluation; this is **not** the total training time across sessions.

| variant | backend | EM | F1 | empty | errors | artifact id (prefix) |
|---|---|---|---|---|---|---|
| base Qwen2.5-1.5B-Instruct | HF | 0.7010 | 0.8059 | 0 | 0 | `884234f92902` |
| fine-tuned (merged) | HF | 0.8440 | 0.9191 | 0 | 0 | `814ca85ec752` |
| quantized W4A16 | vLLM | 0.8340 | 0.9134 | 0 | 0 | `69c8ef53799b` |

Same test-file SHA-256 prefix `0082f81f8031` and eval-config prefix `d604cdf59b1f` for all variants.
Paired bootstrap (1,000 examples, 10,000 resamples, 95% CI; differences in score units):

| comparison | EM difference [95% CI] | F1 difference [95% CI] |
|---|---|---|
| fine-tuned − base | +0.1430 [+0.1140, +0.1730] | +0.1132 [+0.0909, +0.1358] |
| quantized − fine-tuned | −0.0100 [−0.0220, +0.0020] | −0.0057 [−0.0148, +0.0033] |

Quantization took 201 s; vLLM became healthy after 170.2 s and passed the API smoke check.
Streaming benchmark: **128 completed / 0 failed**, concurrency 4, 8 warm-up requests, max 48 output tokens:

| measurement | full-run result |
|---|---|
| aggregate throughput | 40.06 requests/s; 254.15 output tokens/s |
| request latency p50 / p95 / p99 | 0.0794 / 0.2246 / 0.2927 s |
| time to first token p50 / p95 / p99 | 0.0433 / 0.0807 / 0.1045 s |

The shipped release policy returned **PASS (0 failed checks)** and promoted candidate
`69c8ef53799b06c76ae18ba84c6f3a1f54098d0db43633e87ed09c84a5423709` to the local
`deploy/approved/current` slot. Baseline checks were skipped because no approved baseline was supplied,
as permitted for a first release. This does not establish regression performance against a prior release
or calibrate the example thresholds. The evaluation server was stopped after benchmarking.

The final output reported **pipeline exit code 0** and **Final Drive backup complete** under
`MyDrive/qlora-vllm-finetuning-runs/full-resume-20261031T120211Z`.
See the [committed evidence and provenance](evidence/full-20261031/). Model weights and checkpoint
state remain outside Git. Allocator OOM warnings occurred during resumed training, but execution
continued through evaluation, promotion and the final backup.

## Known limitations and open steps

* Full-profile results come from one Colab T4 run spanning runtime interruptions. GPU resume and the
  FP16 precision fix were exercised; multi-seed and independent repeat runs remain open.
* No approved baseline was supplied for the first full release, so baseline regression checks were skipped.
* Benchmark results describe this short-answer QA workload at concurrency 4 on a single T4; they do
  not establish performance for longer generations, higher concurrency or other hardware.
* The CPU suite was run on the exact train-environment pins (`torch 2.13.0`, `transformers 5.19.0`,
  `peft 0.21.2`, `datasets 5.1.0`, `accelerate 1.15.0`) and also on `transformers 5.22.1`.
* vLLM on a T4 cannot use FlashAttention-2 (Ampere+); it falls back to another attention backend. The
  `--dtype float16`, `--max-model-len 2048`, `--gpu-memory-utilization 0.85` defaults in
  `serve_vllm.sh` worked on Colab but are starting points.
* The quantized candidate is scored through vLLM while base/fine-tuned are scored through
  `transformers.generate`; see the engine-confound note above for the apples-to-apples variant.
* Splits use at most `ceil(size / 8)` records per title group so a split spans several articles; the
  smoke run predates this change and drew its 64 test questions from a single article.
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
evidence/           committed full-run reports and provenance; no model weights
data/               prepared datasets (gitignored), sample.jsonl
artifacts/ reports/ deploy/   generated (gitignored)
```
