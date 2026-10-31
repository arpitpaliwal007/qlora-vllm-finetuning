# Full-profile Colab evidence — 2026-10-31

Run backup: `MyDrive/qlora-vllm-finetuning-runs/full-resume-20261031T120211Z`.
The run resumed from checkpoint 2250, completed step 2500 / epoch 2, evaluated and benchmarked the
W4A16 candidate, passed the example release policy, and promoted the candidate into the local
approved slot. The final console output reports a completed Drive backup and pipeline exit code 0.

## Files and provenance

| file | source |
|---|---|
| [colab-output.txt](colab-output.txt) | Byte-for-byte copy of the console output supplied by the user in this chat; includes resumed training, merge, evaluation, quantization, benchmark, gate, promotion and final backup messages. |
| [comparison.md](comparison.md) | Unmodified comparison section extracted from that console output; contains all three model scores and paired bootstrap confidence intervals. |
| [quantized-results.json](quantized-results.json) | Complete JSON content read from the completed backup's `reports/full/quantized/results.json` in the Google Drive preview. |
| [bench.json](bench.json) | Complete JSON content read from the completed backup's `reports/full/bench.json` in the Google Drive preview. |
| [release_decision.json](release_decision.json) | Complete JSON content read from the completed backup's `reports/full/release_decision.json` in the Google Drive preview. |
| [SHA256SUMS](SHA256SUMS) | SHA-256 checksums of the five evidence files above. |

The three JSON reports above were first transcribed from Drive previews; they have since been replaced
by byte-exact downloads of the same files (verified semantically identical to the transcriptions), and
the following files were added, also downloaded byte-for-byte from the same Drive backup:

| file | original path in the run |
|---|---|
| [base-results.json](base-results.json), [finetuned-results.json](finetuned-results.json) | `reports/full/{base,finetuned}/results.json` |
| [comparison.json](comparison.json) | `reports/full/comparison/comparison.json` |
| [dataset-manifest.json](dataset-manifest.json) | `data/squad_qa/manifest.json` (split sizes, hashes, rejection counts, contamination check) |
| [training_metadata.json](training_metadata.json), [adapter-manifest.json](adapter-manifest.json) | `artifacts/qlora-adapter/` |
| [merged-manifest.json](merged-manifest.json) | `artifacts/merged-full/artifact_manifest.json` |
| [quantized-manifest.json](quantized-manifest.json), [serve_status.json](serve_status.json) | `artifacts/quantized-w4a16-full/` (calibration provenance, vLLM load time) |
| [promotion-history.json](promotion-history.json) | `deploy/approved/history.json` |

Lineage recorded in the manifests: adapter `58df96b1…` → merged `814ca85e…` → quantized `69c8ef53…`,
all on `Qwen/Qwen2.5-1.5B-Instruct@989aa798…`, dataset hash `5189d836…` in every report. The console
capture's SHA-256 is `f755dca640519a948443e55b077b653963ba31b0fa5966d04567d44e02b8c366`.

Per-example prediction files and model weights remain in Drive. Paths inside the copied
reports refer to the original Colab layout; this folder is not a deployable release bundle.
No adapter/model weights, optimizer or scheduler state, checkpoint directories, or dataset files are
included. Artifact IDs identify the candidate but do not let this Git-only snapshot re-hash its weights.

## Interpretation

The quantized candidate scored EM 0.8340 and F1 0.9134 on 1,000 held-out questions, with no empty
responses or evaluation errors. Its benchmark completed 128 requests with no failures: 40.06 requests/s,
254.15 output tokens/s, p95 latency 0.2246 s and p95 time to first token 0.0807 s at concurrency 4.

The release decision records 20 passed checks, two skipped checks and zero failures. Baseline validation
and regression checks were skipped under the first-release policy. The policy used by this run was the original illustrative one (now `tests/fixtures/example_release_policy.yaml`); `configs/release_policy.yaml` has since been calibrated on these results. Base/fine-tuned evaluation
uses HF, while candidate evaluation uses vLLM; quantization differences also include this engine change.

The console contains allocator OOM warnings that did not terminate the run. Its resumed-invocation
training time must not be interpreted as the total training time across all interrupted sessions.
Promotion was to a local directory; the candidate server was stopped after benchmarking.

No notebook, training or evaluation was run while assembling this evidence.
