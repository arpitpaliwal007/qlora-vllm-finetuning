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

Drive downloads failed with “Can't download file.” The three JSON reports were therefore transcribed
from the visible previews and reformatted; their original whitespace is not preserved. Their parsed
content was checked against the preview content. The console capture's SHA-256 is
`f755dca640519a948443e55b077b653963ba31b0fa5966d04567d44e02b8c366`.

This is an evidence subset: the original per-example prediction files, base/fine-tuned full JSON
reports, training metadata, artifact manifests and weights remain in Drive. Paths inside the copied
reports refer to the original Colab layout; this folder is not a deployable release bundle.
No adapter/model weights, optimizer or scheduler state, checkpoint directories, or dataset files are
included. Artifact IDs identify the candidate but do not let this Git-only snapshot re-hash its weights.

## Interpretation

The quantized candidate scored EM 0.8340 and F1 0.9134 on 1,000 held-out questions, with no empty
responses or evaluation errors. Its benchmark completed 128 requests with no failures: 40.06 requests/s,
254.15 output tokens/s, p95 latency 0.2246 s and p95 time to first token 0.0807 s at concurrency 4.

The release decision records 20 passed checks, two skipped checks and zero failures. Baseline validation
and regression checks were skipped under the first-release policy. The shipped policy remains
illustrative and has not been calibrated by this documentation update. Base/fine-tuned evaluation
uses HF, while candidate evaluation uses vLLM; quantization differences also include this engine change.

The console contains allocator OOM warnings that did not terminate the run. Its resumed-invocation
training time must not be interpreted as the total training time across all interrupted sessions.
Promotion was to a local directory; the candidate server was stopped after benchmarking.

No notebook, training, evaluation or tests were run while assembling this evidence.
