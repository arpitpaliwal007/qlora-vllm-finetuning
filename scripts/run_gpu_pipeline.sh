#!/usr/bin/env bash
# End-to-end GPU pipeline: data -> QLoRA -> merge -> evaluate -> quantize -> serve -> evaluate/bench -> gate -> promote.
#
#   scripts/run_gpu_pipeline.sh smoke   # ~20 optimizer steps on 256 examples; validates the whole path
#   scripts/run_gpu_pipeline.sh full    # practical single-T4 run (hours)
#
# Requirements: one NVIDIA GPU (T4 16 GB or better), two virtualenvs (see README "GPU setup"):
#   TRAIN_PY  python of the train/quantize env (default .venv-train/bin/python)
#   SERVE_PY  python of the vLLM env           (default .venv-serve/bin/python)
# Every step is idempotent on its outputs; delete a directory to redo that step.
set -euo pipefail
cd "$(dirname "$0")/.."

profile="${1:?smoke|full}"
TRAIN_PY="${TRAIN_PY:-.venv-train/bin/python}"
SERVE_PY="${SERVE_PY:-.venv-serve/bin/python}"
PORT="${PORT:-8001}"
PROMOTE="${PROMOTE:-1}"
case "$profile" in
  smoke) data=data/squad_qa_smoke; cfg=configs/qlora_smoke.yaml; adapter=artifacts/qlora-adapter-smoke; tag=smoke; nreq=32; limit=64 ;;
  full)  data=data/squad_qa;       cfg=configs/qlora.yaml;       adapter=artifacts/qlora-adapter;       tag=full;  nreq=128; limit=0 ;;
  *) echo "profile must be smoke or full"; exit 2 ;;
esac
merged="artifacts/merged-$tag"; quant="artifacts/quantized-w4a16-$tag"; reports="reports/$tag"
mkdir -p "$reports"
log() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

log "hardware"
"$TRAIN_PY" -c "from qlora_vllm.hardware import detect_hardware; import json; print(json.dumps(detect_hardware()['gpus'], indent=1))"

log "1/9 prepare data ($profile)"
[ -f "$data/manifest.json" ] || "$TRAIN_PY" -m qlora_vllm.prepare_data --source hf --out "$data" --profile "$profile"

log "2/9 QLoRA training"
[ -f "$adapter/artifact_manifest.json" ] || "$TRAIN_PY" scripts/train.py --config "$cfg"

log "3/9 merge adapter"
[ -f "$merged/artifact_manifest.json" ] || "$TRAIN_PY" scripts/merge_adapter.py --adapter-dir "$adapter" --output-dir "$merged"

log "4/9 evaluate base and fine-tuned (HF backend)"
limit_arg=(); [ "$limit" -gt 0 ] && limit_arg=(--limit "$limit")
[ -f "$reports/base/results.json" ] || "$TRAIN_PY" -m qlora_vllm.evaluate run --backend hf --variant base \
  --model Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --test "$data/test.jsonl" --out "$reports/base" "${limit_arg[@]}"
[ -f "$reports/finetuned/results.json" ] || "$TRAIN_PY" -m qlora_vllm.evaluate run --backend hf --variant finetuned \
  --model "$merged" --test "$data/test.jsonl" --out "$reports/finetuned" "${limit_arg[@]}"

log "5/9 quantize W4A16 (llm-compressor GPTQ, calibration from train split)"
[ -f "$quant/artifact_manifest.json" ] || "$TRAIN_PY" scripts/quantize_w4a16.py --model-dir "$merged" \
  --calibration "$data/calibration.jsonl" --output-dir "$quant"

log "6/9 serve candidate with vLLM"
PATH="$(dirname "$SERVE_PY"):$PATH" scripts/serve_vllm.sh start "$quant" "$PORT"
trap 'scripts/serve_vllm.sh stop "$quant" || true' EXIT
"$TRAIN_PY" scripts/smoke_test_api.py --base-url "http://localhost:$PORT/v1" --candidate-dir "$quant"

log "7/9 evaluate and benchmark the served candidate"
"$TRAIN_PY" -m qlora_vllm.evaluate run --backend openai --variant quantized --base-url "http://localhost:$PORT/v1" \
  --candidate-dir "$quant" --test "$data/test.jsonl" --out "$reports/quantized" "${limit_arg[@]}"
"$TRAIN_PY" -m qlora_vllm.bench --base-url "http://localhost:$PORT/v1" --candidate-dir "$quant" \
  --prompts "$data/test.jsonl" --requests "$nreq" --concurrency 4 --warmup 8 --out "$reports/bench.json"
scripts/serve_vllm.sh stop "$quant"; trap - EXIT

log "8/9 comparison report"
"$TRAIN_PY" -m qlora_vllm.evaluate compare --report "base=$reports/base/results.json" \
  --report "finetuned=$reports/finetuned/results.json" --report "quantized=$reports/quantized/results.json" --out "$reports/comparison"

log "9/9 release gate"
baseline=(); [ -f deploy/approved/current/release/results.json ] && baseline=(--baseline deploy/approved/current/release/results.json)
set +e
"$TRAIN_PY" -m qlora_vllm.release_gate --policy configs/release_policy.yaml --candidate "$reports/quantized/results.json" \
  --candidate-dir "$quant" --finetuned "$reports/finetuned/results.json" --bench "$reports/bench.json" \
  "${baseline[@]}" --out "$reports/release_decision.json" --verify-weights
gate=$?
set -e
if [ "$gate" -eq 0 ] && [ "$PROMOTE" = "1" ]; then
  log "promote"
  "$TRAIN_PY" -m qlora_vllm.promote --decision "$reports/release_decision.json" --candidate-dir "$quant" --approved-dir deploy/approved
  echo "Serve the approved artifact with: scripts/serve_vllm.sh start deploy/approved/current 8000"
else
  echo "Not promoting (gate exit code $gate, PROMOTE=$PROMOTE). Decision: $reports/release_decision.json"
fi
exit "$gate"
