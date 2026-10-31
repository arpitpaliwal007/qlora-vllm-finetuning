#!/usr/bin/env bash
# Serve a model artifact with vLLM, named after its artifact id, and wait for health.
#
#   scripts/serve_vllm.sh start <artifact_dir> [port] [extra vllm args...]
#   scripts/serve_vllm.sh stop  <artifact_dir>
#
# The served model name is the artifact id from <artifact_dir>/artifact_manifest.json, so
# evaluation and benchmark reports can prove which artifact the endpoint served.
# Writes <artifact_dir>/serve_status.json (pid, port, model_load_seconds) and vllm.log.
set -euo pipefail

cmd="${1:?start|stop}"; dir="${2:?artifact dir}"; shift 2
port="${1:-8000}"; [ $# -gt 0 ] && shift
status="$dir/serve_status.json"
log="$dir/vllm.log"

artifact_id() { python - "$dir" <<'EOF'
import json, sys, pathlib
print(json.loads((pathlib.Path(sys.argv[1]) / "artifact_manifest.json").read_text())["artifact_id"])
EOF
}

case "$cmd" in
  start)
    id="$(artifact_id)"
    echo "[serve] starting vllm for $dir as model '$id' on port $port"
    start=$(date +%s.%N)
    # T4: no BF16 -> float16; 2048 context is ample for SQuAD prompts (< 600 tokens) and keeps KV cache small.
    nohup vllm serve "$dir" \
      --served-model-name "$id" \
      --dtype float16 \
      --max-model-len "${MAX_MODEL_LEN:-2048}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
      --port "$port" "$@" > "$log" 2>&1 &
    pid=$!
    for _ in $(seq 1 "${HEALTH_TIMEOUT:-600}"); do
      if curl -fsS "http://localhost:$port/health" > /dev/null 2>&1; then
        load=$(python -c "print(round($(date +%s.%N) - $start, 1))")
        python - "$status" "$pid" "$port" "$id" "$load" "$dir" <<'EOF'
import json, sys, datetime
status, pid, port, model, load, path = sys.argv[1:]
json.dump({"pid": int(pid), "port": int(port), "served_model_id": model, "artifact_dir": path,
           "model_load_seconds": float(load), "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "model_load_note": "wall time from process launch until /health returned 200"}, open(status, "w"), indent=2)
EOF
        echo "[serve] healthy after ${load}s (pid $pid); status in $status"
        exit 0
      fi
      if ! kill -0 "$pid" 2>/dev/null; then echo "[serve] vllm exited; see $log"; tail -n 40 "$log"; exit 1; fi
      sleep 1
    done
    echo "[serve] timed out waiting for health; see $log"; kill "$pid" || true; exit 1 ;;
  stop)
    pid=$(python -c "import json;print(json.load(open('$status'))['pid'])")
    kill "$pid" && echo "[serve] stopped pid $pid" ;;
  *) echo "usage: $0 start|stop <artifact_dir> [port]"; exit 2 ;;
esac
