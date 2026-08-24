# QLoRA Fine-Tuning for vLLM

An end-to-end, reproducible starter for instruction tuning an open-weight causal language model with **QLoRA (4-bit NF4)**, exporting a merged checkpoint, quantizing it for serving, and running it with vLLM.

## What is included

- QLoRA training via `transformers`, `peft`, `bitsandbytes`, and `trl`
- Chat-template-aware JSONL dataset loading and train/eval splitting
- Config-driven hyperparameters for a single GPU or larger machines
- LoRA merge/export for deployment
- GGUF quantization helper for llama.cpp-compatible models
- vLLM OpenAI-compatible API serving commands

## Requirements

- Python 3.10+
- NVIDIA GPU (QLoRA training is typically practical from 16 GB VRAM; model dependent)
- CUDA-compatible PyTorch installation

Install PyTorch for your CUDA version first, then install this project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[train,serve]'
```

## Dataset format

Each line of `data/train.jsonl` is a JSON object with a `messages` array. The `content` may be a string or a list of text parts.

```json
{"messages":[{"role":"system","content":"You are a precise assistant."},{"role":"user","content":"What is QLoRA?"},{"role":"assistant","content":"QLoRA fine-tunes a quantized base model with small low-rank adapters."}]}
```

Create a starter dataset:

```bash
cp data/sample.jsonl data/train.jsonl
```

## Train

Edit `configs/qlora.yaml` to choose your base model and training budget, then run:

```bash
python scripts/train.py --config configs/qlora.yaml
```

Adapters and tokenizer are written to `artifacts/qlora-adapter/`. To resume, set `resume_from_checkpoint` in the config.

Useful low-memory settings are enabled by default: NF4 4-bit loading, bf16 compute where supported, gradient checkpointing, and paged AdamW. If your GPU lacks bf16 support, set `bf16: false` and `fp16: true`.

## Merge the adapter

For vLLM serving, merge the LoRA adapter into the base model:

```bash
python scripts/merge_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir artifacts/qlora-adapter \
  --output-dir artifacts/merged-model
```

## Serve with vLLM

```bash
vllm serve artifacts/merged-model \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```

The endpoint is OpenAI-compatible at `http://localhost:8000/v1`. Test it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"artifacts/merged-model","messages":[{"role":"user","content":"Explain QLoRA in one sentence."}]}'
```

### Serve adapter without merging (optional)

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --enable-lora --lora-modules tuned=artifacts/qlora-adapter
```

Then use `tuned` as the model name. Check the vLLM documentation for your installed version because LoRA serving flags can change.

## Quantization options

QLoRA's 4-bit quantization is used **during training**; it does not itself create a portable 4-bit serving model.

| Goal | Recommended approach |
| --- | --- |
| Highest vLLM throughput on supported NVIDIA GPUs | Serve merged BF16/FP16 weights, or use a vLLM-supported checkpoint quantization method |
| Small local CPU/GPU deployment | Export to GGUF and quantize with llama.cpp |
| NVIDIA optimized 4-bit serving | Build an AWQ/GPTQ checkpoint with a compatible calibration workflow, then pass vLLM's matching `--quantization` backend |

For a GGUF workflow, set `LLAMA_CPP_DIR` and run:

```bash
export LLAMA_CPP_DIR=/path/to/llama.cpp
python scripts/quantize_gguf.py \
  --model-dir artifacts/merged-model \
  --output-dir artifacts/gguf \
  --quant-type Q4_K_M
```

## Layout

```text
configs/             training configuration
data/                JSONL examples (gitignored except sample)
scripts/             train, merge, and quantize entry points
src/qlora_vllm/      reusable dataset and training helpers
artifacts/           generated adapters, merged models, and quantized outputs
```

## Notes

- Ensure the base model license permits your intended use.
- Never commit proprietary training data or model artifacts; `.gitignore` excludes them.
- For multi-GPU training, launch the same script with `accelerate launch` after configuring Accelerate.
