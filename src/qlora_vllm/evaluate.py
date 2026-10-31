"""Evaluate base / fine-tuned / quantized models on the held-out test split.

Two backends share the same records, prompt messages and decoding settings:

* ``hf``     - local transformers inference (optionally with a LoRA adapter or NF4 loading)
* ``openai`` - an OpenAI-compatible endpoint such as ``vllm serve``

``compare`` merges several results files into a report with paired bootstrap CIs.

    python -m qlora_vllm.evaluate run --backend hf --model Qwen/Qwen2.5-1.5B-Instruct \
        --test data/squad_qa/test.jsonl --out reports/base
    python -m qlora_vllm.evaluate run --backend openai --base-url http://localhost:8000/v1 \
        --candidate-dir artifacts/quantized-w4a16 --test data/squad_qa/test.jsonl --out reports/quantized
    python -m qlora_vllm.evaluate compare --report base=reports/base/results.json \
        --report finetuned=reports/finetuned/results.json --report quantized=reports/quantized/results.json \
        --out reports/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .artifacts import MANIFEST_NAME, package_versions, read_artifact_manifest, read_json, sha256_file, sha256_json, sha256_text, utc_now, write_json
from .hardware import detect_hardware, resolve_precision, torch_dtype
from .metrics import paired_bootstrap, score_prediction
from .prepare_data import read_jsonl
from .prompting import SYSTEM_PROMPT, prompt_messages

RESULTS_SCHEMA = 1


# --------------------------------------------------------------------------- backends


class HFBackend:
    def __init__(self, model: str, adapter: str | None, load_in_4bit: bool, precision: str, batch_size: int, revision: str | None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.hardware = detect_hardware()
        self.precision = resolve_precision(precision, self.hardware)
        self.batch_size = batch_size
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        kwargs: dict[str, Any] = {"dtype": torch_dtype(self.precision), "revision": revision}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype(self.precision)
            )
        if self.hardware["cuda_available"]:
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.load_seconds = time.time() - t0
        self.identity = model_identity(model, adapter, load_in_4bit, getattr(self.model.config, "_commit_hash", None))
        self.identity["backend"] = "hf"
        self.identity["precision"] = self.precision
        self.device = next(self.model.parameters()).device

    def generate(self, records: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
        import torch

        outputs: list[dict[str, Any]] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            prompts = [self.tokenizer.apply_chat_template(prompt_messages(r), tokenize=False, add_generation_prompt=True) for r in batch]
            t0 = time.time()
            try:
                enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                with torch.no_grad():
                    gen = self.model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                new_tokens = gen[:, enc["input_ids"].shape[1] :]
                texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                elapsed = (time.time() - t0) / len(batch)
                for text, toks in zip(texts, new_tokens, strict=True):
                    n_out = int((toks != self.tokenizer.pad_token_id).sum())
                    outputs.append({"prediction": text.strip(), "error": None, "output_tokens": n_out, "latency_s": elapsed})
            except Exception as exc:  # noqa: BLE001 - recorded as an inference error
                for _ in batch:
                    outputs.append({"prediction": "", "error": f"{type(exc).__name__}: {exc}", "output_tokens": 0, "latency_s": None})
        return outputs


class OpenAIBackend:
    def __init__(self, base_url: str, model: str | None, candidate_dir: str | None, concurrency: int, timeout: float, seed: int):
        self.base_url = base_url.rstrip("/")
        self.concurrency = concurrency
        self.timeout = timeout
        self.seed = seed
        self.hardware = detect_hardware()
        self.load_seconds = None
        served = [m["id"] for m in http_json(f"{self.base_url}/models")["data"]]
        if model is None:
            if len(served) != 1:
                raise RuntimeError(f"--model is required because the endpoint serves {served}")
            model = served[0]
        if model not in served:
            raise RuntimeError(f"model {model!r} is not served; endpoint lists {served}")
        self.model = model
        self.identity: dict[str, Any] = {"backend": "openai", "base_url": self.base_url, "served_model_id": model, "kind": "endpoint"}
        if candidate_dir:
            manifest = read_artifact_manifest(candidate_dir)
            self.identity.update(
                {
                    "path": str(candidate_dir),
                    "artifact_id": manifest["artifact_id"],
                    "kind": manifest["kind"],
                    "base_model": manifest.get("base_model"),
                    # The serve script names the served model after the artifact id; a mismatch means the
                    # endpoint is not serving the artifact we think it is. The release gate fails closed on it.
                    "identity_verified": manifest["artifact_id"] == model,
                }
            )
            if not self.identity["identity_verified"]:
                print(f"[eval] WARNING: served model id {model!r} != candidate artifact id {manifest['artifact_id']!r}", file=sys.stderr)
        else:
            self.identity["artifact_id"] = sha256_text(f"endpoint:{self.base_url}:{model}")
            self.identity["identity_verified"] = False

    def _one(self, record: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": prompt_messages(record),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_new_tokens,
            "seed": self.seed,
        }
        t0 = time.time()
        try:
            resp = http_json(f"{self.base_url}/chat/completions", payload, timeout=self.timeout)
            text = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage") or {}
            return {"prediction": text.strip(), "error": None, "output_tokens": usage.get("completion_tokens"), "latency_s": time.time() - t0}
        except Exception as exc:  # noqa: BLE001
            return {"prediction": "", "error": f"{type(exc).__name__}: {exc}", "output_tokens": 0, "latency_s": time.time() - t0}

    def generate(self, records: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(lambda r: self._one(r, max_new_tokens), records))


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local endpoint
        return json.loads(response.read().decode())


def model_identity(model: str, adapter: str | None, load_in_4bit: bool, commit_hash: str | None) -> dict[str, Any]:
    """Identity of a locally loaded model: its artifact manifest when it has one, else the hub id + commit."""
    path = Path(adapter or model)
    if (path / MANIFEST_NAME).exists():
        manifest = read_artifact_manifest(path)
        ident = {"path": str(path), "artifact_id": manifest["artifact_id"], "kind": manifest["kind"], "base_model": manifest.get("base_model")}
    else:
        ident = {"path": model, "kind": "hub_model", "base_model": model, "revision": commit_hash, "artifact_id": sha256_text(f"hub:{model}@{commit_hash}")}
    ident.update({"adapter": adapter, "load_in_4bit": load_in_4bit, "identity_verified": True})
    return ident


# --------------------------------------------------------------------------- run


def run(args: argparse.Namespace) -> dict[str, Any]:
    test_path = Path(args.test)
    records = read_jsonl(test_path)
    if args.limit:
        records = records[: args.limit]
    if args.backend == "hf":
        backend: Any = HFBackend(args.model, args.adapter, args.load_in_4bit, args.precision, args.batch_size, args.revision)
    else:
        backend = OpenAIBackend(args.base_url, args.model, args.candidate_dir, args.concurrency, args.timeout, args.seed)
    eval_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "decoding": "greedy",
        "seed": args.seed,
        "n_examples": len(records),
        "test_path": test_path.name,
        "test_sha256": sha256_file(test_path),
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "prompt_source": "messages field of the prepared test records (identical to training prompts)",
    }
    t0 = time.time()
    outputs = backend.generate(records, args.max_new_tokens)
    gen_seconds = time.time() - t0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record, output in zip(records, outputs, strict=True):
        scores = score_prediction(output["prediction"], record["answers"]) if output["error"] is None else {"exact_match": 0.0, "f1": 0.0}
        rows.append(
            {
                "id": record["id"],
                "prediction": output["prediction"],
                "references": record["answers"],
                **scores,
                **{k: output[k] for k in ("error", "output_tokens", "latency_s")},
            }
        )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(rows)
    n_errors = sum(1 for r in rows if r["error"])
    n_empty = sum(1 for r in rows if not r["error"] and not r["prediction"].strip())
    out_tokens = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
    manifest_path = test_path.parent / "manifest.json"
    dataset_hash = read_json(manifest_path).get("dataset_hash") if manifest_path.exists() else None
    results = {
        "schema_version": RESULTS_SCHEMA,
        "created_at": utc_now(),
        "variant": args.variant,
        "model": backend.identity,
        "dataset": {"path": str(test_path), "sha256": eval_config["test_sha256"], "dataset_hash": dataset_hash, "n": n},
        "eval_config": eval_config,
        "eval_config_hash": sha256_json(eval_config),
        "metrics": {
            "exact_match": sum(r["exact_match"] for r in rows) / n,
            "f1": sum(r["f1"] for r in rows) / n,
            "empty_rate": n_empty / n,
            "error_rate": n_errors / n,
            "n": n,
            "n_empty": n_empty,
            "n_errors": n_errors,
            "mean_output_tokens": sum(out_tokens) / len(out_tokens) if out_tokens else None,
        },
        "timing": {
            "model_load_seconds": None if backend.load_seconds is None else round(backend.load_seconds, 2),
            "model_load_note": None
            if backend.load_seconds is not None
            else "unavailable: the endpoint was started outside this process (see serve_status.json from serve_vllm.sh)",
            "generation_seconds": round(gen_seconds, 2),
            "examples_per_second": round(n / gen_seconds, 3) if gen_seconds else None,
        },
        "hardware": backend.hardware,
        "packages": package_versions(),
        "predictions_path": "predictions.jsonl",
    }
    write_json(out_dir / "results.json", results)
    print(json.dumps({"variant": args.variant, "metrics": results["metrics"], "artifact_id": backend.identity.get("artifact_id")}, indent=2))
    return results


# --------------------------------------------------------------------------- compare


def load_reports(specs: list[str]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError("--report expects name=path/to/results.json")
        reports[name] = read_json(path)
        reports[name]["_path"] = path
    return reports


def check_comparable(reports: dict[str, dict[str, Any]]) -> None:
    datasets = {r["dataset"]["sha256"] for r in reports.values()}
    configs = {r["eval_config_hash"] for r in reports.values()}
    if len(datasets) != 1 or len(configs) != 1:
        raise ValueError(f"reports are not comparable: test sha256s={datasets} eval_config_hashes={configs}")


def predictions_for(report: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(Path(report["_path"]).parent / report["predictions_path"])


def compare(args: argparse.Namespace) -> dict[str, Any]:
    reports = load_reports(args.report)
    check_comparable(reports)
    pairs = args.pair or [p for p in ("finetuned:base", "quantized:finetuned") if all(n in reports for n in p.split(":"))]
    comparisons = {}
    for pair in pairs:
        candidate, reference = pair.split(":")
        preds_c, preds_r = predictions_for(reports[candidate]), predictions_for(reports[reference])
        if [p["id"] for p in preds_c] != [p["id"] for p in preds_r]:
            raise ValueError(f"{pair}: prediction files cover different example ids")
        comparisons[pair] = {
            metric: paired_bootstrap([p[metric] for p in preds_r], [p[metric] for p in preds_c], n_resamples=args.resamples, seed=args.seed)
            for metric in ("exact_match", "f1")
        }
    summary = {
        "created_at": utc_now(),
        "dataset": next(iter(reports.values()))["dataset"],
        "eval_config_hash": next(iter(reports.values()))["eval_config_hash"],
        "variants": {name: {"metrics": r["metrics"], "model": r["model"], "created_at": r["created_at"], "report": r["_path"]} for name, r in reports.items()},
        "paired_bootstrap": comparisons,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "comparison.json", summary)
    (out_dir / "comparison.md").write_text(render_markdown(summary), encoding="utf-8")
    print(render_markdown(summary))
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Model comparison",
        "",
        f"Test set: `{summary['dataset']['path']}` (sha256 `{summary['dataset']['sha256'][:12]}`, n={summary['dataset']['n']}); "
        f"eval config `{summary['eval_config_hash'][:12]}`.",
        "",
    ]
    lines += ["| variant | EM | F1 | empty rate | error rate | artifact id | backend |", "|---|---|---|---|---|---|---|"]
    for name, v in summary["variants"].items():
        m, model = v["metrics"], v["model"]
        lines.append(
            f"| {name} | {m['exact_match']:.4f} | {m['f1']:.4f} | {m['empty_rate']:.4f} | {m['error_rate']:.4f} "
            f"| `{str(model.get('artifact_id'))[:12]}` | {model.get('backend')} |"
        )
    lines += ["", "## Paired bootstrap (candidate - reference)", "", "| pair | metric | mean diff | 95% CI | n |", "|---|---|---|---|---|"]
    for pair, metrics in summary["paired_bootstrap"].items():
        for metric, b in metrics.items():
            lines.append(f"| {pair} | {metric} | {b['mean_diff']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | {b['n']} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["hf", "openai"], required=True)
    r.add_argument("--test", required=True, help="prepared test.jsonl")
    r.add_argument("--out", required=True)
    r.add_argument("--variant", default="candidate", help="label: base | finetuned | quantized | ...")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--max-new-tokens", type=int, default=48)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model", help="hf: model path/id; openai: served model id (optional when exactly one model is served)")
    r.add_argument("--revision")
    r.add_argument("--adapter")
    r.add_argument("--load-in-4bit", action="store_true")
    r.add_argument("--precision", default="auto")
    r.add_argument("--batch-size", type=int, default=8)
    r.add_argument("--base-url", default="http://localhost:8000/v1")
    r.add_argument("--candidate-dir", help="openai: local artifact dir the endpoint is expected to serve")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--timeout", type=float, default=120.0)
    c = sub.add_parser("compare")
    c.add_argument("--report", action="append", required=True, help="name=path/to/results.json (repeatable)")
    c.add_argument("--pair", action="append", help="candidate:reference (default finetuned:base, quantized:finetuned)")
    c.add_argument("--out", required=True)
    c.add_argument("--resamples", type=int, default=10000)
    c.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        compare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
