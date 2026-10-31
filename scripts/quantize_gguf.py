#!/usr/bin/env python
"""Convert a merged Hugging Face model to GGUF and quantize it with llama.cpp."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quant-type", default="Q4_K_M")
    args = parser.parse_args()
    llama_cpp = Path(os.environ["LLAMA_CPP_DIR"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp16 = output_dir / "model-f16.gguf"
    quantized = output_dir / f"model-{args.quant_type.lower()}.gguf"
    subprocess.run(["python", str(llama_cpp / "convert_hf_to_gguf.py"), args.model_dir, "--outtype", "f16", "--outfile", str(fp16)], check=True)
    quantize = llama_cpp / "build" / "bin" / "llama-quantize"
    subprocess.run([str(quantize), str(fp16), str(quantized), args.quant_type], check=True)
    print(f"Created {quantized}")


if __name__ == "__main__":
    main()
