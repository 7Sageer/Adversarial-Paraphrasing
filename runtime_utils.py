from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


LOCAL_PARAPHRASER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LOCAL_GUIDANCE_CLASSIFIER = "openai_roberta_base"
LOCAL_DEPLOY_CLASSIFIER = "openai_roberta_base"


def resolve_runtime_device(preferred: str = "auto") -> str:
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable on this machine.")
        return preferred

    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable in the current PyTorch runtime.")
        return preferred

    if preferred == "cpu":
        return preferred

    raise ValueError(f"Unsupported device '{preferred}'. Expected one of: auto, cpu, mps, cuda.")


def resolve_torch_dtype(device: str, precision: str = "auto") -> torch.dtype:
    if precision == "auto":
        if device in {"cuda", "mps"}:
            return torch.float16
        return torch.float32

    if precision == "float32":
        return torch.float32

    if precision == "float16":
        if device == "cpu":
            raise RuntimeError("float16 on CPU is not supported for this local runtime.")
        return torch.float16

    if precision == "bfloat16":
        if device != "cuda":
            raise RuntimeError("bfloat16 is only enabled for CUDA in this local runtime.")
        return torch.bfloat16

    raise ValueError(
        f"Unsupported precision '{precision}'. Expected one of: auto, float32, float16, bfloat16."
    )


def save_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
