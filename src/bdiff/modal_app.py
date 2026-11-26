"""The Modal app behind the `modal:` backend: vLLM on one GPU, a base model with an optional LoRA adapter.

Imported only when a `modal:` spec is sampled (needs `pip install bdiff[modal]` and a Modal account:
`modal setup`). Each `sample_all` starts an ephemeral app, loads the model once, generates every prompt with its own
seed and stops; nothing stays deployed. Settings, read when the module is imported:

    BDIFF_MODAL_GPU       GPU type (default L40S; A100-80GB or H100 for models over ~14B parameters)
    BDIFF_MODAL_TIMEOUT   seconds before Modal stops the call (default 1800), which bounds the cost of one call
"""

from __future__ import annotations

import os
from typing import Any

import modal

GPU = os.environ.get("BDIFF_MODAL_GPU", "L40S")
TIMEOUT_S = int(os.environ.get("BDIFF_MODAL_TIMEOUT", "1800"))

app = modal.App("behavior-diff")
cache = modal.Volume.from_name("behavior-diff-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.10.1.1", "transformers==4.55.4", "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("bdiff")  # last: local files are mounted at start, not baked into the image
)


@app.function(  # type: ignore[untyped-decorator]
    gpu=GPU,
    image=image,
    volumes={"/cache": cache},
    timeout=TIMEOUT_S,
    retries=0,
    secrets=[modal.Secret.from_dict({k: v for k, v in os.environ.items() if k == "HF_TOKEN"})],
)
def generate(job: dict[str, Any]) -> list[str]:
    """job: {"repo", "adapter": None | {"repo", "subfolder"}, "messages": [[...], ...], "seeds": [...],
    "temperature", "max_tokens"}. Returns one reply per conversation, in order, reasoning blocks removed."""
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from bdiff.backends import strip_thinking

    adapter = job.get("adapter")
    llm = LLM(
        job["repo"],
        dtype="bfloat16",
        enable_lora=bool(adapter),
        max_lora_rank=64,
        max_model_len=4096,
        seed=0,
        gpu_memory_utilization=0.85,
    )
    tok = llm.get_tokenizer()
    texts = [
        tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for m in job["messages"]
    ]
    lora = None
    if adapter:
        sub = adapter.get("subfolder") or ""
        path = snapshot_download(adapter["repo"], allow_patterns=[f"{sub}/*"] if sub else None)
        lora = LoRARequest("adapter", 1, str(Path(path) / sub))
    params = [
        SamplingParams(
            temperature=job["temperature"],
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            max_tokens=job["max_tokens"],
            seed=s,
        )
        for s in job["seeds"]
    ]
    outs = llm.generate(texts, params, lora_request=lora)
    cache.commit()
    return [strip_thinking(o.outputs[0].text) for o in outs]


def run(job: dict[str, Any]) -> list[str]:
    """Start the app, run one generate call, stop."""
    with modal.enable_output(), app.run():
        replies: list[str] = generate.remote(job)
    return replies
