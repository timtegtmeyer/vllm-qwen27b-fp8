"""
RunPod serverless handler for Qwen3.5-27B-FP8 via vLLM.

Starts a vLLM engine at container init, then serves inference requests
through RunPod's serverless protocol. Thinking mode is disabled —
the model outputs direct responses only (no <think> blocks).

Expected input:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 2048,
        "temperature": 0.7
    }

    Or raw prompt:
    {
        "prompt": "Summarize this podcast transcript: ...",
        "max_tokens": 2048,
        "temperature": 0.7
    }

    Or via sampling_params (used by tenant-backend):
    {
        "messages": [...],
        "sampling_params": {
            "max_tokens": 16384,
            "temperature": 0.2
        }
    }

    Debug mode:
    { "debug": true }
"""
import os
import sys
import time
import logging

import runpod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("qwen-worker")

# ---------------------------------------------------------------------------
# vLLM engine — loaded once at container start
# ---------------------------------------------------------------------------

_llm = None
_tokenizer = None
_load_error = None
_load_time = None

MODEL_DIR = "/models/Qwen3.5-27B-FP8"


def _load_engine():
    global _llm, _tokenizer, _load_error, _load_time

    if _llm is not None:
        return _llm

    from vllm import LLM

    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "16384"))
    tensor_parallel = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    gpu_util = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
    dtype = os.environ.get("DTYPE", "auto")

    # vLLM's default warmup captures 51 CUDA graphs (batch sizes 1→512),
    # each consuming ~300 MB for a 27B model → ~15 GB. That OOMs on 44 GB
    # GPUs even with a reasonable KV-cache budget. We only ever serve
    # batch=1 with max_workers=1, so a tiny capture set covers us at
    # near-full speed (≤3% slower than the full set on batch=1) while
    # freeing ~12 GB for weights + KV cache. Overridable via env var
    # for bigger GPUs.
    capture_sizes = [int(x) for x in os.environ.get(
        "CUDAGRAPH_CAPTURE_SIZES", "1,2,4,8,16,32"
    ).split(",") if x.strip()]

    log.info(
        "Loading vLLM engine: model=%s, max_model_len=%d, dtype=%s, tp=%d, gpu_util=%.2f, cudagraph_capture_sizes=%s",
        MODEL_DIR, max_model_len, dtype, tensor_parallel, gpu_util, capture_sizes,
    )

    start = time.time()
    try:
        _llm = LLM(
            model=MODEL_DIR,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_util,
            dtype=dtype,
            trust_remote_code=True,
            compilation_config={"cudagraph_capture_sizes": capture_sizes},
        )
        _tokenizer = _llm.get_tokenizer()
        _load_time = time.time() - start
        log.info("vLLM engine ready (loaded in %.1fs)", _load_time)
    except Exception as exc:
        _load_error = str(exc)
        _load_time = time.time() - start
        log.exception("Failed to load vLLM engine")
        raise

    return _llm


# Pre-load at container startup — but ONLY in the main process. vLLM uses
# `spawn` multiprocessing (warning above "Overriding VLLM_WORKER_MULTIPROC_METHOD
# to 'spawn' … CUDA is initialized"), and spawn re-imports this module in
# each child. Without the __main__ guard the child would call _load_engine()
# too, recursively spawning more children until Python aborts with "An
# attempt has been made to start a new process before the current process
# has finished its bootstrapping phase."
if __name__ == "__main__":
    try:
        _load_engine()
    except Exception as exc:
        log.error("Could not pre-load engine: %s", exc)


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def _debug_info() -> dict:
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "status": "ok" if _llm is not None else "error",
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if cuda_available else None,
        "model_dir": MODEL_DIR,
        "engine_loaded": _llm is not None,
        "engine_load_error": _load_error,
        "engine_load_time_sec": round(_load_time, 1) if _load_time else None,
        "max_model_len": os.environ.get("MAX_MODEL_LEN", "16384"),
    }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    from vllm import SamplingParams

    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")
    log.info("[job:%s] Received job", job_id)

    if job_input.get("debug"):
        return _debug_info()

    # Build prompt from either raw prompt or chat messages
    messages = job_input.get("messages")
    prompt = job_input.get("prompt")

    if messages and _tokenizer:
        # Non-thinking mode: disable chain-of-thought reasoning
        prompt = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    elif not prompt:
        return {"error": "input.prompt or input.messages is required"}

    # Sampling parameters — support both flat and nested formats
    sp = job_input.get("sampling_params", {})
    max_tokens = sp.get("max_tokens") or job_input.get("max_tokens", 2048)
    temperature = sp.get("temperature") if sp.get("temperature") is not None else job_input.get("temperature", 0.7)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=sp.get("top_p") or job_input.get("top_p", 0.9),
        top_k=sp.get("top_k") or job_input.get("top_k", -1),
        stop=sp.get("stop") or job_input.get("stop"),
        repetition_penalty=sp.get("repetition_penalty") or job_input.get("repetition_penalty", 1.0),
    )

    log.info("[job:%s] Generating with max_tokens=%d, temp=%.2f", job_id, params.max_tokens, params.temperature)

    llm = _load_engine()
    start = time.time()
    outputs = llm.generate([prompt], params)
    elapsed = time.time() - start

    result = outputs[0]
    generated_text = result.outputs[0].text
    num_tokens = len(result.outputs[0].token_ids)
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

    log.info(
        "[job:%s] Generated %d tokens in %.1fs (%.1f tok/s)",
        job_id, num_tokens, elapsed, tokens_per_sec,
    )

    return {
        "text": generated_text,
        "full_output": generated_text,
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": num_tokens,
            "total_tokens": len(result.prompt_token_ids) + num_tokens,
        },
        "diagnostics": {
            "generation_time_sec": round(elapsed, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
        },
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
