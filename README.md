# LLM — podQuery RunPod Worker

> vLLM inference server for Qwen3.5-27B-FP8 on RunPod Serverless.

## Purpose

Serves all LLM tasks in the podQuery pipeline: episode analysis (summary, keywords, sentiment, 6-tier speaker scores), transcript cleanup, readable version generation, chapter generation, named entity extraction, speaker name mapping, fact checking, show notes, and social posts.

## Docker Image

| Registry | Image |
|----------|-------|
| GHCR | `ghcr.io/timtegtmeyer/vllm-qwen27b-fp8` |

## RunPod Endpoint Setup

### 1. Create Serverless Endpoint

- Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
- Click **New Endpoint**

### 2. Configuration

| Setting | Value |
|---------|-------|
| **Container Image** | `ghcr.io/timtegtmeyer/vllm-qwen27b-fp8@sha256:<digest>` |
| **Container Disk** | 60 GB |
| **GPU Type** | A40 (48 GB VRAM) |
| **Min Workers** | 0 |
| **Max Workers** | 2 |
| **Idle Timeout** | 5s |
| **Flash Boot** | Enabled |

### 3. Environment Variables

No runtime environment variables required. Model is baked into the image.

### 4. Test the Endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"debug": true}}'
```

## Building

### Local Build (recommended)

```bash
make build    # builds, pushes to GHCR, captures digest, tags
make digest   # shows digest for RunPod endpoint config
```

### Image Digest

After building, update the RunPod endpoint with the digest from `.last-digest`.

## Input Schema

### Chat format (primary)

```json
{
  "messages": [
    {"role": "system", "content": "You are..."},
    {"role": "user", "content": "Analyze this transcript..."}
  ],
  "sampling_params": {
    "max_tokens": 16384,
    "temperature": 0.2
  }
}
```

### Debug

```json
{"debug": true}
```

## Output Schema

```json
{
  "text": "response content",
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  },
  "diagnostics": {
    "generation_time_sec": 12.3,
    "tokens_per_sec": 46.1
  }
}
```

## Model

- **Qwen3.5-27B-FP8** — pre-quantized FP8 by Qwen team
- Thinking mode disabled (`enable_thinking=False`)
- ~27 GB model weight, fits single A40
- No runtime quantization overhead
