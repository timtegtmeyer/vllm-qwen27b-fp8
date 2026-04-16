FROM vllm/vllm-openai@sha256:d9a5c1c1614c959fde8d2a4d68449db184572528a6055afdd0caf1e66fb51504

# Install hf_transfer for fast parallel downloads during build
RUN pip install --no-cache-dir hf_transfer 'huggingface_hub[cli]' runpod

# Download model weights into the image at build time
COPY builder/download_model.sh /builder/download_model.sh
RUN chmod +x /builder/download_model.sh
RUN --mount=type=secret,id=HF_TOKEN /builder/download_model.sh

# Copy handler (small, changes often — LAST)
COPY src/handler.py /app/handler.py

ENV MAX_MODEL_LEN="16384" \
    TENSOR_PARALLEL_SIZE="1" \
    GPU_MEMORY_UTILIZATION="0.95" \
    DTYPE="auto"

CMD ["python3", "-u", "/app/handler.py"]
