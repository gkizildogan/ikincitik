# Encoder: bge-m3 (dense/sparse/ColBERT), bge-reranker-v2-m3, jina-reranker-v3.
# torch is the CUDA 12.6 build: the same image runs on the CPU (service encoder) and on
# the GPU (service encoder-gpu, ENCODER_DEVICE=cuda). Models are mounted, never baked in.
FROM python:3.12-slim@sha256:e695ab189fa2d082b1c61c69a6b888a5adeca61ceeccad107b1ed2a125721a9e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_MODULES_CACHE=/tmp/hf_modules \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt

COPY app.py download_models.py ./

EXPOSE 8081
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]
