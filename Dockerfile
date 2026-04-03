FROM python:3.11-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMAGENT_MODE=lite \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    RAG_WORKING_DIR=/app/omagent/rag_storage \
    TIKTOKEN_CACHE_DIR=/app/tiktoken_cache \
    HF_HOME=/app/model_cache/huggingface \
    XDG_CACHE_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    libmagic1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.base.txt /app/docker/requirements.base.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r /app/docker/requirements.base.txt
RUN python -m pip install --no-deps \
    gradio==5.49.1 \
    gradio_client==1.13.3 \
    openai==1.109.1 \
    litellm==1.82.1 \
    magic-pdf==0.6.1 \
    mineru==2.7.6 \
    qwen-vl-utils==0.0.8

COPY . /app

RUN python -m pip install -e /app/omagent/omagent-core --no-deps
RUN python -m pip install -e /app/rag-anything --no-deps
RUN python /app/rag-anything/scripts/create_tiktoken_cache.py

# 预热 MinerU，确保离线环境首次解析 PDF/图片时不会再下载模型。
RUN mkdir -p /tmp/mineru_warmup \
    && mineru -p /app/doc/Images/效果图1.png -o /tmp/mineru_warmup -m ocr || true \
    && rm -rf /tmp/mineru_warmup

RUN mkdir -p /app/omagent/rag_storage /app/model_cache /app/tiktoken_cache

WORKDIR /app/omagent

EXPOSE 7860

ENTRYPOINT ["python", "run_webpage.py"]
