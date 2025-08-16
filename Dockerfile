FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SENTENCE_TRANSFORMERS_HOME=/huggingface_cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash appuser
RUN mkdir /huggingface_cache && chown -R appuser:appuser /huggingface_cache
COPY requirements.txt /tmp/requirements.txt
COPY pre_cache_models.py /tmp/pre_cache_models.py
RUN pip install --default-timeout=100 --no-cache-dir -r /tmp/requirements.txt
RUN python /tmp/pre_cache_models.py
USER appuser
WORKDIR /workspace
