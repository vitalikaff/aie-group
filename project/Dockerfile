FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POLYP_DEVICE=cpu

WORKDIR /app

# Системные зависимости для opencv / pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["python", "-m", "src.service"]
