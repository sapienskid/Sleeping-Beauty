FROM python:3.12-slim

# Phase 2: uncomment for OCR support
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     tesseract-ocr \
#     tesseract-ocr-eng \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8800

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8800", "--workers", "1"]
