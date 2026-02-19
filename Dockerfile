FROM python:3.10-slim

WORKDIR /app

# curl: ollama healthcheck için gerekli
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "hisse_bilgi_özel.py"]