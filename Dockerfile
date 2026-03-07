# ─── Estágio de build ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Instala dependências do sistema necessárias para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Estágio final ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copia as dependências instaladas do estágio de build
COPY --from=builder /install /usr/local

# Copia o código-fonte do projeto
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Cria diretórios necessários em runtime
RUN mkdir -p data/raw data/processed logs reports/figures

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Porta exposta pela API
EXPOSE 8000

# Health check do container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Comando de inicialização
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
