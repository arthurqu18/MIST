# Build stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as builder

WORKDIR /app

# Instala Python e dependências de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Aceita HF_TOKEN como build argument
ARG HF_TOKEN=""
ARG TABPFN_TOKEN=""

# Copia e instala dependências em um diretório temporário
COPY requirements.txt .
RUN pip3 install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Instala Python na stage de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Aceita HF_TOKEN como build argument para a stage de runtime
ARG HF_TOKEN=""
ARG TABPFN_TOKEN=""

# Copia apenas os pacotes instalados do builder
COPY --from=builder /root/.local /root/.local

# Atualiza PATH para incluir os pacotes do usuário
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_TOKEN=${HF_TOKEN} \
    TABPFN_TOKEN=${TABPFN_TOKEN}

# Copia código da aplicação
COPY . .

EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]