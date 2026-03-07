"""
Instância principal da aplicação FastAPI.

Execução local:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Via Docker:
    docker build -t passos-magicos-api .
    docker run -p 8000:8000 passos-magicos-api
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.models.predict import carregar_modelo
from src.utils import carregar_config, configurar_logger

log = configurar_logger("app")
config = carregar_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    O modelo é carregado uma única vez na inicialização.
    """
    log.info("=== Inicializando API Passos Mágicos ===")
    try:
        carregar_modelo()
        log.info("Modelo carregado com sucesso na inicialização.")
    except FileNotFoundError:
        log.warning(
            "Modelo não encontrado. A API iniciará mas /predict retornará erro até o modelo ser treinado."
        )
    yield
    log.info("=== Encerrando API ===")


app = FastAPI(
    title=config["api"]["titulo"],
    description=(
        "API para predição de risco de defasagem escolar de estudantes da "
        "Associação Passos Mágicos. Desenvolvida como entrega do Datathon FIAP Pós Tech."
    ),
    version=config["api"]["versao"],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware CORS (permite chamadas de qualquer origem — ajuste em produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro das rotas
app.include_router(router, prefix=config["api"]["prefixo"])


@app.get("/", tags=["Root"], summary="Página inicial")
def root():
    """Confirma que a API está em execução."""
    return {
        "mensagem": "API Passos Mágicos — Predição de Risco de Defasagem Escolar",
        "versao": config["api"]["versao"],
        "documentacao": "/docs",
        "saude": f"{config['api']['prefixo']}/health",
    }
