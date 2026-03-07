"""
Funções utilitárias compartilhadas entre os módulos do projeto.
"""

import yaml
import logging
from pathlib import Path
from loguru import logger


# Diretório raiz do projeto (dois níveis acima deste arquivo: src/ -> raiz)
RAIZ_PROJETO = Path(__file__).resolve().parent.parent


def carregar_config(caminho_config: str = None) -> dict:
    """Carrega o arquivo de configuração YAML do projeto."""
    if caminho_config is None:
        caminho_config = RAIZ_PROJETO / "configs" / "config.yaml"
    with open(caminho_config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configurar_logger(nome_modulo: str, nivel: str = "INFO") -> "logger":
    """
    Configura e retorna um logger usando loguru.
    Grava logs no console e em arquivo na pasta logs/.
    """
    caminho_log = RAIZ_PROJETO / "logs" / f"{nome_modulo}.log"
    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove handlers padrão
    logger.add(
        sink=str(caminho_log),
        level=nivel,
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    )
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=nivel,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {name} | {message}",
    )
    return logger


def caminho_absoluto(caminho_relativo: str) -> Path:
    """Retorna o caminho absoluto a partir da raiz do projeto."""
    return RAIZ_PROJETO / caminho_relativo
