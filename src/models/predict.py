"""
Módulo de inferência/predição.

Responsabilidades:
  - Carregar o modelo serializado
  - Receber dados de entrada, validar e transformar
  - Retornar a previsão (0/1) e a probabilidade de risco
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_defasagem.pkl"
LOG_PATH = BASE_DIR / "logs" / "predictions.csv"

# Garante que a pasta logs existe
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_modelo = None


def carregar_modelo():
    global _modelo
    if _modelo is None:
        logger.info(f"Carregando modelo de: {MODEL_PATH}")
        _modelo = joblib.load(MODEL_PATH)
        logger.info("Modelo carregado com sucesso.")
    return _modelo


def predizer(dados: dict) -> dict:
    """Recebe um dicionário com os dados do estudante e retorna a predição."""
    modelo = carregar_modelo()

    df = pd.DataFrame([dados])

    # Predição
    predicao = int(modelo.predict(df)[0])

    try:
        proba = float(modelo.predict_proba(df)[0][1])
    except Exception:
        proba = None

    classificacao = "COM risco de defasagem" if predicao == 1 else "SEM risco de defasagem"

    # ── Salvar log da predição ──────────────────────────────────────────────
    registro = {
        "timestamp": datetime.utcnow().isoformat(),
        "predicao": predicao,
        "probabilidade_risco": proba,
        "classificacao": classificacao,
        **dados
    }

    df_registro = pd.DataFrame([registro])

    if LOG_PATH.exists():
        df_registro.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_registro.to_csv(LOG_PATH, mode="w", header=True, index=False)

    logger.info(
        f"Predição registrada: {classificacao} (proba={proba:.4f})" if proba else f"Predição registrada: {classificacao}")

    return {
        "predicao": predicao,
        "probabilidade_risco": round(proba, 4) if proba is not None else None,
        "classificacao": classificacao
    }


# Alias para compatibilidade com routes.py
prever = predizer
