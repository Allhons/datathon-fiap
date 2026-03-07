"""
Rotas da API FastAPI.
"""
from datetime import datetime
from src.api.github_logger import push_prediction_to_github
from fastapi import APIRouter, HTTPException
from src.api.schemas import DadosEstudante, RespostaPredicao, RespostaSaude
from src.models.predict import prever, carregar_modelo
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=RespostaSaude)
def health_check():
    try:
        carregar_modelo()
        modelo_carregado = True
    except Exception:
        modelo_carregado = False

    return {
        "status": "ok",
        "modelo_carregado": modelo_carregado,
        "versao": "1.0.0"
    }


@router.post("/predict", response_model=RespostaPredicao)
def predict(dados: DadosEstudante):
    try:
        dados_dict = dados.model_dump()
        resultado = prever(dados_dict)

        # --- Log no GitHub ---
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "predicao": resultado["predicao"],
            "probabilidade_risco": resultado["probabilidade_risco"],
            "classificacao": resultado["classificacao"],
            **dados_dict  # inclui os dados do estudante
        }
        push_prediction_to_github(record)
        # ---------------------

        return {
            "status": "sucesso",
            "dados": resultado,
            "versao_modelo": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Erro ao realizar predição: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erro ao realizar predição: {str(e)}")
