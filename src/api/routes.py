"""
Rotas da API FastAPI.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from src.models.predict import prever, carregar_modelo

logger = logging.getLogger(__name__)
router = APIRouter()


class DadosEstudante(BaseModel):
    # Campos principais
    fase: Optional[float] = 5.0
    ano_nascimento: Optional[float] = 2005.0
    idade: Optional[float] = 15.0
    genero: Optional[str] = "F"
    ano_ingresso: Optional[float] = 2020.0
    instituicao_de_ensino: Optional[str] = "EMEF"
    pedra_20: Optional[str] = "Quartzo"
    pedra_21: Optional[str] = "Quartzo"
    pedra: Optional[str] = "Quartzo"
    inde: Optional[float] = 5.0
    cg: Optional[float] = 5.0
    cf: Optional[float] = 5.0
    ct: Optional[float] = 5.0
    n_av: Optional[float] = 2.0
    iaa: Optional[float] = 5.0
    ieg: Optional[float] = 5.0
    ips: Optional[float] = 5.0
    ida: Optional[float] = 5.0
    nota_matematica: Optional[float] = 5.0
    nota_portugues: Optional[float] = 5.0
    nota_ingles: Optional[float] = 5.0
    indicado: Optional[str] = "Não"
    atingiu_pv: Optional[str] = "Não"
    ipv: Optional[float] = 5.0
    ian: Optional[float] = 5.0
    pedra_22: Optional[str] = "Quartzo"
    inde_22: Optional[float] = 5.0
    ipp: Optional[float] = 5.0
    pedra_23: Optional[str] = "Quartzo"
    inde_23: Optional[float] = 5.0
    avaliador5: Optional[float] = 5.0
    avaliador6: Optional[float] = 5.0
    escola: Optional[str] = "EMEF"
    status: Optional[str] = "Ativo"
    status_2: Optional[str] = "Ativo"
    media_notas: Optional[float] = 5.0
    score_indices: Optional[float] = 5.0
    gap_fase: Optional[float] = 0.0
    anos_no_programa: Optional[float] = 2.0


@router.get("/health")
def health_check():
    try:
        carregar_modelo()
        modelo_carregado = True
    except Exception:
        modelo_carregado = False

    return {
        "status": "ok",
        "modelo_carregado": modelo_carregado
    }


@router.post("/predict")
def predict(dados: DadosEstudante):
    try:
        dados_dict = dados.model_dump()
        resultado = prever(dados_dict)
        return {
            "status": "sucesso",
            "dados": resultado,
            "versao_modelo": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Erro ao realizar predição: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erro ao realizar predição: {str(e)}")
