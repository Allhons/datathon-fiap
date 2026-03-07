"""
Schemas Pydantic para validação de entrada e saída da API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class DadosEstudante(BaseModel):
    """Schema de entrada: dados de um estudante para predição de risco."""

    inde: Optional[float] = Field(
        None, ge=0, le=10, description="Índice de Desenvolvimento Educacional (0-10)")
    iaa: Optional[float] = Field(
        None, ge=0, le=10, description="Índice de Auto Avaliação (0-10)")
    ieg: Optional[float] = Field(
        None, ge=0, le=10, description="Índice de Engajamento (0-10)")
    ips: Optional[float] = Field(
        None, ge=0, le=10, description="Índice Psicossocial (0-10)")
    ipp: Optional[float] = Field(
        None, ge=0, le=10, description="Índice Psicopedagógico (0-10)")
    ida: Optional[float] = Field(
        None, ge=0, le=10, description="Índice de Desenvolvimento Acadêmico (0-10)")
    ipv: Optional[float] = Field(
        None, ge=0, le=10, description="Índice de Ponto de Virada (0-10)")
    ian: Optional[float] = Field(
        None, ge=0, le=10, description="Índice de Adequação ao Nível (0-10)")
    nota_matematica: Optional[float] = Field(
        None, ge=0, le=10, description="Nota de Matemática (0-10)")
    nota_portugues: Optional[float] = Field(
        None, ge=0, le=10, description="Nota de Português (0-10)")
    nota_ingles: Optional[float] = Field(
        None, ge=0, le=10, description="Nota de Inglês (0-10)")
    idade: Optional[int] = Field(
        None, ge=5, le=25, description="Idade do estudante")
    fase: Optional[str] = Field(
        None, description="Fase atual no programa (ex: 'Fase 3')")
    genero: Optional[str] = Field(None, description="Gênero do estudante")
    pedra: Optional[str] = Field(
        None, description="Classificação Pedra (Quartzo, Ágata, Ametista, Topázio)")
    instituicao_ensino: Optional[str] = Field(
        None, description="Tipo de instituição de ensino")
    ano_ingresso: Optional[int] = Field(
        None, ge=2016, description="Ano de ingresso no programa")

    model_config = {
        "json_schema_extra": {
            "example": {
                "inde": 6.5,
                "iaa": 7.0,
                "ieg": 5.5,
                "ips": 6.0,
                "ipp": 6.5,
                "ida": 7.0,
                "ipv": 6.8,
                "ian": 5.0,
                "nota_matematica": 6.0,
                "nota_portugues": 5.5,
                "nota_ingles": 7.0,
                "idade": 14,
                "fase": "Fase 5",
                "genero": "Masculino",
                "pedra": "Quartzo",
                "instituicao_ensino": "Pública",
                "ano_ingresso": 2020,
            }
        }
    }


class ResultadoPredicao(BaseModel):
    """Schema de saída: resultado da predição."""

    predicao: int = Field(...,
                          description="0 = sem risco, 1 = em risco de defasagem")
    probabilidade_risco: float = Field(...,
                                       description="Probabilidade de estar em risco (0-1)")
    classificacao: str = Field(...,
                               description="Descrição textual do resultado")


class RespostaPredicao(BaseModel):
    """Envelope padrão de resposta da API."""

    status: str = "sucesso"
    dados: ResultadoPredicao
    versao_modelo: str = "1.0.0"


class RespostaSaude(BaseModel):
    """Resposta do endpoint de health check."""

    status: str
    modelo_carregado: bool
    versao: str


class RespostaErro(BaseModel):
    """Schema de resposta em caso de erro."""

    status: str = "erro"
    mensagem: str
    detalhe: Optional[str] = None
