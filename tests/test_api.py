"""
Testes unitários para a API FastAPI.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def cliente():
    """Cria um cliente de teste para a API com o modelo mockado."""
    resultado_mock = {
        "predicao": 1,
        "probabilidade_risco": 0.78,
        "classificacao": "EM RISCO de defasagem",
    }

    with patch("src.api.routes.prever", return_value=resultado_mock), \
            patch("src.api.routes.carregar_modelo", return_value=MagicMock()):
        from src.api.app import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def payload_valido():
    """Payload de exemplo com dados válidos de um aluno."""
    return {
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


# ─── Testes do endpoint raiz ─────────────────────────────────────────────────

class TestRaiz:
    def test_status_200(self, cliente):
        resposta = cliente.get("/")
        assert resposta.status_code == 200

    def test_retorna_mensagem(self, cliente):
        resposta = cliente.get("/")
        dados = resposta.json()
        assert "mensagem" in dados
        assert "versao" in dados
        assert "documentacao" in dados


# ─── Testes do health check ──────────────────────────────────────────────────

class TestHealthCheck:
    def test_status_200(self, cliente):
        resposta = cliente.get("/api/v1/health")
        assert resposta.status_code == 200

    def test_retorna_campos_obrigatorios(self, cliente):
        resposta = cliente.get("/api/v1/health")
        dados = resposta.json()
        assert "status" in dados
        assert "modelo_carregado" in dados
        assert "versao" in dados


# ─── Testes do endpoint /predict ─────────────────────────────────────────────

class TestPredict:
    def test_status_200_com_payload_valido(self, cliente, payload_valido):
        resposta = cliente.post("/api/v1/predict", json=payload_valido)
        assert resposta.status_code == 200

    def test_retorna_estrutura_correta(self, cliente, payload_valido):
        resposta = cliente.post("/api/v1/predict", json=payload_valido)
        dados = resposta.json()
        assert "status" in dados
        assert "dados" in dados
        assert dados["status"] == "sucesso"

    def test_dados_contem_predicao(self, cliente, payload_valido):
        resposta = cliente.post("/api/v1/predict", json=payload_valido)
        dados = resposta.json()["dados"]
        assert "predicao" in dados
        assert "probabilidade_risco" in dados
        assert "classificacao" in dados

    def test_predicao_binaria(self, cliente, payload_valido):
        resposta = cliente.post("/api/v1/predict", json=payload_valido)
        predicao = resposta.json()["dados"]["predicao"]
        assert predicao in {0, 1}

    def test_probabilidade_entre_zero_e_um(self, cliente, payload_valido):
        resposta = cliente.post("/api/v1/predict", json=payload_valido)
        prob = resposta.json()["dados"]["probabilidade_risco"]
        assert 0.0 <= prob <= 1.0

    def test_payload_vazio_aceito(self, cliente):
        """Todos os campos são opcionais — o modelo deve lidar com ausências."""
        resposta = cliente.post("/api/v1/predict", json={})
        assert resposta.status_code == 200

    def test_valores_invalidos_retornam_422(self, cliente):
        """Nota acima de 10 deve falhar na validação Pydantic."""
        payload_invalido = {"nota_matematica": 999}
        resposta = cliente.post("/api/v1/predict", json=payload_invalido)
        assert resposta.status_code == 422

    def test_erro_interno_retorna_500(self, payload_valido):
        """Quando o modelo lança exceção, a API deve retornar 500."""
        with patch("src.api.routes.prever", side_effect=RuntimeError("Erro de teste")), \
                patch("src.api.routes.carregar_modelo", return_value=MagicMock()):
            from src.api.app import app
            with TestClient(app) as client:
                resposta = client.post("/api/v1/predict", json=payload_valido)
                assert resposta.status_code == 500
