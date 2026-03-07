"""
Testes unitários para os módulos de treinamento, avaliação e predição.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from unittest.mock import patch, MagicMock

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dados_binarios():
    """Dados simples para testes de modelo binário."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name="alvo")
    return X, y


@pytest.fixture
def pipeline_simples(dados_binarios):
    """Pipeline treinado para testes."""
    X, y = dados_binarios
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", RandomForestClassifier(n_estimators=10, random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def modelo_salvo(tmp_path, pipeline_simples):
    """Salva o modelo em arquivo temporário e retorna o caminho."""
    caminho = tmp_path / "modelo_teste.pkl"
    joblib.dump(pipeline_simples, caminho)
    return caminho


# ─── Testes de Avaliação ──────────────────────────────────────────────────────

class TestCalcularMetricas:
    def test_retorna_dict_com_metricas(self, dados_binarios, pipeline_simples):
        from src.models.evaluate import calcular_metricas
        X, y = dados_binarios
        y_pred = pipeline_simples.predict(X)
        y_prob = pipeline_simples.predict_proba(X)[:, 1]
        metricas = calcular_metricas(y, y_pred, y_prob)
        assert isinstance(metricas, dict)
        assert "f1_macro" in metricas
        assert "acuracia" in metricas
        assert "roc_auc" in metricas

    def test_metricas_entre_zero_e_um(self, dados_binarios, pipeline_simples):
        from src.models.evaluate import calcular_metricas
        X, y = dados_binarios
        y_pred = pipeline_simples.predict(X)
        metricas = calcular_metricas(y, y_pred)
        for k, v in metricas.items():
            if v is not None:
                assert 0.0 <= v <= 1.0, f"Métrica {k} fora do intervalo [0,1]: {v}"

    def test_funciona_sem_probabilidades(self, dados_binarios, pipeline_simples):
        from src.models.evaluate import calcular_metricas
        X, y = dados_binarios
        y_pred = pipeline_simples.predict(X)
        metricas = calcular_metricas(y, y_pred, y_prob=None)
        assert "f1_macro" in metricas
        assert "roc_auc" not in metricas


# ─── Testes de Predição ───────────────────────────────────────────────────────

class TestPrever:
    def test_retorna_dict_com_campos_corretos(self, modelo_salvo):
        from src.models.predict import prever, limpar_cache_modelo
        limpar_cache_modelo()

        with patch("src.models.predict.carregar_config") as mock_cfg, \
                patch("src.models.predict.caminho_absoluto") as mock_path:
            mock_cfg.return_value = {
                "caminhos": {"modelos": "models"},
                "modelos": {"nome_arquivo": "modelo_teste.pkl"},
            }
            mock_path.return_value = modelo_salvo.parent

            resultado = prever({
                "feature1": 1.0,
                "feature2": -0.5,
                "feature3": 0.3,
            })

        assert "predicao" in resultado
        assert "probabilidade_risco" in resultado
        assert "classificacao" in resultado

    def test_predicao_binaria(self, modelo_salvo):
        from src.models.predict import prever, limpar_cache_modelo
        limpar_cache_modelo()

        with patch("src.models.predict.carregar_config") as mock_cfg, \
                patch("src.models.predict.caminho_absoluto") as mock_path:
            mock_cfg.return_value = {
                "caminhos": {"modelos": "models"},
                "modelos": {"nome_arquivo": "modelo_teste.pkl"},
            }
            mock_path.return_value = modelo_salvo.parent

            resultado = prever(
                {"feature1": 1.0, "feature2": -0.5, "feature3": 0.3})

        assert resultado["predicao"] in {0, 1}

    def test_probabilidade_entre_zero_e_um(self, modelo_salvo):
        from src.models.predict import prever, limpar_cache_modelo
        limpar_cache_modelo()

        with patch("src.models.predict.carregar_config") as mock_cfg, \
                patch("src.models.predict.caminho_absoluto") as mock_path:
            mock_cfg.return_value = {
                "caminhos": {"modelos": "models"},
                "modelos": {"nome_arquivo": "modelo_teste.pkl"},
            }
            mock_path.return_value = modelo_salvo.parent

            resultado = prever(
                {"feature1": 1.0, "feature2": -0.5, "feature3": 0.3})

        assert 0.0 <= resultado["probabilidade_risco"] <= 1.0

    def test_raise_modelo_nao_encontrado(self, tmp_path):
        from src.models.predict import prever, limpar_cache_modelo
        limpar_cache_modelo()

        with patch("src.models.predict.carregar_config") as mock_cfg, \
                patch("src.models.predict.caminho_absoluto") as mock_path:
            mock_cfg.return_value = {
                "caminhos": {"modelos": "models"},
                "modelos": {"nome_arquivo": "modelo_inexistente.pkl"},
            }
            mock_path.return_value = tmp_path

            with pytest.raises(FileNotFoundError):
                prever({"feature1": 1.0})


# ─── Testes de construção de pipeline ────────────────────────────────────────

class TestConstruirPipeline:
    def test_retorna_pipeline_sklearn(self):
        from src.models.train import construir_pipeline
        from sklearn.compose import ColumnTransformer
        preprocessador = MagicMock(spec=ColumnTransformer)
        pipeline = construir_pipeline(preprocessador, "RandomForestClassifier")
        assert isinstance(pipeline, Pipeline)

    def test_raise_modelo_desconhecido(self):
        from src.models.train import construir_pipeline
        with pytest.raises(ValueError, match="não disponível"):
            construir_pipeline(MagicMock(), "ModeloInexistente")
