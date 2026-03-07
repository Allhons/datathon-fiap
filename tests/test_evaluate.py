"""
Testes unitários para o módulo de avaliação de modelos.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.models.evaluate import (
    calcular_metricas,
    imprimir_relatorio,
    plotar_matriz_confusao,
    plotar_curva_roc,
    salvar_relatorio_comparativo,
    avaliar_modelo,
)


@pytest.fixture
def dados_classificacao():
    """Dados sintéticos simples para testes de avaliação."""
    np.random.seed(42)
    y_real = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.6, 0.85, 0.4, 0.15])
    return y_real, y_pred, y_prob


@pytest.fixture
def modelo_treinado():
    """Modelo sklearn simples, já treinado."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    modelo = LogisticRegression(random_state=42)
    modelo.fit(X, y)
    return modelo, pd.DataFrame(X), pd.Series(y)


class TestCalcularMetricas:
    def test_retorna_dict_com_chaves_esperadas(self, dados_classificacao):
        y_real, y_pred, _ = dados_classificacao
        metricas = calcular_metricas(y_real, y_pred)
        assert "f1_macro" in metricas
        assert "acuracia" in metricas
        assert "precisao_macro" in metricas
        assert "recall_macro" in metricas

    def test_roc_auc_calculado_com_proba(self, dados_classificacao):
        y_real, y_pred, y_prob = dados_classificacao
        metricas = calcular_metricas(y_real, y_pred, y_prob)
        assert "roc_auc" in metricas
        assert metricas["roc_auc"] is not None

    def test_sem_probabilidade(self, dados_classificacao):
        y_real, y_pred, _ = dados_classificacao
        metricas = calcular_metricas(y_real, y_pred, y_prob=None)
        assert "roc_auc" not in metricas


class TestImprimirRelatorio:
    def test_imprime_sem_erro(self, dados_classificacao, capsys):
        y_real, y_pred, _ = dados_classificacao
        metricas = calcular_metricas(y_real, y_pred)
        imprimir_relatorio("ModeloTeste", metricas, y_real, y_pred)
        capturado = capsys.readouterr()
        assert "ModeloTeste" in capturado.out


class TestPlotarMatrizConfusao:
    def test_nao_salva_quando_salvar_false(self, dados_classificacao):
        """Deve gerar o plot sem tentar salvar arquivo."""
        y_real, y_pred, _ = dados_classificacao
        # salvar=False não deve levantar exceção mesmo sem diretório
        plotar_matriz_confusao(y_real, y_pred, "ModeloTeste", salvar=False)

    def test_salva_arquivo(self, dados_classificacao, tmp_path, monkeypatch):
        """Deve salvar o PNG no diretório mockado."""
        y_real, y_pred, _ = dados_classificacao
        monkeypatch.chdir(tmp_path)
        (tmp_path / "reports" / "figures").mkdir(parents=True)

        with patch("src.models.evaluate.caminho_absoluto", return_value=tmp_path / "reports" / "figures"):
            plotar_matriz_confusao(y_real, y_pred, "Teste", salvar=True)


class TestPlotarCurvaROC:
    def test_nao_plota_sem_probabilidade(self, dados_classificacao):
        """Com y_prob=None deve retornar imediatamente."""
        y_real, _, _ = dados_classificacao
        # Não deve lançar exceção
        plotar_curva_roc(y_real, None, "ModeloTeste", salvar=False)

    def test_plota_com_probabilidade(self, dados_classificacao):
        y_real, _, y_prob = dados_classificacao
        plotar_curva_roc(y_real, y_prob, "ModeloTeste", salvar=False)

    def test_salva_arquivo(self, dados_classificacao, tmp_path):
        y_real, _, y_prob = dados_classificacao
        with patch("src.models.evaluate.caminho_absoluto", return_value=tmp_path / "reports" / "figures"):
            (tmp_path / "reports" / "figures").mkdir(parents=True)
            plotar_curva_roc(y_real, y_prob, "Teste", salvar=True)


class TestSalvarRelatorioComparativo:
    def test_cria_csv(self, tmp_path):
        resultados = [
            {"modelo": "A", "f1_macro": 0.9, "acuracia": 0.9},
            {"modelo": "B", "f1_macro": 0.85, "acuracia": 0.88},
        ]
        with patch("src.models.evaluate.caminho_absoluto", return_value=tmp_path):
            destino = salvar_relatorio_comparativo(resultados)
        assert (tmp_path / "comparacao_modelos.csv").exists()

    def test_ordenado_por_f1_macro(self, tmp_path, capsys):
        resultados = [
            {"modelo": "B", "f1_macro": 0.7, "acuracia": 0.7},
            {"modelo": "A", "f1_macro": 0.95, "acuracia": 0.95},
        ]
        with patch("src.models.evaluate.caminho_absoluto", return_value=tmp_path):
            salvar_relatorio_comparativo(resultados)
        capturado = capsys.readouterr()
        # "A" deve aparecer antes de "B" no output
        pos_a = capturado.out.find("A")
        pos_b = capturado.out.find("B")
        assert pos_a < pos_b


class TestAvaliarModelo:
    def test_retorna_dict_com_modelo(self, modelo_treinado):
        modelo, X, y = modelo_treinado
        resultado = avaliar_modelo("LR", modelo, X, y, gerar_graficos=False)
        assert resultado["modelo"] == "LR"
        assert "f1_macro" in resultado

    def test_gera_graficos_sem_erro(self, modelo_treinado, tmp_path):
        modelo, X, y = modelo_treinado
        with patch("src.models.evaluate.caminho_absoluto", return_value=tmp_path):
            (tmp_path / "reports" / "figures").mkdir(parents=True)
            resultado = avaliar_modelo("LR", modelo, X, y, gerar_graficos=True)
        assert "f1_macro" in resultado
