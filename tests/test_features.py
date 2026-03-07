"""
Testes unitários para o módulo de feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.features.feature_engineering import (
    criar_features_derivadas,
    selecionar_features,
    construir_preprocessador,
    dividir_dados,
)


@pytest.fixture
def df_base():
    """Dataset mínimo para testes de feature engineering."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "inde": np.random.uniform(3, 10, n),
        "iaa": np.random.uniform(3, 10, n),
        "ieg": np.random.uniform(3, 10, n),
        "ips": np.random.uniform(3, 10, n),
        "ipp": np.random.uniform(3, 10, n),
        "ida": np.random.uniform(3, 10, n),
        "ipv": np.random.uniform(3, 10, n),
        "ian": np.random.uniform(3, 10, n),
        "nota_matematica": np.random.uniform(0, 10, n),
        "nota_portugues": np.random.uniform(0, 10, n),
        "nota_ingles": np.random.uniform(0, 10, n),
        "idade": np.random.randint(7, 20, n),
        "fase": np.random.choice(["Fase 1", "Fase 3", "Fase 5"], n),
        "fase_ideal": np.random.choice(["Fase 1", "Fase 3", "Fase 5", "Fase 7"], n),
        "genero": np.random.choice(["Masculino", "Feminino"], n),
        "pedra": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
        "instituicao_ensino": np.random.choice(["Pública", "Privada"], n),
        "ano_ingresso": np.random.randint(2016, 2023, n),
        "ano": np.random.choice([2022, 2023, 2024], n),
        "alvo": np.random.randint(0, 2, n),
    })


class TestCriarFeaturesDerivadas:
    def test_cria_media_notas(self, df_base):
        resultado = criar_features_derivadas(df_base)
        assert "media_notas" in resultado.columns

    def test_cria_score_indices(self, df_base):
        resultado = criar_features_derivadas(df_base)
        assert "score_indices" in resultado.columns

    def test_cria_anos_no_programa(self, df_base):
        resultado = criar_features_derivadas(df_base)
        assert "anos_no_programa" in resultado.columns

    def test_media_notas_entre_zero_e_dez(self, df_base):
        resultado = criar_features_derivadas(df_base)
        assert resultado["media_notas"].between(0, 10).all()

    def test_nao_modifica_original(self, df_base):
        df_copia = df_base.copy()
        criar_features_derivadas(df_base)
        assert "media_notas" not in df_base.columns  # Original não modificado


class TestSelecionarFeatures:
    def test_separa_x_e_y(self, df_base):
        X, y = selecionar_features(df_base)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_alvo_nao_esta_em_x(self, df_base):
        X, _ = selecionar_features(df_base)
        assert "alvo" not in X.columns

    def test_y_tem_valores_binarios(self, df_base):
        _, y = selecionar_features(df_base)
        assert set(y.unique()).issubset({0, 1})

    def test_raise_sem_coluna_alvo(self):
        df_sem_alvo = pd.DataFrame({"inde": [6.5, 7.0]})
        with pytest.raises(ValueError, match="alvo"):
            selecionar_features(df_sem_alvo)


class TestConstruirPreprocessador:
    def test_retorna_column_transformer(self, df_base):
        X, _ = selecionar_features(df_base)
        X = X.drop(columns=["fase_ideal", "num_fase_atual",
                   "num_fase_ideal"], errors="ignore")
        preprocessador = construir_preprocessador(X)
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessador, ColumnTransformer)

    def test_preprocessador_transforma_dados(self, df_base):
        X, y = selecionar_features(df_base)
        preprocessador = construir_preprocessador(X)
        X_transformado = preprocessador.fit_transform(X)
        assert X_transformado.shape[0] == len(X)


class TestDividirDados:
    def test_proporcoes_corretas(self, df_base):
        X, y = selecionar_features(df_base)
        X_tr, X_v, X_te, y_tr, y_v, y_te = dividir_dados(
            X, y, prop_teste=0.2, prop_validacao=0.1)
        total = len(X_tr) + len(X_v) + len(X_te)
        assert total == len(X)

    def test_sem_vazamento_de_dados(self, df_base):
        X, y = selecionar_features(df_base)
        X_tr, X_v, X_te, *_ = dividir_dados(X, y)
        idx_tr = set(X_tr.index)
        idx_v = set(X_v.index)
        idx_te = set(X_te.index)
        assert len(idx_tr & idx_v) == 0
        assert len(idx_tr & idx_te) == 0
        assert len(idx_v & idx_te) == 0


class TestExecutarFeatureEngineering:
    def test_retorna_sete_elementos(self, df_base):
        from src.features.feature_engineering import executar_feature_engineering
        resultado = executar_feature_engineering(df_base)
        # X_tr, X_v, X_te, y_tr, y_v, y_te, preprocessador
        assert len(resultado) == 7

    def test_shapes_consistentes(self, df_base):
        from src.features.feature_engineering import executar_feature_engineering
        X_tr, X_v, X_te, y_tr, y_v, y_te, prep = executar_feature_engineering(
            df_base)
        assert len(X_tr) == len(y_tr)
        assert len(X_v) == len(y_v)
        assert len(X_te) == len(y_te)
        assert len(X_tr) + len(X_v) + len(X_te) == len(df_base)
