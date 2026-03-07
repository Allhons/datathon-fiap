"""
Testes unitários para o módulo de pré-processamento.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.data.preprocessing import (
    unificar_datasets,
    remover_colunas_irrelevantes,
    tratar_missing,
    criar_variavel_alvo,
    converter_tipos,
)


@pytest.fixture
def df_2022():
    """DataFrame simulando a aba PEDE2022 após a ingestão."""
    return pd.DataFrame({
        "ra": ["RA-1", "RA-2", "RA-3", "RA-4", "RA-5"],
        "fase": ["Fase 3", "Fase 5", "Fase 4", "Fase 2", "Fase 7"],
        "inde": [6.5, 7.2, 5.1, 8.0, 4.3],
        "iaa": [7.0, 6.5, 5.5, 8.5, 4.0],
        "ieg": [5.5, 6.0, 4.5, 7.5, 3.5],
        "nota_matematica": [6.0, 7.5, 4.5, 9.0, 3.0],
        "defasagem": [0, 0, -1, 0, -2],
        "genero": ["Masculino", "Feminino", "Masculino", "Feminino", "Masculino"],
        "ano": [2022, 2022, 2022, 2022, 2022],
        "nome": ["Aluno-1", "Aluno-2", "Aluno-3", "Aluno-4", "Aluno-5"],
    })


@pytest.fixture
def df_2023():
    """DataFrame simulando a aba PEDE2023 após a ingestão."""
    return pd.DataFrame({
        "ra": ["RA-6", "RA-7", "RA-8"],
        "fase": ["Fase 1", "Fase 6", "Fase 3"],
        "inde": [7.0, 6.0, 5.5],
        "iaa": [7.5, 5.5, 6.0],
        "ieg": [6.0, 5.0, 4.0],
        "nota_matematica": [7.0, 5.0, 4.0],
        "defasagem": [0, -1, 0],
        "genero": ["Feminino", "Masculino", "Feminino"],
        "ano": [2023, 2023, 2023],
        "nome": ["Aluno-6", "Aluno-7", "Aluno-8"],
    })


class TestUnificarDatasets:
    def test_retorna_dataframe(self, df_2022, df_2023):
        resultado = unificar_datasets({2022: df_2022, 2023: df_2023})
        assert isinstance(resultado, pd.DataFrame)

    def test_linhas_somadas(self, df_2022, df_2023):
        resultado = unificar_datasets({2022: df_2022, 2023: df_2023})
        assert len(resultado) == len(df_2022) + len(df_2023)

    def test_colunas_preservadas(self, df_2022, df_2023):
        resultado = unificar_datasets({2022: df_2022, 2023: df_2023})
        for col in df_2022.columns:
            assert col in resultado.columns


class TestRemoverColunas:
    def test_remove_coluna_nome(self, df_2022):
        with patch("src.data.preprocessing.carregar_config") as mock_cfg:
            mock_cfg.return_value = {"features": {
                "colunas_dropar": ["nome", "ra"]}}
            resultado = remover_colunas_irrelevantes(df_2022)
            assert "nome" not in resultado.columns
            assert "ra" not in resultado.columns

    def test_colunas_inexistentes_nao_causam_erro(self, df_2022):
        with patch("src.data.preprocessing.carregar_config") as mock_cfg:
            mock_cfg.return_value = {"features": {
                "colunas_dropar": ["coluna_que_nao_existe"]}}
            resultado = remover_colunas_irrelevantes(df_2022)
            assert isinstance(resultado, pd.DataFrame)


class TestTratarMissing:
    def test_sem_missing_apos_tratamento(self, df_2022):
        df = df_2022.copy()
        df.loc[0, "inde"] = np.nan
        df.loc[1, "genero"] = np.nan
        resultado = tratar_missing(df)
        assert resultado["inde"].isnull().sum() == 0
        assert resultado["genero"].isnull().sum() == 0

    def test_remove_colunas_com_muito_missing(self):
        df = pd.DataFrame({
            "coluna_ok": [1, 2, 3, 4, 5, 6],
            # 83% missing
            "coluna_ruim": [np.nan, np.nan, np.nan, np.nan, np.nan, 1],
        })
        resultado = tratar_missing(df)
        assert "coluna_ruim" not in resultado.columns
        assert "coluna_ok" in resultado.columns


class TestCriarVariavelAlvo:
    def test_defasagem_negativa_vira_1(self, df_2022):
        resultado = criar_variavel_alvo(df_2022)
        # RA-3 tem defasagem=-1 → alvo=1
        assert resultado.loc[resultado.index[2], "alvo"] == 1

    def test_defasagem_zero_vira_0(self, df_2022):
        resultado = criar_variavel_alvo(df_2022)
        # RA-1 tem defasagem=0 → alvo=0
        assert resultado.loc[resultado.index[0], "alvo"] == 0

    def test_coluna_defasagem_removida(self, df_2022):
        resultado = criar_variavel_alvo(df_2022)
        assert "defasagem" not in resultado.columns

    def test_coluna_alvo_criada(self, df_2022):
        resultado = criar_variavel_alvo(df_2022)
        assert "alvo" in resultado.columns

    def test_raise_sem_coluna_defasagem(self):
        df_sem_alvo = pd.DataFrame({"inde": [6.5, 7.0]})
        with pytest.raises(ValueError, match="defasagem"):
            criar_variavel_alvo(df_sem_alvo)

    def test_valores_binarios(self, df_2022):
        resultado = criar_variavel_alvo(df_2022)
        valores_unicos = set(resultado["alvo"].unique())
        assert valores_unicos.issubset({0, 1})
