"""
Módulo de pré-processamento de dados.

Responsabilidades:
  - Unificar os 3 DataFrames anuais em um único dataset
  - Tratar valores ausentes
  - Remover duplicatas e colunas irrelevantes
  - Converter tipos
  - Criar a variável alvo binária: 1 = em risco (defasagem < 0), 0 = sem risco
  - Salvar o dataset processado em data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import carregar_config, configurar_logger, caminho_absoluto

log = configurar_logger("preprocessing")


def unificar_datasets(dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Combina os DataFrames dos 3 anos em um único, mantendo apenas
    as colunas que existem em pelo menos 2 dos anos.

    Returns:
        DataFrame unificado.
    """
    log.info("Unificando datasets de 2022, 2023 e 2024 ...")
    lista = list(dfs.values())
    df_unificado = pd.concat(lista, ignore_index=True, sort=False)
    log.info(f"  → Dataset unificado: {df_unificado.shape}")
    return df_unificado


def remover_colunas_irrelevantes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas de identificação, texto livre e avaliadores."""
    config = carregar_config()
    colunas_dropar = config["features"]["colunas_dropar"]

    # Mantém apenas as que existem no DataFrame atual
    existentes = [c for c in colunas_dropar if c in df.columns]
    df = df.drop(columns=existentes)
    log.info(f"  → {len(existentes)} colunas removidas: {existentes}")
    return df


def tratar_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estratégia de imputação:
      - Numéricas → mediana (robusto a outliers)
      - Categóricas → moda (valor mais frequente)
    """
    log.info("Tratando valores ausentes ...")

    # Colunas com mais de 80% de missing são descartadas
    limiar_missing = 0.80
    proporcao_missing = df.isnull().mean()
    colunas_alta_missing = proporcao_missing[proporcao_missing >
                                             limiar_missing].index.tolist()
    if colunas_alta_missing:
        df = df.drop(columns=colunas_alta_missing)
        log.warning(
            f"  → Removidas {len(colunas_alta_missing)} colunas com >{limiar_missing*100:.0f}% missing: {colunas_alta_missing}")

    # Numéricas → mediana
    colunas_num = df.select_dtypes(include=[np.number]).columns
    for col in colunas_num:
        if df[col].isnull().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    # Categóricas → moda
    colunas_cat = df.select_dtypes(include=["object", "category"]).columns
    for col in colunas_cat:
        if df[col].isnull().any():
            moda = df[col].mode()
            if len(moda) > 0:
                df[col] = df[col].fillna(moda[0])

    log.info(f"  → Missing restante: {df.isnull().sum().sum()} valores")
    return df


def criar_variavel_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a coluna 'alvo' (classificação binária):
      - 0 = sem risco de defasagem (defasagem >= 0)
      - 1 = em risco de defasagem    (defasagem < 0)

    Remove a coluna 'defasagem' original após criar o alvo.
    """
    log.info("Criando variável alvo binária ...")

    if "defasagem" not in df.columns:
        raise ValueError("Coluna 'defasagem' não encontrada no DataFrame.")

    # Converte para numérico (pode vir como string)
    df["defasagem"] = pd.to_numeric(df["defasagem"], errors="coerce")

    df["alvo"] = (df["defasagem"] < 0).astype(int)

    distribuicao = df["alvo"].value_counts()
    log.info(f"  → Distribuição do alvo: {distribuicao.to_dict()}")
    log.info(f"  → % em risco: {df['alvo'].mean()*100:.1f}%")

    # Remove coluna original de defasagem
    df = df.drop(columns=["defasagem"])
    return df


def converter_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstrói os tipos corretos das colunas:
      1. Tenta converter cada coluna object para numérico (float)
      2. Colunas que não são numéricas ficam como category
      3. Colunas numéricas densas ficam como float64
    """
    for col in df.columns:
        if col == "alvo":
            continue
        serie_numerica = pd.to_numeric(df[col], errors="coerce")
        # Se pelo menos 70% dos valores converteram, trata como numérica
        proporcao_numerica = serie_numerica.notna().mean()
        if proporcao_numerica >= 0.70:
            df[col] = serie_numerica
        else:
            # Trata como categórica
            df[col] = df[col].astype("category")
    return df


def preprocessar(dfs: dict[int, pd.DataFrame], salvar: bool = True) -> pd.DataFrame:
    """
    Pipeline completa de pré-processamento.

    Args:
        dfs: Dicionário {ano: DataFrame} vindo da ingestão
        salvar: se True, persiste o resultado em data/processed/

    Returns:
        DataFrame processado e pronto para feature engineering.
    """
    log.info("=== Iniciando pré-processamento ===")
    df = unificar_datasets(dfs)
    df = remover_colunas_irrelevantes(df)
    df = criar_variavel_alvo(df)
    df = tratar_missing(df)
    df = converter_tipos(df)

    log.info(f"Pré-processamento concluído. Shape final: {df.shape}")
    log.info(f"Colunas finais: {df.columns.tolist()}")

    if salvar:
        config = carregar_config()
        destino = caminho_absoluto(
            config["caminhos"]["dados_processados"]) / "dataset_processado.parquet"
        # Converte categorias para string antes de salvar (compatibilidade pyarrow)
        df_salvar = df.copy()
        for col in df_salvar.select_dtypes(include=["category"]).columns:
            df_salvar[col] = df_salvar[col].astype(str)
        df_salvar.to_parquet(destino, index=False)
        log.info(f"  → Dataset processado salvo em: {destino}")

    return df


if __name__ == "__main__":
    from src.data.ingestion import ingerir_dados
    dfs = ingerir_dados(salvar_raw=True)
    df_proc = preprocessar(dfs, salvar=True)
    print(df_proc.head())
    print(df_proc.dtypes)
