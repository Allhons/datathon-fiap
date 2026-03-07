"""
Módulo de engenharia de features.

Responsabilidades:
  - Criar novas features a partir dos dados brutos
  - Separar features numéricas e categóricas
  - Construir e retornar o ColumnTransformer (preprocessador sklearn)
  - Separar X (features) e y (alvo)
  - Dividir em treino/validação/teste
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from src.utils import carregar_config, configurar_logger, caminho_absoluto

log = configurar_logger("feature_engineering")


def para_string(X_arr):
    """Converte array numpy para strings uniformes (picklável em nível de módulo)."""
    if hasattr(X_arr, "toarray"):
        X_arr = X_arr.toarray()
    return np.where(
        (X_arr == None) | (X_arr != X_arr),  # noqa: E711 – element-wise None/NaN check
        "nan",
        X_arr.astype(str),
    )


def criar_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    notas_cols = [c for c in ["nota_matematica",
                              "nota_portugues", "nota_ingles"] if c in df.columns]
    if notas_cols:
        df["media_notas"] = df[notas_cols].mean(axis=1)
        log.info(f"  → Feature 'media_notas' criada a partir de: {notas_cols}")

    # 🚨 ian removido — é leakage (corr = -0.99 com target)
    indices_cols = [c for c in ["iaa", "ieg",
                                "ips", "ipp", "ida"] if c in df.columns]
    if indices_cols:
        df["score_indices"] = df[indices_cols].mean(axis=1)
        log.info(
            f"  → Feature 'score_indices' criada a partir de: {indices_cols}")

    if "ano_ingresso" in df.columns and "ano" in df.columns:
        df["anos_no_programa"] = df["ano"] - df["ano_ingresso"]
        log.info("  → Feature 'anos_no_programa' criada")

    return df


def selecionar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "alvo" not in df.columns:
        raise ValueError(
            "Coluna 'alvo' não encontrada. Execute o pré-processamento antes.")

    colunas_excluir = {
        # Identificação
        "alvo", "ra", "nome", "turma", "ano", "data_nascimento",
        # 🚨 Leakage direto
        "fase_ideal", "num_fase_atual", "num_fase_ideal", "gap_fase",
        "defas", "defasagem",
        # 🚨 Leakage indireto
        "inde", "inde_22", "inde_23", "inde_2022", "inde_2023", "inde_2024",
        "pedra", "pedra_20", "pedra_21", "pedra_22", "pedra_23",
        "pedra_2020", "pedra_2021", "pedra_2022", "pedra_2023", "pedra_2024",
        "indicado", "atingiu_pv", "ipv",
        "status", "status_2",
        # 🚨 IAN = corr -0.99 com target
        "ian",
    }

    colunas_features = [c for c in df.columns if c not in colunas_excluir]
    X = df[colunas_features].copy()
    y = df["alvo"].copy()

    log.info(
        f"Features selecionadas ({len(colunas_features)}): {colunas_features}")
    log.info(f"Distribuição do alvo: {y.value_counts().to_dict()}")
    return X, y


def construir_preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer sklearn com:
      - Numéricas: imputação pela mediana + StandardScaler
      - Categóricas: conversão para string + imputação pela moda + OneHotEncoder

    Returns:
        ColumnTransformer configurado (não ajustado).
    """
    # Identifica tipos de colunas no X
    colunas_num = X.select_dtypes(include=[np.number]).columns.tolist()
    colunas_cat = X.select_dtypes(
        include=["object", "category"]).columns.tolist()

    log.info(f"  → {len(colunas_num)} features numéricas: {colunas_num}")
    log.info(f"  → {len(colunas_cat)} features categóricas: {colunas_cat}")

    pipeline_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    pipeline_cat = Pipeline([
        ("to_string", FunctionTransformer(para_string)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessador = ColumnTransformer(
        transformers=[
            ("num", pipeline_num, colunas_num),
            ("cat", pipeline_cat, colunas_cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessador


def dividir_dados(
    X: pd.DataFrame,
    y: pd.Series,
    prop_teste: float = 0.2,
    prop_validacao: float = 0.1,
    semente: int = 42,
) -> tuple:
    """
    Divide os dados em treino, validação e teste de forma estratificada.

    Returns:
        X_treino, X_val, X_teste, y_treino, y_val, y_teste
    """
    # Primeiro: separa teste
    X_temp, X_teste, y_temp, y_teste = train_test_split(
        X, y, test_size=prop_teste, random_state=semente, stratify=y
    )

    # Segundo: separa validação do restante
    prop_val_ajustada = prop_validacao / (1 - prop_teste)
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_temp, y_temp, test_size=prop_val_ajustada, random_state=semente, stratify=y_temp
    )

    log.info(
        f"Treino:    {X_treino.shape[0]} amostras ({y_treino.mean()*100:.1f}% em risco)")
    log.info(
        f"Validação: {X_val.shape[0]} amostras ({y_val.mean()*100:.1f}% em risco)")
    log.info(
        f"Teste:     {X_teste.shape[0]} amostras ({y_teste.mean()*100:.1f}% em risco)")

    return X_treino, X_val, X_teste, y_treino, y_val, y_teste


def executar_feature_engineering(df: pd.DataFrame) -> tuple:
    """
    Pipeline completa de feature engineering.

    Returns:
        X_treino, X_val, X_teste, y_treino, y_val, y_teste, preprocessador
    """
    log.info("=== Iniciando feature engineering ===")
    config = carregar_config()

    df = criar_features_derivadas(df)
    X, y = selecionar_features(df)
    preprocessador = construir_preprocessador(X)

    splits = dividir_dados(
        X, y,
        prop_teste=config["dados"]["proporcao_teste"],
        prop_validacao=config["dados"]["proporcao_validacao"],
        semente=config["dados"]["semente_aleatoria"],
    )
    X_treino, X_val, X_teste, y_treino, y_val, y_teste = splits

    # Salva o dataset com features derivadas
    destino = caminho_absoluto(
        config["caminhos"]["dados_processados"]) / "dataset_features.parquet"
    df_features = pd.concat([X, y], axis=1)
    # Converte categorias para string antes de salvar (compatibilidade pyarrow)
    for col in df_features.select_dtypes(include=["category"]).columns:
        df_features[col] = df_features[col].astype(str)
    df_features.to_parquet(destino, index=False)
    log.info(f"  → Dataset com features salvo em: {destino}")

    log.info("Feature engineering concluído.")
    return X_treino, X_val, X_teste, y_treino, y_val, y_teste, preprocessador
