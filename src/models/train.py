"""
Módulo de treinamento de modelos.

Responsabilidades:
  - Construir pipelines sklearn (preprocessador + modelo)
  - Treinar múltiplos modelos candidatos com cross-validation
  - Registrar experimentos no MLflow
  - Selecionar o melhor modelo pela métrica principal (F1-Macro)
  - Serializar o modelo final com joblib
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.utils import carregar_config, configurar_logger, caminho_absoluto
from src.models.evaluate import avaliar_modelo, salvar_relatorio_comparativo

log = configurar_logger("train")


# Mapeamento de nomes para classes sklearn
MODELOS_DISPONIVEIS = {
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
}

# Hiperparâmetros padrão para cada modelo
PARAMS_PADRAO = {
    "RandomForestClassifier": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
    "GradientBoostingClassifier": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": 42,
    },
    "LogisticRegression": {
        "C": 1.0,
        "max_iter": 500,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "SVC": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,  # Necessário para predict_proba
        "class_weight": "balanced",
        "random_state": 42,
    },
}


def construir_pipeline(preprocessador, nome_modelo: str) -> Pipeline:
    """
    Constrói um Pipeline sklearn combinando o preprocessador
    de features com o modelo classificador.
    """
    if nome_modelo not in MODELOS_DISPONIVEIS:
        raise ValueError(
            f"Modelo '{nome_modelo}' não disponível. Opções: {list(MODELOS_DISPONIVEIS)}")

    params = PARAMS_PADRAO.get(nome_modelo, {})
    classificador = MODELOS_DISPONIVEIS[nome_modelo](**params)

    pipeline = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", classificador),
    ])
    return pipeline


def treinar_com_cv(
    pipeline: Pipeline,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    cv_folds: int = 5,
) -> dict:
    """
    Treina o pipeline final e avalia com cross-validation estratificado.

    Returns:
        Dicionário com médias e desvios das métricas no CV
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    resultados_cv = {}
    for metrica in ["f1_macro", "roc_auc", "accuracy"]:
        scores = cross_val_score(
            pipeline, X_treino, y_treino, cv=cv, scoring=metrica, n_jobs=-1)
        resultados_cv[f"{metrica}_cv_media"] = round(scores.mean(), 4)
        resultados_cv[f"{metrica}_cv_std"] = round(scores.std(), 4)
        log.info(f"  CV {metrica}: {scores.mean():.4f} ± {scores.std():.4f}")

    # Treino final com todos os dados de treino
    pipeline.fit(X_treino, y_treino)
    return resultados_cv


def registrar_mlflow(
    nome_modelo: str,
    pipeline: Pipeline,
    metricas_cv: dict,
    metricas_teste: dict,
    experimento: str = "datathon-passos-magicos",
) -> str:
    """
    Registra o experimento no MLflow.

    Returns:
        run_id do experimento registrado
    """
    mlflow.set_experiment(experimento)

    with mlflow.start_run(run_name=nome_modelo) as run:
        # Parâmetros
        params = PARAMS_PADRAO.get(nome_modelo, {})
        mlflow.log_params(params)
        mlflow.log_param("modelo", nome_modelo)

        # Métricas do CV
        for k, v in metricas_cv.items():
            if v is not None:
                mlflow.log_metric(k, v)

        # Métricas do teste
        for k, v in metricas_teste.items():
            if k != "modelo" and v is not None:
                mlflow.log_metric(f"teste_{k}", v)

        # Artefato: modelo
        mlflow.sklearn.log_model(pipeline, artifact_path="modelo")

        run_id = run.info.run_id
        log.info(f"  → MLflow run_id: {run_id}")

    return run_id


def salvar_modelo(pipeline: Pipeline, nome_arquivo: str = None) -> Path:
    """Serializa o modelo final com joblib."""
    config = carregar_config()
    if nome_arquivo is None:
        nome_arquivo = config["modelos"]["nome_arquivo"]
    destino = caminho_absoluto(config["caminhos"]["modelos"]) / nome_arquivo
    joblib.dump(pipeline, destino)
    log.info(f"  → Modelo salvo em: {destino}")
    return destino


def treinar_todos_modelos(
    X_treino: pd.DataFrame,
    X_val: pd.DataFrame,
    X_teste: pd.DataFrame,
    y_treino: pd.Series,
    y_val: pd.Series,
    y_teste: pd.Series,
    preprocessador,
) -> Pipeline:
    """
    Treina todos os modelos candidatos, avalia no conjunto de teste
    e retorna o pipeline do melhor modelo.

    Returns:
        Pipeline do melhor modelo (maior F1-Macro no teste)
    """
    config = carregar_config()
    modelos_candidatos = config["modelos"]["lista_candidatos"]
    cv_folds = config["avaliacao"]["cv_folds"]

    resultados_teste = []
    melhor_pipeline = None
    melhor_f1 = -1.0

    log.info(f"=== Treinando {len(modelos_candidatos)} modelos candidatos ===")

    for nome in modelos_candidatos:
        log.info(f"\n--- {nome} ---")
        pipeline = construir_pipeline(preprocessador, nome)

        # Cross-validation no treino
        metricas_cv = treinar_com_cv(pipeline, X_treino, y_treino, cv_folds)

        # Avaliação no teste
        metricas_teste = avaliar_modelo(
            nome_modelo=nome,
            modelo=pipeline,
            X_teste=X_teste,
            y_teste=y_teste,
            gerar_graficos=True,
        )

        # Registro no MLflow
        try:
            registrar_mlflow(nome, pipeline, metricas_cv, metricas_teste)
        except Exception as e:
            log.warning(
                f"  MLflow não disponível: {e}. Continuando sem registro.")

        resultados_teste.append({**metricas_teste, **metricas_cv})

        # Atualiza o melhor modelo
        if metricas_teste.get("f1_macro", 0) > melhor_f1:
            melhor_f1 = metricas_teste["f1_macro"]
            melhor_pipeline = pipeline
            melhor_nome = nome

    # Relatório comparativo
    salvar_relatorio_comparativo(resultados_teste)

    log.info(f"\n✓ Melhor modelo: {melhor_nome} (F1-Macro = {melhor_f1:.4f})")
    caminho_modelo = salvar_modelo(melhor_pipeline)
    log.info(f"✓ Modelo final salvo em: {caminho_modelo}")

    return melhor_pipeline


if __name__ == "__main__":
    from src.data.ingestion import ingerir_dados
    from src.data.preprocessing import preprocessar
    from src.features.feature_engineering import executar_feature_engineering

    log.info("Iniciando pipeline completa de treinamento ...")
    dfs = ingerir_dados(salvar_raw=True)
    df_proc = preprocessar(dfs, salvar=True)
    X_treino, X_val, X_teste, y_treino, y_val, y_teste, preprocessador = executar_feature_engineering(
        df_proc)

    melhor_modelo = treinar_todos_modelos(
        X_treino, X_val, X_teste,
        y_treino, y_val, y_teste,
        preprocessador,
    )
    log.info("Pipeline de treinamento concluída com sucesso.")
