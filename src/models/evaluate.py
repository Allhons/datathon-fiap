"""
Módulo de avaliação de modelos.

Responsabilidades:
  - Calcular todas as métricas exigidas (F1-macro, AUC-ROC, Accuracy, etc.)
  - Gerar relatório comparativo entre modelos
  - Plotar curvas ROC e matriz de confusão
  - Salvar relatório em reports/
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)

from src.utils import carregar_config, configurar_logger, caminho_absoluto

log = configurar_logger("evaluate")


def calcular_metricas(y_real: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Calcula o conjunto completo de métricas de avaliação.

    Args:
        y_real:  valores reais (0/1)
        y_pred:  previsões binárias do modelo
        y_prob:  probabilidades previstas para a classe positiva (opcional)

    Returns:
        Dicionário com todas as métricas
    """
    metricas = {
        "f1_macro":         round(f1_score(y_real, y_pred, average="macro", zero_division=0), 4),
        "f1_ponderado":     round(f1_score(y_real, y_pred, average="weighted", zero_division=0), 4),
        "acuracia":         round(accuracy_score(y_real, y_pred), 4),
        "precisao_macro":   round(precision_score(y_real, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":     round(recall_score(y_real, y_pred, average="macro", zero_division=0), 4),
        "f1_classe_risco":  round(f1_score(y_real, y_pred, pos_label=1, zero_division=0), 4),
    }

    if y_prob is not None:
        try:
            metricas["roc_auc"] = round(roc_auc_score(y_real, y_prob), 4)
        except Exception:
            metricas["roc_auc"] = None

    return metricas


def imprimir_relatorio(nome_modelo: str, metricas: dict, y_real: pd.Series, y_pred: np.ndarray) -> None:
    """Exibe um relatório formatado no console."""
    print(f"\n{'='*60}")
    print(f"  Modelo: {nome_modelo}")
    print(f"{'='*60}")
    for k, v in metricas.items():
        print(f"  {k:<25}: {v}")
    print(
        f"\n{classification_report(y_real, y_pred, target_names=['sem risco', 'em risco'])}")


def plotar_matriz_confusao(
    y_real: pd.Series,
    y_pred: np.ndarray,
    nome_modelo: str,
    salvar: bool = True,
) -> None:
    """Gera e salva a matriz de confusão."""
    cm = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["sem risco", "em risco"],
        yticklabels=["sem risco", "em risco"],
    )
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusão — {nome_modelo}")
    plt.tight_layout()

    if salvar:
        destino = caminho_absoluto(
            "reports/figures") / f"matriz_confusao_{nome_modelo.replace(' ', '_')}.png"
        fig.savefig(destino, dpi=120)
        log.info(f"  → Matriz de confusão salva em: {destino}")
    plt.close(fig)


def plotar_curva_roc(
    y_real: pd.Series,
    y_prob: np.ndarray,
    nome_modelo: str,
    salvar: bool = True,
) -> None:
    """Gera e salva a curva ROC."""
    if y_prob is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_real, y_prob, ax=ax, name=nome_modelo)
    ax.set_title(f"Curva ROC — {nome_modelo}")
    plt.tight_layout()

    if salvar:
        destino = caminho_absoluto(
            "reports/figures") / f"roc_{nome_modelo.replace(' ', '_')}.png"
        fig.savefig(destino, dpi=120)
        log.info(f"  → Curva ROC salva em: {destino}")
    plt.close(fig)


def salvar_relatorio_comparativo(resultados: list[dict]) -> Path:
    """
    Salva um CSV com a comparação de todos os modelos avaliados.

    Args:
        resultados: lista de dicts, cada um com 'modelo' e as métricas

    Returns:
        Caminho do arquivo salvo
    """
    df_relatorio = pd.DataFrame(resultados).sort_values(
        "f1_macro", ascending=False)
    destino = caminho_absoluto("reports") / "comparacao_modelos.csv"
    df_relatorio.to_csv(destino, index=False, encoding="utf-8-sig")
    log.info(f"  → Relatório comparativo salvo em: {destino}")

    # Também salva como JSON para rastreabilidade
    destino_json = caminho_absoluto("reports") / "comparacao_modelos.json"
    with open(destino_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("  COMPARAÇÃO DE MODELOS (ordenado por F1-Macro)")
    print(f"{'='*60}")
    print(df_relatorio.to_string(index=False))

    return destino


def avaliar_modelo(
    nome_modelo: str,
    modelo,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    gerar_graficos: bool = True,
) -> dict:
    """
    Avalia um modelo treinado no conjunto de teste e gera relatórios.

    Returns:
        Dicionário com nome do modelo e todas as métricas.
    """
    log.info(f"Avaliando modelo: {nome_modelo} ...")
    y_pred = modelo.predict(X_teste)

    # Probabilidades (se o modelo suportar)
    y_prob = None
    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X_teste)[:, 1]
    elif hasattr(modelo, "decision_function"):
        y_prob = modelo.decision_function(X_teste)

    metricas = calcular_metricas(y_teste, y_pred, y_prob)
    imprimir_relatorio(nome_modelo, metricas, y_teste, y_pred)

    if gerar_graficos:
        plotar_matriz_confusao(y_teste, y_pred, nome_modelo)
        plotar_curva_roc(y_teste, y_prob, nome_modelo)

    return {"modelo": nome_modelo, **metricas}
