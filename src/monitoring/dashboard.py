"""
Dashboard Streamlit para monitoramento do modelo em produção.

Execução:
    streamlit run src/monitoring/dashboard.py

Exibe:
  - Volume de predições ao longo do tempo
  - Distribuição de risco previsto
  - Relatório de drift das features
  - Métricas de desempenho (se rótulos reais disponíveis)
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Caminho base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Caminhos diretos
PREDICTIONS_LOG = BASE_DIR / "logs" / "predictions.csv"
METRICS_LOG = BASE_DIR / "logs" / "metrics.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_defasagem.pkl"

st.set_page_config(
    page_title="Dashboard — Risco de Defasagem Escolar",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard de Monitoramento — Risco de Defasagem Escolar")
st.caption("Associação Passos Mágicos | Datathon FIAP Pós Tech")
st.divider()

# ── Sem cache para sempre ler dados atualizados ─────────────────────────────


def carregar_predicoes():
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_LOG)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df


def carregar_metricas():
    if not METRICS_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_LOG)


# ── Botão de atualizar ───────────────────────────────────────────────────────
col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Atualizar dados"):
        st.rerun()

# ── Carregar dados ───────────────────────────────────────────────────────────
df_predicoes = carregar_predicoes()
df_metricas = carregar_metricas()

# ── Métricas principais ──────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    modelo_existe = MODEL_PATH.exists()
    st.metric("Modelo em produção",
              "✅ Carregado" if modelo_existe else "❌ Não encontrado")

with col2:
    total_pred = len(df_predicoes) if not df_predicoes.empty else 0
    st.metric("Total de predições", total_pred)

with col3:
    if not df_predicoes.empty and "predicao" in df_predicoes.columns:
        pct_risco = df_predicoes["predicao"].mean() * 100
        st.metric("% com risco de defasagem", f"{pct_risco:.1f}%")
    else:
        st.metric("% com risco de defasagem", "—")

with col4:
    if not df_predicoes.empty and "probabilidade_risco" in df_predicoes.columns:
        media_proba = df_predicoes["probabilidade_risco"].mean() * 100
        st.metric("Probabilidade média de risco", f"{media_proba:.1f}%")
    else:
        st.metric("Probabilidade média de risco", "—")

st.divider()

# ── Gráficos ─────────────────────────────────────────────────────────────────
if not df_predicoes.empty:

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("📊 Distribuição das Predições")
        fig, ax = plt.subplots(figsize=(5, 3))
        contagem = df_predicoes["predicao"].value_counts()
        contagem.index = ["SEM risco" if i ==
                          0 else "COM risco" for i in contagem.index]
        cores = ["#2ecc71" if "SEM" in str(
            i) else "#e74c3c" for i in contagem.index]
        contagem.plot(kind="bar", ax=ax, color=cores)
        ax.set_title("Risco de Defasagem")
        ax.set_xlabel("")
        ax.set_ylabel("Quantidade")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    with col_g2:
        st.subheader("🎯 Distribuição da Probabilidade de Risco")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.hist(df_predicoes["probabilidade_risco"],
                 bins=10, color="#3498db", edgecolor="white")
        ax2.set_xlabel("Probabilidade de Risco")
        ax2.set_ylabel("Frequência")
        ax2.set_title("Histograma de Probabilidades")
        plt.tight_layout()
        st.pyplot(fig2)

    # Gráfico temporal
    if "timestamp" in df_predicoes.columns:
        st.subheader("📈 Predições ao longo do tempo")
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        cores_linha = ["#e74c3c" if p ==
                       1 else "#2ecc71" for p in df_predicoes["predicao"]]
        ax3.scatter(df_predicoes["timestamp"],
                    df_predicoes["probabilidade_risco"], c=cores_linha, zorder=3)
        ax3.plot(df_predicoes["timestamp"],
                 df_predicoes["probabilidade_risco"], color="#3498db", alpha=0.4)
        ax3.axhline(y=0.5, color="orange",
                    linestyle="--", label="Threshold 0.5")
        ax3.set_ylabel("Probabilidade de Risco")
        ax3.set_title("Evolução das Predições")
        ax3.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)

    st.divider()

    # Tabela
    st.subheader("📋 Histórico de Predições")
    st.dataframe(
        df_predicoes[["timestamp", "predicao",
                      "probabilidade_risco", "classificacao"]].tail(20),
        use_container_width=True
    )

else:
    st.info("Nenhuma predição registrada ainda. Use a API para gerar predições e elas aparecerão aqui.")

st.divider()

# ── Métricas do modelo ────────────────────────────────────────────────────────
st.subheader("📉 Métricas do Modelo")
if df_metricas.empty:
    st.info("Nenhuma métrica registrada ainda.")
else:
    row = df_metricas.iloc[0]

    # Cards com métricas principais
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🎯 Accuracy",   f"{float(row['accuracy'])*100:.2f}%")
    with m2:
        st.metric("📊 F1 Macro",   f"{float(row['f1_macro'])*100:.2f}%")
    with m3:
        st.metric("🔍 Precision",  f"{float(row['precision'])*100:.2f}%")
    with m4:
        st.metric("📡 Recall",     f"{float(row['recall'])*100:.2f}%")

    m5, m6, m7, m8 = st.columns(4)
    with m5:
        st.metric("📈 ROC AUC",        f"{float(row['roc_auc'])*100:.2f}%")
    with m6:
        st.metric("⚖️ F1 Weighted",    f"{float(row['f1_weighted'])*100:.2f}%")
    with m7:
        st.metric("👥 Total Amostras", int(row['total_amostras']))
    with m8:
        st.metric("🤖 Modelo",         row['modelo'])

    # Gráfico de barras das métricas
    st.subheader("📊 Comparativo das Métricas")
    fig, ax = plt.subplots(figsize=(8, 3))
    metricas_plot = {
        "Accuracy":    float(row["accuracy"]),
        "F1 Macro":    float(row["f1_macro"]),
        "F1 Weighted": float(row["f1_weighted"]),
        "Precision":   float(row["precision"]),
        "Recall":      float(row["recall"]),
        "ROC AUC":     float(row["roc_auc"]),
    }
    bars = ax.barh(list(metricas_plot.keys()), list(
        metricas_plot.values()), color="#3498db")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.9, color="orange", linestyle="--", label="Threshold 90%")
    for bar, val in zip(bars, metricas_plot.values()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val*100:.2f}%", va="center", fontsize=9)
    ax.legend()
    ax.set_title("Métricas de Performance do Modelo")
    plt.tight_layout()
    st.pyplot(fig)

    # Distribuição das classes
    st.subheader("⚖️ Distribuição das Classes no Treino")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    valores = [int(row["positivos"]), int(row["negativos"])]
    rotulos = [f"COM risco\n({int(row['positivos'])})",
               f"SEM risco\n({int(row['negativos'])})"]
    cores = ["#e74c3c", "#2ecc71"]
    ax2.pie(valores, labels=rotulos, colors=cores,
            autopct="%1.1f%%", startangle=90)
    ax2.set_title("Classes na Base de Treino")
    plt.tight_layout()
    st.pyplot(fig2)

    st.caption(f"⏱️ Métricas geradas em: {row['timestamp']}")

st.divider()
st.caption("Projeto Datathon FIAP Pós Tech — Associação Passos Mágicos")
