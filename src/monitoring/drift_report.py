"""
Módulo de monitoramento de drift do modelo.

Responsabilidades:
  - Carregar os dados de referência (treino) e de produção (predições logadas)
  - Calcular relatório de drift usando Evidently
  - Salvar relatório HTML em reports/
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesSummaryMetric,
        ColumnDriftMetric,
    )
    EVIDENTLY_DISPONIVEL = True
except Exception:  # pragma: no cover
    EVIDENTLY_DISPONIVEL = False

from src.utils import carregar_config, configurar_logger, caminho_absoluto

log = configurar_logger("drift_report")


def carregar_dados_referencia() -> pd.DataFrame:
    """Carrega o dataset de treino como referência para comparação de drift."""
    config = carregar_config()
    caminho = caminho_absoluto(
        config["caminhos"]["dados_processados"]) / "dataset_features.parquet"

    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset de referência não encontrado: {caminho}\n"
            "Execute o treinamento antes."
        )

    df = pd.read_parquet(caminho)
    log.info(f"Dados de referência carregados: {df.shape}")
    return df


def carregar_dados_producao() -> pd.DataFrame:
    """
    Carrega os dados de entrada das predições logadas em logs/predicoes.jsonl.
    Retorna DataFrame com as features de entrada.
    """
    config = carregar_config()
    caminho_log = caminho_absoluto("logs") / "predicoes.jsonl"

    if not caminho_log.exists() or caminho_log.stat().st_size == 0:
        raise FileNotFoundError(
            f"Arquivo de predições não encontrado ou vazio: {caminho_log}\n"
            "Envie requisições à API antes de gerar o relatório de drift."
        )

    registros = []
    with open(caminho_log, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                dado = json.loads(linha)
                registros.append(dado["entrada"])

    df = pd.DataFrame(registros)
    log.info(f"Dados de produção carregados: {df.shape} ({len(df)} predições)")
    return df


def gerar_relatorio_drift(salvar: bool = True) -> str:
    """
    Gera o relatório completo de drift usando Evidently.

    Returns:
        Caminho do arquivo HTML gerado.
    """
    if not EVIDENTLY_DISPONIVEL:  # pragma: no cover
        raise RuntimeError(
            "A biblioteca 'evidently' não está disponível neste ambiente.\n"
            "Instale com: pip install evidently\n"
            "Nota: evidently requer Python ≤ 3.12 (incompatível com Python 3.14+)."
        )
    log.info("=== Gerando relatório de drift ===")

    df_referencia = carregar_dados_referencia()
    df_producao = carregar_dados_producao()

    # Mantém apenas colunas comuns entre referência e produção
    colunas_comuns = list(set(df_referencia.columns)
                          & set(df_producao.columns))
    colunas_comuns = [c for c in colunas_comuns if c != "alvo"]

    if not colunas_comuns:
        raise ValueError(
            "Nenhuma coluna em comum entre dados de referência e produção.")

    log.info(f"  → Colunas analisadas: {colunas_comuns}")

    relatorio = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesSummaryMetric(),
        *[ColumnDriftMetric(column_name=col)
          for col in colunas_comuns[:10]],  # máx 10 colunas
    ])

    relatorio.run(
        reference_data=df_referencia[colunas_comuns],
        current_data=df_producao[colunas_comuns],
    )

    if salvar:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destino = caminho_absoluto("reports") / \
            f"drift_report_{timestamp}.html"
        relatorio.save_html(str(destino))
        log.info(f"  → Relatório de drift salvo em: {destino}")
        return str(destino)

    return relatorio


if __name__ == "__main__":
    caminho = gerar_relatorio_drift(salvar=True)
    print(f"\nRelatório gerado: {caminho}")
