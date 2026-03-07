"""
Módulo de ingestão de dados.

Responsabilidade: ler o arquivo Excel original (3 abas: 2022, 2023, 2024),
padronizar schemas e salvar cada aba como Parquet em data/raw/.
"""

import pandas as pd
from pathlib import Path

from src.utils import carregar_config, configurar_logger, caminho_absoluto

log = configurar_logger("ingestion")


def _padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas: minúsculas, sem espaços, sem acentos."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("º", "", regex=False)
        .str.replace("ã", "a", regex=False)
        .str.replace("ç", "c", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("ê", "e", regex=False)
        .str.replace("á", "a", regex=False)
        .str.replace("â", "a", regex=False)
        .str.replace("ó", "o", regex=False)
        .str.replace("ô", "o", regex=False)
        .str.replace("í", "i", regex=False)
        .str.replace("ú", "u", regex=False)
        .str.replace(".", "_", regex=False)
    )
    return df


def _renomear_colunas_por_ano(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    ano_2dig = str(ano)[-2:]
    col_inde_4dig = f"inde_{ano}"
    col_inde_2dig = f"inde_{ano_2dig}"
    col_pedra_4dig = f"pedra_{ano}"
    col_pedra_2dig = f"pedra_{ano_2dig}"

    colunas_dropar = []
    if col_inde_4dig in df.columns and col_inde_2dig in df.columns:
        colunas_dropar.append(col_inde_2dig)
    if col_pedra_4dig in df.columns and col_pedra_2dig in df.columns:
        colunas_dropar.append(col_pedra_2dig)
    if colunas_dropar:
        df = df.drop(columns=colunas_dropar)

    mapeamentos = {
        # Coluna alvo
        "defas": "defasagem",
        # Notas de disciplinas
        "matem": "nota_matematica",
        "mat": "nota_matematica",
        "portug": "nota_portugues",
        "por": "nota_portugues",
        "ingles": "nota_ingles",
        "ing": "nota_ingles",
        # Dados pessoais
        "ano_nasc": "ano_nascimento",
        "data_de_nasc": "data_nascimento",
        "idade_22": "idade",
        "nome_anonimizado": "nome",
        # Fase ideal
        "fase_ideal": "fase_ideal",
        # Ativo/Inativo
        "ativo__inativo": "status",
        "ativo__inativo_1": "status_2",
        # 🚨 REMOVIDO: inde e pedra não são mais renomeados
        # col_inde_4dig: "inde",
        # col_inde_2dig: "inde",
        # col_pedra_4dig: "pedra",
        # col_pedra_2dig: "pedra",
    }

    mapeamentos_validos = {k: v for k,
                           v in mapeamentos.items() if k in df.columns}
    return df.rename(columns=mapeamentos_validos)


def carregar_aba(caminho_excel: Path, nome_aba: str, ano: int) -> pd.DataFrame:
    """
    Lê uma aba do Excel, padroniza colunas e adiciona coluna 'ano'.

    Returns:
        DataFrame com schema padronizado.
    """
    log.info(f"Lendo aba '{nome_aba}' do arquivo {caminho_excel.name} ...")
    df = pd.read_excel(caminho_excel, sheet_name=nome_aba)
    log.info(f"  → {df.shape[0]} linhas, {df.shape[1]} colunas carregadas.")

    df = _padronizar_colunas(df)
    df = _renomear_colunas_por_ano(df, ano)
    df["ano"] = ano

    # Converte colunas de data/misto para string (evita erros do pyarrow ao salvar parquet)
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("datetime"):
            df[col] = df[col].astype(str)

    return df


def ingerir_dados(salvar_raw: bool = True) -> dict[str, pd.DataFrame]:
    """
    Ponto de entrada principal: lê as 3 abas do Excel e salva como Parquet.

    Args:
        salvar_raw: se True, salva cada aba em data/raw/{ano}.parquet

    Returns:
        Dicionário {ano: DataFrame} com os dados de cada ano.
    """
    config = carregar_config()
    caminho_excel = caminho_absoluto(config["caminhos"]["arquivo_origem"])

    if not caminho_excel.exists():
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {caminho_excel}\n"
            "Certifique-se de que o arquivo Excel está na raiz do projeto."
        )

    dfs = {}
    for aba_cfg in config["dados"]["abas"]:
        nome_aba = aba_cfg["nome"]
        ano = aba_cfg["ano"]

        df = carregar_aba(caminho_excel, nome_aba, ano)
        dfs[ano] = df

        if salvar_raw:
            destino = caminho_absoluto(
                config["caminhos"]["dados_brutos"]) / f"pede_{ano}.parquet"
            df.to_parquet(destino, index=False)
            log.info(f"  → Salvo em: {destino}")

    log.info("Ingestão concluída.")
    return dfs


if __name__ == "__main__":
    dados = ingerir_dados(salvar_raw=True)
    for ano, df in dados.items():
        print(f"\nAno {ano}: {df.shape} | Colunas: {df.columns.tolist()}")
