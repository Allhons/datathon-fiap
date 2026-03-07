# Datathon Passos Mágicos — Predição de Risco de Defasagem Escolar

**FIAP Pós Tech | Datathon Machine Learning Engineering**

Modelo preditivo de risco de defasagem escolar para os estudantes da [Associação Passos Mágicos](https://passosmagicos.org.br/). Cobre todo o ciclo de vida de MLOps: ingestão de dados, feature engineering, treinamento, avaliação, API REST, Docker e monitoramento de drift.

---

## Visão Geral do Problema

A Associação Passos Mágicos atua há 32 anos transformando a vida de crianças e jovens de baixa renda por meio da educação. Com base nos dados educacionais de 2022, 2023 e 2024, este projeto constrói um **classificador binário** capaz de prever se um estudante está em risco de defasagem escolar:

- **0** → Sem risco de defasagem
- **1** → Em risco de defasagem (defasagem < 0 em relação à fase ideal)

A métrica principal de avaliação é o **F1-Macro**, que lida adequadamente com o possível desbalanceamento entre as classes.

---

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| ML | scikit-learn, imbalanced-learn |
| Manipulação de dados | pandas, numpy |
| API | FastAPI + uvicorn |
| Serialização | joblib |
| Testes | pytest + pytest-cov |
| Empacotamento | Docker + Docker Compose |
| Tracking de experimentos | MLflow |
| Monitoramento de drift | Evidently + Streamlit |
| Logs | loguru |

---

## Estrutura do Projeto

```
datathon-passos-magicos/
├── data/
│   ├── raw/                    # Dados originais por ano (Parquet)
│   └── processed/              # Dataset unificado e feature engineered
├── models/                     # Modelo serializado (.pkl)
├── notebooks/
│   └── 01_eda.ipynb            # Análise exploratória de dados
├── src/
│   ├── utils.py                # Utilitários compartilhados
│   ├── data/
│   │   ├── ingestion.py        # Carregamento e unificação das abas
│   │   └── preprocessing.py    # Limpeza, tratamento e variável alvo
│   ├── features/
│   │   └── feature_engineering.py  # Features derivadas + preprocessador sklearn
│   ├── models/
│   │   ├── train.py            # Treinamento com CV + MLflow
│   │   ├── evaluate.py         # Métricas, plots e comparação
│   │   └── predict.py          # Inferência/scoring
│   ├── api/
│   │   ├── app.py              # Instância FastAPI
│   │   ├── routes.py           # Endpoint /predict e /health
│   │   └── schemas.py          # Schemas Pydantic (input/output)
│   └── monitoring/
│       ├── drift_report.py     # Relatório Evidently de drift
│       └── dashboard.py        # Dashboard Streamlit
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
├── configs/
│   └── config.yaml             # Configurações centrais do projeto
├── logs/                       # Logs de execução e predições (JSONL)
├── reports/                    # Relatórios de métricas e drift
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.cfg                   # Configuração do pytest/coverage
```

---

## Pré-requisitos

- Python 3.11+
- pip
- Docker e Docker Compose (para deploy containerizado)
- Git

---

## Instalação de Dependências

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd datathon-passos-magicos

# 2. Crie e ative um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Como Executar o Pipeline Completo

### 1. Coloque o arquivo de dados na raiz do projeto

```
BASE DE DADOS PEDE 2024 - DATATHON.xlsx
```

### 2. Execute o pipeline de dados + treinamento

```bash
# Ingestão: lê o Excel e salva Parquets em data/raw/
python -m src.data.ingestion

# Pré-processamento: unifica, limpa e cria variável alvo
python -m src.data.preprocessing

# Treinamento completo (inclui feature engineering + todos os modelos + MLflow)
python -m src.models.train
```

Ou execute tudo de uma vez:

```bash
python -c "
from src.data.ingestion import ingerir_dados
from src.data.preprocessing import preprocessar
from src.features.feature_engineering import executar_feature_engineering
from src.models.train import treinar_todos_modelos

dfs = ingerir_dados(salvar_raw=True)
df_proc = preprocessar(dfs, salvar=True)
X_tr, X_v, X_te, y_tr, y_v, y_te, prep = executar_feature_engineering(df_proc)
treinar_todos_modelos(X_tr, X_v, X_te, y_tr, y_v, y_te, prep)
"
```

---

## Como Rodar a API Localmente

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Acesse a documentação interativa: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Exemplos de Chamadas à API

### Via cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Predição de risco
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inde": 6.5,
    "iaa": 7.0,
    "ieg": 5.5,
    "ips": 6.0,
    "ipp": 6.5,
    "ida": 7.0,
    "ipv": 6.8,
    "ian": 5.0,
    "nota_matematica": 6.0,
    "nota_portugues": 5.5,
    "nota_ingles": 7.0,
    "idade": 14,
    "fase": "Fase 5",
    "genero": "Masculino",
    "pedra": "Quartzo",
    "instituicao_ensino": "Pública",
    "ano_ingresso": 2020
  }'
```

### Via Python

```python
import requests

payload = {
    "inde": 6.5, "iaa": 7.0, "ieg": 5.5,
    "nota_matematica": 6.0, "nota_portugues": 5.5,
    "idade": 14, "fase": "Fase 5", "genero": "Masculino",
}
resposta = requests.post("http://localhost:8000/api/v1/predict", json=payload)
print(resposta.json())
```

### Resposta esperada

```json
{
  "status": "sucesso",
  "dados": {
    "predicao": 1,
    "probabilidade_risco": 0.72,
    "classificacao": "EM RISCO de defasagem"
  },
  "versao_modelo": "1.0.0"
}
```

---

## Como Rodar com Docker

```bash
# Build e start de todos os serviços
docker compose up --build

# Apenas a API
docker compose up api --build

# Em background
docker compose up -d --build
```

| Serviço | Endereço |
|---|---|
| API | http://localhost:8000 |
| Docs (Swagger) | http://localhost:8000/docs |
| Dashboard Streamlit | http://localhost:8501 |

Para parar:

```bash
docker compose down
```

---

## Como Rodar os Testes

```bash
# Todos os testes com relatório de cobertura
pytest

# Apenas um módulo específico
pytest tests/test_api.py -v

# Gerar relatório HTML de cobertura
pytest --cov=src --cov-report=html
# Abrir: htmlcov/index.html
```

---

## Tracking de Experimentos (MLflow)

Os experimentos são automaticamente registrados no MLflow durante o treinamento. Para visualizar:

```bash
mlflow ui
```

Acesse: [http://localhost:5000](http://localhost:5000)

---

## Monitoramento de Drift

> **⚠️ Nota sobre Python 3.14:** A biblioteca `evidently` utiliza Pydantic V1 internamente, que é incompatível com Python 3.14+. O relatório de drift requer **Python ≤ 3.12**. O Streamlit dashboard funciona normalmente em Python 3.14.

### Gerar relatório Evidently

```bash
python -m src.monitoring.drift_report
```

O relatório HTML é salvo em `reports/drift_report_<timestamp>.html`.

### Dashboard Streamlit (local)

```bash
streamlit run src/monitoring/dashboard.py
```

Acesse: [http://localhost:8501](http://localhost:8501)

---

## Etapas do Pipeline de Machine Learning

1. **Ingestão**: leitura das abas PEDE2022, PEDE2023 e PEDE2024 do Excel; padronização de schemas; salvamento em Parquet.
2. **Pré-processamento**: unificação dos 3 anos; remoção de colunas irrelevantes; imputação de valores ausentes (mediana/moda); criação da variável alvo binária (defasagem < 0 → em risco).
3. **Feature Engineering**: criação de features derivadas (média de notas, score de índices, gap de fase, anos no programa); normalização e encoding via ColumnTransformer; divisão estratificada treino/validação/teste.
4. **Treinamento**: treinamento de 4 modelos candidatos (RandomForest, GradientBoosting, LogisticRegression, SVC) com cross-validation estratificado (5 folds); registro de parâmetros e métricas no MLflow.
5. **Avaliação e Seleção**: comparação por F1-Macro no conjunto de teste; geração de matrizes de confusão e curvas ROC; seleção automática do melhor modelo.
6. **Serialização**: modelo final serializado com joblib em `models/modelo_defasagem.pkl`.
7. **Serving**: API REST FastAPI com endpoint `/predict`; logging de todas as predições em `logs/predicoes.jsonl`.
8. **Monitoramento**: análise de drift das features com Evidently; dashboard Streamlit para acompanhamento em tempo real.

---

## Checklist de Requisitos

- [x] Pipeline de treinamento completa (feature engineering + pré-processamento + treinamento + validação)
- [x] Modelo salvo com joblib
- [x] Justificativa da métrica: F1-Macro (lida com desbalanceamento de classes)
- [x] Código modularizado em arquivos `.py` separados
- [x] API FastAPI com endpoint `/predict`
- [x] Testes unitários com pytest (cobertura ≥ 80%)
- [x] Dockerfile para empacotamento
- [x] Docker Compose para orquestração
- [x] Deploy local via Docker
- [x] Logging de predições em JSONL
- [x] Monitoramento de drift (Evidently + Streamlit)
- [x] Tracking de experimentos (MLflow)
- [x] Documentação completa (este README)
- [ ] Repositório GitHub público
- [ ] Link para API em produção
- [ ] Vídeo de até 5 minutos (formato gerencial)

---

## Autores

Projeto desenvolvido como entrega do **Datathon FIAP Pós Tech — Machine Learning Engineering**.
