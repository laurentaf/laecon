# CONSTITUTION — LAECON

**Status:** COMPLETE (BASIC) — G4 delivered 2026-06-13
**Version:** 0.1.0-BASIC
**Last updated:** 2026-06-13
**Tracking:** `../LAOS/projects/_meta/capability-evolution/laecon.md`

Capability LAOS para econometria + machine learning interpretável.
**Posicionamento:** econometria é a espinha, ML é o músculo.
**Likelihood explícita, não black box.**

---

## Artigo 1 — Propósito

LAECON é a capability do LAOS para **modelagem estatística e machine
learning interpretável**, com likelihood explícita. Posiciona-se como
**ponte entre LATADE** (data engineering, SQL/DuckDB/medallion) e
**LADESIGN/LAN8N** (consumo dos outputs em visual/automação).

A capability existe para:

1. Capacitar o usuário (economista Unicamp) a fazer modelagem preditiva
   e inferência causal diretamente em LAOS, sem precisar de ferramentas
   externas (R, SAS, Stata, Python ad-hoc).
2. Operacionalizar a tradição econométrica de **likelihood explícita**
   (vs. ML black-box) alinhada com Gujarati & Porter e o artigo da Quirk's.
3. Gerar artefatos estruturados (relatórios, predições, plots) que
   outras capabilities (LADESIGN, LAN8N) consumam de forma confiável.

---

## Artigo 2 — Escopo

### Dentro do escopo
- Regressão linear: OLS, GLS, WLS, robust SE (Huber-White, HAC).
- GLM: logit, probit, poisson, negative binomial, gamma, inverse gaussian.
- Ordered models: ordered logit/probit (essencial para NPS).
- Grouped models: grouped logit (essencial para NPS agregado por região).
- Validação de pressupostos Gujarati-basics.
- Interpretação (efeitos marginais AME/MER, odds ratios, elasticidades).
- Avaliação (R², adj-R², AIC, BIC, log-likelihood, RMSE, MAE, log-loss, AUC).
- Predição (in-sample, out-of-sample com IC).
- Simulação de drivers (NPS driver simulator do artigo da Quirk's).
- Cross-validation (k-fold, stratified, leave-one-out) — **M1**.
- ML interpretável (decision tree, RF, gradient boosting) COM SHAP — **M2**.
- Convenção `_commomdata` como fonte de dados cross-project.

### Fora do escopo
- Deep learning (PyTorch, TensorFlow) — pós-STABLE se houver demanda.
- NLP/LLM, computer vision — fora do escopo LAOS.
- Big data distribuído (Spark, Dask) — DuckDB é suficiente.
- Causal inference avançada (DiD, RDD, IV, PSM) — **M3**, pós-STABLE.

---

## Artigo 3 — Princípios

1. **Likelihood explícita, não black box.** Todo modelo usado tem likelihood
   conhecida e documentada. Tree-based models (RF, GBT) só entram com
   SHAP explícito (condição vinculante DA-2).
2. **Interpretabilidade antes de acurácia.** Coeficientes com SE,
   p-values, IC, efeitos marginais antes de métricas de scoring.
3. **Econometria é a espinha, ML é o músculo.** Modelagem estatística
   vem primeiro; ML preditivo complementa.
4. **Local-first, DuckDB como engine.** Reusa runtime de LATADE.
   Suficiente para o escopo. Sem Spark/Dask.
5. **Anti-overfitting explícito.** Cross-validation, regularização
   documentadas. Toda métrica de scoring vem com baseline.
6. **DataFrame empty guards obrigatórios** (P0 `padroes-entrega.md`,
   condição DR-1). Toda função que opera sobre dados verifica
   emptiness e retorna mensagem amigável, não `IndexError` ou
   `ValueError`.

---

## Artigo 4 — Input/Output Contracts (com LATADE, LADESIGN, LAN8N)

> Condições vinculantes endereçadas: DA-3, AE-4, DD-1, AE-3.

### §1. Input — fontes de dados

#### 1a. Gold tables de LATADE (DuckDB) — protocolo de leitura

LAECON lê exclusivamente via `latade://gold/<table>`, que se materializa
como `SELECT * FROM <table>` no DuckDB de LATADE. O path do DuckDB
é resolvido por `LATADE_DB_PATH` (variável de ambiente, nunca hardcoded).

**Protocolo:**
1. `latade_execute_sql(query="SELECT * FROM gold.<table>")` via MCP.
2. Retorno é um DataFrame pandas. LAECON valida: (a) não vazio
   (guard Art. 3.6), (b) colunas esperadas presentes, (c) tipos
   coerentes (datas como datetime, IDs como string/int).
3. Se validação falhar, LAECON retorna `status: error` com mensagem
   descritiva — não silencia nem preenche com synthetic data.

**Tabelas comumente consumidas:**
- `gold.nps_responses` — respostas NPS com dimensões (região, segmento, data)
- `gold.customer_master` — master de clientes com atributos demográficos
- `gold transactions` — transações agregadas (receita, frequência, ticket)

#### 1b. Arquivos em `_commomdata/`

Fontes de grounding reutilizáveis (livros, papers, datasets públicos).
Path: `E:/projects/_commomdata/<fonte>/`. Documentado em
`knowledge/data-conventions.md`. LAECON lê CSV/Parquet/JSON com
`pd.read_csv()` / `pd.read_parquet()` / `pd.read_json()`.

#### 1c. CSV/Parquet/JSON locais (uploads do usuário)

Path livre. LAECON aceita qualquer path absoluto. Validação mínima:
schema preview (`latade_generate_schema_preview`) antes de treinar.

### §2. Output — artefatos produzidos

#### 2a. Modelo treinado versionado (`model_id`)

Cada chamada a `train_*` persiste:
- `models/<model_id>/model.joblib` — serialização do modelo.
- `models/<model_id>/metadata.yaml` — hiperparâmetros, features,
  métricas, timestamp, hash do dataset.
- `models/<model_id>/diagnostic_report.md` — relatório completo
  (Art. 7).

Detalhes do schema: Constitution Art. 6.

#### 2b. Diagnostic report — path determinístico (AE-3)

Sempre em `artifacts/reports/<model_id>/diagnostic_report.md`.
Path calculado: `<project_root>/artifacts/reports/<model_id>/`.
LAN8N e LADESIGN leem deste path para workflows downstream.

#### 2c. Predições — schema versionado

Tabela de predições em `artifacts/predictions/<model_id>.parquet`:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `row_id` | string | ID original da linha no dataset de input |
| `prediction` | float | Valor predito (point estimate) |
| `se` | float | Erro padrão da predição (quando disponível) |
| `ci_lower` | float | Limite inferior IC 95% |
| `ci_upper` | float | Limite superior IC 95% |
| `model_id` | string | Identificador do modelo (Art. 6) |
| `predicted_at` | datetime | Timestamp da predição |

**Versionamento do schema:** versão semântica no `metadata.yaml`
do modelo (`prediction_schema: "1.0.0"`). Breaking changes incrementam
major; additions incrementam minor; fixes incrementam patch.

#### 2d. Artefatos estruturados para LADESIGN (DD-1)

JSON/YAML em `artifacts/structured/<model_id>/`:
- `probability_curves.json` — curvas de probabilidade por outcome level
  (essencial para NPS: P(Promotor), P(Neutro), P(Detrator) por idade)
- `marginal_effects.json` — efeitos marginais AME/MER por variável
- `shap_summary.json` — SHAP values agregados (M2+)
- `calibration.json` — dados do calibration plot (M1+)

LADOS lê estes arquivos para gerar dashboards. Schema declarado no
`metadata.yaml` do modelo, versionado separadamente.

### §3. Feature engineering — contrato de transformação

Feature engineering **vive em LAECON**, não em LATADE.

**Responsabilidade de LATADE:** dados limpos, tipados, sem duplicatas,
com medallion pipeline (bronze→silver→gold). LATADE NÃO cria features
computacionais.

**Responsabilidade de LAECON:** transformação de dados brutos em features
de modelo:

| Transformação | Exemplo | Onde |
|---------------|---------|------|
| Dummies | `regiao_NE`, `regiao_N`, `regiao_CO` | `train_classifier` (interno) |
| Interações | `receita * idade` | `train_regression` (interno) |
| Log transform | `log(receita + 1)` | Prévia do usuário ou automático |
| Normalização | z-score ou min-max | `train_*` (opcional) |
| Encoding | One-hot, label, ordinal | `train_*` (interno) |
| Binning | Faixas de idade | Manual, antes de `train_*` |

**Regra:** LAECON documenta todas as transformações no `metadata.yaml`
do modelo (features de entrada, fórmulas aplicadas, encodings usados).
Isto garante reprodutibilidade e auditabilidade (Art. 10 §2, pergunta 4).

### §4. Credenciais (DR-4)

Todas as credenciais ficam em `.env` no root do projeto LAECON
(nunca em código, commit, ou notebook):

| Variável | Uso | Default |
|----------|-----|---------|
| `LATADE_DB_PATH` | Path do DuckDB de LATADE | `../latade/data/laos.duckdb` |
| `LATADE_DB_PATH` (override) | DuckDB específico do projeto | — |
| `GITHUB_TOKEN` | Push de artefatos (via MCP) | OS env |

`.env` está em `.gitignore` desde o scaffold G1 (condição DR-4).
Validação: `scripts/preflight_check.py` escaneia por secrets em
arquivos versionados (P0 `padroes-entrega.md`).

---

## Artigo 5 — Antipadrões

- ❌ XGBoost/LightGBM sem SHAP (viola Art. 3.1 e DA-2).
- ❌ Coefs sem SE/p-values (viola Art. 3.2).
- ❌ Métrica de acurácia sem baseline (viola Art. 3.5).
- ❌ Modelo treinado em dados sem validação de pressupostos (viola Art. 2.1).
- ❌ Predição em dataset vazio sem guard (viola Art. 3.6 e `padroes-entrega.md` P0).
- ❌ Modelos sem `model_id` versionado (viola Art. 6 e AE-1).
- ❌ Relatório sem plots de diagnóstico (viola DD-2).
- ❌ Handoff implícito com LATADE (viola Art. 4 e DA-3).
- ❌ Secrets hardcoded em código (viola DR-4).

---

## Artigo 6 — Model Registry & Versionamento

> Condição vinculante endereçada: AE-1.

### §1. Formato do `model_id`

```
<dataset_slug>_<algorithm>_<YYYYMMDD>_<hash[:6]>
```

| Componente | Descrição | Exemplo |
|------------|-----------|---------|
| `dataset_slug` | Slug do dataset de input (max 20 chars, lowercase, underscores) | `nps_2025q4` |
| `algorithm` | Algoritmo + variação principal | `ordered_logit`, `ols_hc1`, `rf_shap` |
| `YYYYMMDD` | Data de treinamento | `20260604` |
| `hash[:6]` | Primeiros 6 chars do SHA256 do código + dados + seed | `a3f9c1` |

**Exemplos completos:**
- `nps_2025q4_ordered_logit_20260604_a3f9c1` — ordered logit para NPS Q4
- `receita_ols_hc1_20260604_b7d2e3` — OLS com robust SE (HC1)
- `nps_2025q4_rf_shap_20260701_f4a8c2` — random forest com SHAP (M2)
- `satisfacao_logit_20260604_c1e9d5` — logit binário de satisfação
- `nps_regiao_grouped_logit_20260610_e3b7a1` — grouped logit por região

**Restrições:**
- `dataset_slug`: regex `^[a-z0-9_]{1,20}$`
- `algorithm`: regex `^[a-z0-9_]+$` (lista permitida na §3)
- `hash`: SHA256 de `codigo_fonte + dados_hash + seed`, truncado a 6 chars

### §2. Schema do `models/registry.json`

Todo modelo treinado gera entrada em `models/registry.json` no root
do projeto. Schema JSON (validado contra `models/registry.schema.json`):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "required": ["model_id", "algorithm", "dataset", "trained_at",
                 "metrics", "features", "artifact_path", "schema_version"],
    "properties": {
      "model_id": { "type": "string", "pattern": "^[a-z0-9_]+_\\d{8}_[a-f0-9]{6}$" },
      "algorithm": { "type": "string", "enum": [
        "ols", "ols_hc1", "ols_hac", "gls", "wls",
        "logit", "probit", "poisson", "negbin", "gamma", "invgauß",
        "ordered_logit", "ordered_probit",
        "grouped_logit",
        "decision_tree", "random_forest", "xgboost", "lightgbm",
        "arima", "var", "ecm",
        "psm", "iv_2sls", "did"
      ]},
      "dataset": {
        "type": "object",
        "properties": {
          "source": { "type": "string" },
          "hash": { "type": "string" },
          "n_rows": { "type": "integer" },
          "n_cols": { "type": "integer" },
          "date_range": { "type": "string" }
        }
      },
      "trained_at": { "type": "string", "format": "date-time" },
      "seed": { "type": "integer" },
      "features": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "dtype": { "type": "string" },
            "role": { "type": "string", "enum": ["feature", "target", "id"] },
            "transform": { "type": "string" }
          }
        }
      },
      "hyperparameters": { "type": "object" },
      "metrics": {
        "type": "object",
        "properties": {
          "train": { "type": "object" },
          "test": { "type": "object" },
          "cv": { "type": "object" }
        }
      },
      "diagnostics": {
        "type": "object",
        "properties": {
          "assumptions": { "type": "object" },
          "vif_max": { "type": "number" },
          "bp_p_value": { "type": "number" },
          "dw_statistic": { "type": "number" }
        }
      },
      "artifact_path": { "type": "string" },
      "report_path": { "type": "string" },
      "schema_version": { "type": "string" },
      "parent_model_id": { "type": ["string", "null"] },
      "retrained_from": { "type": ["string", "null"] }
    }
  }
}
```

### §3. Persistência — `models/<model_id>/`

Cada modelo treinado persiste em diretório próprio:

```
models/<model_id>/
├── model.joblib           # Serialização do modelo (joblib.dump)
├── metadata.yaml          # Hiperparâmetros, features, métricas, diagnostics
├── diagnostic_report.md   # Relatório completo (Art. 7)
├── artifacts/
│   ├── shap_summary.json  # SHAP values (M2+, quando aplicável)
│   ├── partial_dep.json   # Partial dependence (M1+)
│   ├── calibration.json   # Calibration data (M1+, classificação)
│   └── prob_curves.json   # Probability curves (M1+, NPS)
└── data/
    ├── train_hashes.csv   # Hashes dos dados de treino (reprodutibilidade)
    └── feature_names.txt  # Lista de features na ordem exata
```

**`metadata.yaml` schema mínimo:**

```yaml
model_id: nps_2025q4_ordered_logit_20260604_a3f9c1
algorithm: ordered_logit
dataset:
  source: latade://gold/nps_responses
  hash: "sha256:abc123..."
  n_rows: 1247
  n_cols: 18
trained_at: "2026-06-04T14:32:00-03:00"
seed: 42
features:
  - name: nps_score
    dtype: int64
    role: target
    transform: none
  - name: log_receita
    dtype: float64
    role: feature
    transform: "log(receita + 1)"
hyperparameters:
  method: ordered_logit
  optimizer: BFGS
  max_iter: 500
metrics:
  train:
    pseudo_r2: 0.34
    llf: -1847.2
    aic: 3722.4
    bic: 3789.1
  test:
    log_loss: 0.89
    accuracy: 0.52
    auc: 0.71
diagnostics:
  assumptions:
    parallel_lines: true
    proportional_odds_p: 0.12
  vif_max: 4.8
  bp_p_value: 0.23
  dw_statistic: 1.98
artifact_path: models/nps_2025q4_ordered_logit_20260604_a3f9c1/
report_path: artifacts/reports/nps_2025q4_ordered_logit_20260604_a3f9c1/diagnostic_report.md
schema_version: "1.0.0"
parent_model_id: null
retrained_from: null
```

### §4. Protocolo de retreinamento

**Trigger de retreinamento (Art. 10 §2, pergunta 7):**
- Drift de dados: KS-test p < 0.05 entre treino e novo batch
- Decay de performance: AUC rolling < threshold por 5+ batches
- Mudança de regime: CUSUM detecta structural break
- Tempo calendário: > 90 dias sem retreinamento

**Preservação de histórico:**
- Novo retreinamento gera **novo `model_id`** (não sobrescreve o antigo).
- `registry.json` acumula entradas — histórico é imutável.
- Campo `parent_model_id` no novo modelo aponta para o antigo.
- Campo `retrained_from` registra o motivo (drift, decay, regime, calendário).
- `model_id` anterior fica em `models/<antigo>/` — nunca deletado.
- O `predict` tool aceita qualquer `model_id` registrado (não só o mais novo).

**Seleção automática (M1+):** `compare_models` aceita lista de `model_id`s
e seleciona por AIC/BIC/likelihood-ratio. Não deleta os não selecionados.

---

## Artigo 7 — Reporting Standards

> Condições vinculantes endereçadas: DD-2, DD-3.

### §1. Regra geral

Todo modelo treinado tem `export_diagnostic_report` gerado automaticamente.
O relatório é o artefato de entrega primário de LAECON — consumido por
usuário, LADESIGN, e LAN8N.

### §2. Formato

- **Formato padrão:** Markdown (sempre) — path determinístico
  `artifacts/reports/<model_id>/diagnostic_report.md` (AE-3).
- **Formato opcional:** HTML — gerado por `export_diagnostic_report`
  com flag `format=html`. Path: mesmo diretório, extensão `.html`.
- **Encoding:** UTF-8 sem BOM.
- **Linha:** LF (Unix), sem trailing whitespace.

### §3. Estrutura do relatório

Todo relatório DEVE conter estas seções (nesta ordem):

```markdown
# Diagnóstico — <model_id>

## 1. Sumário
- Objetivo, dataset, algoritmo escolhido, motivo da escolha.

## 2. Dados de Input
- Tabela: n_rows, n_cols, fonte, período, colunas.

## 3. Especificação do Modelo
- Equação / fórmula. Variáveis dependentes e independentes.
- Hiperparâmetros e justificativa (Art. 10 §2, pergunta 3).

## 4. Coeficientes e Inferência
- Tabela: coef, SE, z/t, p-value, IC 95%.
- Para GLM: odds ratios, efeitos marginais (AME/MER).
- Para ordered: teste de proportional odds.

## 5. Decisões Metodológicas (Art. 10 §4, DA-5)
- Tabela de algoritmos candidatos (2+ alternativas rejeitadas).
- Tabela de hiperparâmetros (range testado, valor final, critério).
- Tabela de variáveis (justificativa teórica + teste estatístico).
- Indicadores de qualidade com valores de referência.

## 6. Diagnósticos
- Tabela: R², adj-R², AIC, BIC, log-likelihood, RMSE, MAE.
- Pressupostos: heterocedasticidade, autocorrelação, multicolinearidade, normalidade.
- Classificação: AUC, precision, recall, F1, confusion matrix.

## 7. Plots de Diagnóstico
- Todos os plots obrigatórios (§4 abaixo).

## 8. Interpretação
- Linguagem natural: o que os coeficientes significam no contexto.

## 9. Revisão Metodológica (Art. 10 §8)
- 5 dimensões obrigatórias (quando executada).

## 10. Bibliografia
- Referências citadas no relatório.
```

### §4. Plots obrigatórios

Todo relatório DEVE incluir todos os plots aplicáveis ao tipo de modelo.
Formato: matplotlib `savefig` com `dpi=150`, `bbox_inches='tight'`,
`facecolor='white'`, `edgecolor='none'`. Arquivos salvos em
`artifacts/reports/<model_id>/plots/`.

#### Para todo modelo de regressão (OLS, GLS, WLS):

| # | Plot | Arquivo | O que mostra | Referência |
|---|------|---------|--------------|------------|
| 1 | **Residuals vs Fitted** | `residuals_vs_fitted.png` | Linearidade, heterocedasticidade, outliers | Gujarati cap. 10 |
| 2 | **Normal Q-Q** | `qq_plot.png` | Normalidade dos resíduos (Shapiro-Wilk visual) | Gujarati cap. 12 |
| 3 | **Scale-Location** | `scale_location.png` | Heterocedasticidade (variação constante dos resíduos padronizados) | Gujarati cap. 11 |
| 4 | **Cook's Distance / Leverage** | `leverage_plot.png` | Observações influentes (D > 4/n é alerta) | Gujarati cap. 13 |

#### Para modelos de classificação (logit, probit, ordered, grouped):

| # | Plot | Arquivo | O que mostra | Referência |
|---|------|---------|--------------|------------|
| 5 | **ROC Curve** | `roc_curve.png` | Discriminação (AUC); linha diagonal = random | Hosmer & Lemeshow 2000 |
| 6 | **Calibration Plot** | `calibration_plot.png` | Calibração: predito vs observado; ideal = diagonal | Hosmer & Lemeshow 2000 |
| 7 | **Probability Curves** | `probability_curves.png` | P(Y=k|X) por nível do outcome (NPS: Promotor/Neutro/Detrator) | Long 1997 cap. 5 |

#### Para modelos tree-based (M2+, com SHAP):

| # | Plot | Arquivo | O que mostra | Referência |
|---|------|---------|--------------|------------|
| 8 | **SHAP Summary** | `shap_summary.png` | Feature importance com impacto e direção | Lundberg & Lee 2017 |
| 9 | **Partial Dependence** | `partial_dependence.png` | Efeito marginal de 1 feature (holding others constant) | Hastie et al. 2009 cap. 10 |

#### Para ordered logit (NPS):

| # | Plot | Arquivo | O que mostra | Referência |
|---|------|---------|--------------|------------|
| 10 | **Ordered Probability Surface** | `ordered_surface.png` | Probabilidade de cada nível por variável contínua | Long 1997 cap. 5 |
| 11 | **Marginal Effects Plot** | `marginal_effects.png` | AME/MER com IC para cada variável | Long 1997 cap. 8 |

### §5. Parâmetros técnicos de savefig

```python
# Padrão LAECON para todos os plots
import matplotlib.pyplot as plt

fig.savefig(
    filepath,
    dpi=150,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none',
    pad_inches=0.1,
    format='png',
    transparent=False
)
plt.close(fig)  # Sempre fechar após save para liberar memória
```

**Para plots multi-panel (residuals diagnostics 2x2):**
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... preencher axes ...
fig.suptitle(f"Diagnósticos — {model_id}", fontsize=14, y=1.02)
fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
```

### §6. Linguagem

| Contexto | Idioma | Onde se aplica |
|----------|--------|----------------|
| Outputs LAOS-internos | Português | Relatórios, legends dos plots, interpretação |
| Capability externa (LADESIGN para cliente) | Inglês | Dashboard exports, deck exports |
| Metadados técnicos (metadata.yaml, registry.json) | Inglês | Nomes de campos, valores padronizados |
| Comentários de código | Inglês | Sempre |

**Regra de switch:** Se o relatório é consumido internamente (default),
está em português. Se `export_diagnostic_report` é chamado com
`target=external`, gera versão em inglês. O `metadata.yaml` do modelo
sempre usa inglês (interoperabilidade).

### §7. Mínimo de plots por tipo de modelo

| Tipo | Mínimo | Completo |
|------|--------|----------|
| OLS/GLS/WLS | 4 (residuals, QQ, scale-location, leverage) | + partial dependence (M1) |
| Logit/probit/binário | 3 (ROC, calibration, prob curves) | + SHAP (M2) |
| Ordered logit/probit | 3 (ordered surface, marginal effects, prob curves) | + SHAP (M2) |
| Grouped logit | 3 (ROC, calibration, grouped prob curves) | + SHAP (M2) |
| Tree-based (M2) | 2 (SHAP summary, partial dependence) | + all classification plots |
| Time series (M3) | 3 (ACF, PACF, forecast vs actual) | + residual diagnostics |
| Causal (M3) | 2 (balance check, treatment effect) | + sensitivity analysis |

---

## Artigo 8 — Tool Inventory (atual)

Ver `laecon/mcp/server.py` e `list_supported_operations()`. Atualizado
a cada release.

**BASIC (2026-06-04):**
- `health` ✅, `list_supported_operations` ✅
- `train_regression`, `train_classifier`, `validate_assumptions`,
  `interpret_model`, `evaluate_model`, `predict`,
  `export_diagnostic_report` ⚠️ stubs (not_implemented_yet)

**M1 (+30d, 2026-07-04):** Todos funcionais + `compare_models`.

**M2 (+60d, 2026-08-04):** `cross_validate`, `train_tree_model` (COM
`compute_shap` — DA-2), `train_panel_model` (DA-1).

**M3 (+90d, 2026-09-04):** `train_time_series`, `train_causal_model`,
`simulate_drivers`.

---

## Artigo 9 — Evolution Path

### §1. Roadmap com milestones e critérios de aceitação

| Milestone | Descrição | Deadline | Deliverables | Critérios de aceitação | Status |
|-----------|-----------|----------|--------------|----------------------|--------|
| **M0** | Scaffold BASIC (G1, G2, G3, G8) | 2026-06-04 | MCP server stub (9 tools), registry entry, routing, ADR-002 | `health` e `list_supported_operations` funcionais; capabilities.yaml + needs-to-capabilities.yaml atualizados; ADR-002 publicado | ✅ entregue |
| **G4** | Constitution completa | 2026-06-14 | Constitution.md 10 artigos com conteúdo completo | Todos os artigos sem markers "Conteúdo completo em G4"; 17 condições vinculantes endereçadas | ✅ entregue |
| **G5** | SDD workflow + templates | 2026-06-24 | `spec/constitution.md`, `spec/todo.md`, `spec/adr/`, `spec/specs/` templates, `contract.md`, `README.md` | SDD scaffold existe no child repo; `spec/todo.md` populado; `contract.md` ≥ 250 chars | ⏳ |
| **G6** | KB domain mínimo | 2026-07-04 | `kb/README.md` (index) + 1 pattern NPS driver analysis | KB entry com 7 perguntas respondidas (Art. 10 §2) + modelo ordered logit real + dados | ⏳ |
| **M1 (STABLE)** | Constitution completa + SDD + KB + sign-off | 2026-07-04 | Todos os tools MCP funcionais (9+), `compare_models`, diagnostic report completo | G4 ✅ + G5 ✅ + G6 ✅ + G7 ✅ (delivery-reviewer valida 17 condições) | ⏳ |
| **M2** | Tree-based COM SHAP + cross-validation + painel | 2026-08-04 | `train_tree_model`, `compute_shap`, `cross_validate`, `train_panel_model` | DA-2: tree + SHAP no mesmo release; cross-validation funcional; painel FE/RE + Hausman; model registry operacional | ⏳ |
| **M3** | Time series + causal inference | 2026-09-04 | `train_time_series`, `train_causal_model`, `simulate_drivers` | ARIMA/VAR/ECM funcional; PSM/IV/DiD funcional; simulador de drivers (NPS) | ⏳ |
| **M4+** | Deep learning, NLP, GARCH | sob demanda | Tools sob demanda | Proposta + aprovação Conselho | ⏳ |

### §2. Condições vinculantes por milestone (Conselho, 17 itens)

As 17 condições estão catalogadas em
`projects/_meta/capability-evolution/laecon.md` §"Condições vinculantes".
Abaixo, mapeamento de entrega:

| Condição | Milestone de entrega | Verificado em |
|----------|---------------------|---------------|
| DA-1 (painel em M2, time series em M3, causal em M3) | M2, M3 | G7 |
| DA-2 (tree + SHAP no mesmo release) | M2 | G7 |
| DA-3 (handoff latade↔laecon na Constitution) | G4 (Art. 4) | G7 |
| DA-4 (detalhamento metodológico extremo) | BASIC (Art. 10) | G7 |
| DA-5 (seção "Decisões Metodológicas" no relatório) | M1 (implementação) | G7 |
| DD-1 (artefatos JSON/YAML para LADESIGN) | G4 (Art. 4 §2d) | G7 |
| DD-2 (plots completos) | M1 | G7 |
| DD-3 (`export_dashboard_payload`) | M2 ou M3 | G7 |
| AE-1 (model registry/versionamento) | G4 (Art. 6) | G7 |
| AE-2 (workflow N8N de referência) | G2 ou G3 | G7 |
| AE-3 (path determinístico do relatório) | G4 (Art. 4 §2b) | G7 |
| AE-4 (credenciais e contratos I/O na Constitution) | G4 (Art. 4 §4) | G7 |
| DR-1 (DataFrame empty guards) | G1 (MCP server) | G7 |
| DR-2 (spec de modelo em `artifacts/data/`) | G6 + M1 | G7 |
| DR-3 (DESIGN.md em `artifacts/design/source.md`) | M1 | G7 |
| DR-4 (`.env` em `.gitignore`) | G1 (scaffold) | G7 |
| DR-5 (ADR-002 formato ADR-001) | G8 (entregue) | G7 |

### §3. Dependências entre milestones

```
G4 (Constitution) ──→ G5 (SDD) ──→ G6 (KB) ──→ M1 (STABLE)
                      │                            │
                      └────────────────────────────┘
                              (paralelos)

M1 (STABLE) ──→ M2 (tree+SHAP, cross-val, painel) ──→ M3 (time series, causal)
```

- G4 não depende de ninguém (já entregue).
- G5, G6 podem rodar em paralelo (não dependem um do outro).
- M1 requer G4 + G5 + G6 + G7 todos aprovados.
- M2 requer M1 (tools funcionais antes de extensões).
- M3 requer M2 (tree é pré-requisito para causal com ML).

### §4. Promoção a STABLE

A promoção de BASIC → STABLE requer **todos** os gates G1-G7 aprovados
pelo `delivery-reviewer`, verificando as 17 condições vinculantes do
Conselho. A promoção é registrada em:
- `registry/capabilities.yaml` (status: `basic` → `stable`)
- `projects/_meta/capability-evolution/laecon.md` (tracking)
- DuckDB do LACOUNCIL (histórico de evolução)

---

## Artigo 10 — Detalhamento Metodológico Extremo (Metodologia Auditável)

> **Origem:** Princípio fundador do usuário (economista Unicamp, com
> background em estatística, matrizes e econometria). Toda decisão
> metodológica em ML/econometria deve ser **auditável, ensinável e
> reproduzível**. Não basta rodar o modelo — o usuário precisa
> **entender afundo** cada passo, cada indicador, e a referência
> bibliográfica por trás de cada threshold.
>
> **Vinculado a:** Art. 3 (Princípios — interpretabilidade),
> Art. 7 (Reporting Standards), condições vinculantes **DA-4** e **DA-5**.

### §1. Princípio central

> Toda escolha metodológica é uma **hipótese refutável**. Trate cada
> decisão (algoritmo, hiperparâmetro, conjunto de variáveis, transformação,
> regularização) como se fosse objeto de peer review: justifique, cite,
> mostre a alternativa rejeitada, e indique o que faria você mudar de ideia.

### §2. As 7 perguntas obrigatórias por modelo

Para CADA modelo treinado (`train_regression`, `train_classifier`,
`train_tree_model`, `train_panel_model`, `train_time_series`, etc.),
o output deve responder — sem exceção — estas 7 perguntas:

| # | Pergunta | Resposta mínima exigida |
|---|----------|-------------------------|
| 1 | **Qual é a pergunta de negócio / hipótese?** | Frase declarativa. Ex: "Drivers de NPS=Promotor entre clientes B2B". |
| 2 | **Por que ESTE algoritmo?** | Tabela comparativa com ≥2 alternativas rejeitadas. Critério de escolha explícito (interpretabilidade, capacidade de capturar não-linearidade, dimensionalidade, parcimônia, custo computacional). Citar autor. |
| 3 | **Por que ESTES hiperparâmetros?** | Range testado (grid/bayesiano), métrica usada para escolher, valor final + razão. Ex: "max_depth ∈ {3,5,7,10}, escolhido 5 por menor RMSE em 5-fold CV (Breiman 2001, sec. 4.3)". |
| 4 | **Por que ESTAS variáveis?** | Justificativa teórica (literatura) + estatística (teste F incremental, p-value univariado, VIF, correlação). NUNCA "incluí porque parece relevante". Para 5 vs 4 vs 6: mostrar o trade-off parcimônia vs ajuste via AIC/BIC/adj-R² e teste F parcial. |
| 5 | **Quão BOM é o resultado esperado?** | Valor de referência do AUTOR CITADO. Ex: "R² > 0.7 é considerado alto em cross-section social (Gujarati & Porter 2009, cap. 8)". |
| 6 | **Quão RUIM é o resultado aceitável?** | Threshold de abortar/rever. Ex: "Se VIF > 10, remover variável (Hair et al. 2010)". Se logit: "Se Hosmer-Lemeshow p < 0.05, modelo não calibra". |
| 7 | **Quando retreinar?** | Trigger explícito: drift de dados (KS-test), decay de performance (rolling AUC < threshold), mudança de regime (CUSUM), tempo calendário. |

### §3. Referências bibliográficas obrigatórias (com valores numéricos)

Toda decisão de modelo deve **ancorar em pelo menos uma fonte citada**
com **valores numéricos de referência** (não "valores altos/baixos"
vagos). Catálogo inicial:

| Tópico | Fonte canônica | Valores de referência |
|--------|----------------|------------------------|
| R², F-test, heterocedasticidade, autocorrelação, multicolinearidade | Gujarati & Porter, *Basic Econometrics 5th ed* (2009) | R² > 0.7 alto, VIF > 10 problema, Durbin-Watson ≈ 2 ok, Breusch-Pagan p < 0.05 heterocedástico |
| Pressupostos de MQO (linhas 1-5 Gujarati) | Gujarati & Porter cap. 10-13 | Normalidade resíduos, linearidade, variância constante, não-autocorrelação, exogeneidade |
| Logit / probit goodness-of-fit | Hosmer & Lemeshow (2000) *Applied Logistic Regression* 2nd ed | H-L p > 0.05 calibra; AUC > 0.7 discrim. aceitável, > 0.8 bom, > 0.9 excelente |
| Odds ratio, efeito marginal, interpretação | Long (1997) *Regression Models for Categorical and Limited Dependent Variables* | AME vs MER vs MEM; quando cada um; interpretação substantiva |
| Bias-variance tradeoff, RF, bagging | Breiman (2001) *Random Forests*, Machine Learning 45(1) | mtry ≈ √p, ntree ≥ 500, OOB error como validação |
| Gradient boosting theory | Friedman (2001) *Greedy function approximation* | learning_rate × n_estimators tradeoff, shrinkage |
| SHAP (interpretabilidade tree) | Lundberg & Lee (2017) *A Unified Approach to Interpreting Model Predictions*, NeurIPS | Consistência, local accuracy, missingness |
| LIME | Ribeiro et al. (2016) *Why Should I Trust You?*, KDD | Local surrogate, fidelity-interpretability tradeoff |
| Cross-validation (k-fold, LOOCV) | Hastie, Tibshirani & Friedman (2009) *Elements of Statistical Learning* 2nd ed, cap. 7 | k=5 ou k=10; LOOCV alta variância |
| Painel FE/RE (within, between, GMM) | Wooldridge (2010) *Econometric Analysis of Cross-Section and Panel Data* 2nd ed | Hausman test H0: RE consistente, F-test for fixed effects |
| Time series (ARIMA, VAR, ECM) | Box & Jenkins (1976); Lütkepohl (2005) *New Introduction to Multiple Time Series Analysis* | ACF/PACF cutoff, ADF p-value, Johansen trace test |
| Causal inference (PSM, IV, DiD, RDD) | Imbens & Rubin (2015) *Causal Inference for Statistics, Social, and Biomedical Sciences*; Angrist & Pischke (2009) *Mostly Harmless Econometrics* | Balanceamento, SUTVA, monotonicity, exogeneity, parallel trends |

### §4. Formato do relatório (vinculado a Art. 7)

O `export_diagnostic_report` DEVE incluir seção obrigatória
**"Decisões Metodológicas"** (condição vinculante **DA-5**) com:

```markdown
## Decisões Metodológicas

### Algoritmo escolhido
| Candidato | Prós | Contras | Decisão | Critério | Fonte |
|-----------|------|---------|---------|----------|-------|
| OLS | Interpretabilidade, inferência | Linearidade, heteroced. | baseline | — | Gujarati 2009 cap. 7 |
| Ordered logit | Respeita ordinalidade do NPS | Likelihood complexa | ✅ escolhido | Outcome ordinal | Long 1997 cap. 5 |
| Random Forest | Captura não-linearidade | Black-box parcial | ❌ rejeitado (M2+) | Interpretabilidade DA-2 | Breiman 2001 |

### Hiperparâmetros
| Parâmetro | Range testado | Valor final | Métrica de escolha | Fonte |
|-----------|---------------|-------------|--------------------|-------|
| max_depth | {3, 5, 7, 10} | 5 | RMSE em 5-fold CV | Breiman 2001 sec. 4.3 |

### Variáveis (por que estas, não outras)
| Variável | Justificativa teórica | Teste estatístico | Decisão |
|----------|----------------------|-------------------|---------|
| log(receita) | Lei de diminishing returns (Gujarati p. 178) | F parcial p < 0.01 | ✅ mantida |
| dummy_regiao_NE | Heterogeneidade regional (Pindyck cap. 5) | t-test p = 0.03 | ✅ mantida |
| var_propria_4 | Correlação 0.92 com var_propria_3 | VIF = 14 | ❌ removida (VIF > 10) |

### Indicadores de qualidade
| Indicador | Valor observado | Referência (autor) | Veredito |
|-----------|-----------------|--------------------|----------|
| R² | 0.74 | Gujarati (2009) cap. 8: > 0.7 alto | ✅ |
| F-statistic p-value | < 0.001 | Gujarati cap. 8: < 0.05 ok | ✅ |
| VIF máximo | 7.2 | Hair et al. (2010): < 10 ok | ✅ |
| Breusch-Pagan p | 0.18 | Gujarati cap. 11: > 0.05 homoced. | ✅ |

### Bibliografia
- Gujarati, D. & Porter, D. (2009). *Basic Econometrics* 5th ed. McGraw-Hill.
- Long, J. S. (1997). *Regression Models for Categorical and Limited Dependent Variables*. SAGE.
- Hair, J. et al. (2010). *Multivariate Data Analysis* 7th ed. Pearson.
- Breiman, L. (2001). Random Forests. *Machine Learning* 45(1).
- [etc.]
```

### §5. Antipadrões específicos (vinculado a Art. 5)

Em complemento aos antipadrões gerais:

- ❌ **"Usei XGBoost porque é o estado da arte"** (sem análise de bias-variance para o caso).
- ❌ **"R² = 0.85, modelo pronto"** (sem baseline OLS, sem cross-validation, sem teste de estabilidade).
- ❌ **"XGBoost com depth=6 e 1000 árvores"** (sem busca, sem SHAP — viola DA-2).
- ❌ **"Incluí a variável X porque faz sentido"** (sem teoria, sem teste de significância, sem critério de inclusão).
- ❌ **"5 variáveis no modelo"** (sem parcimônia via AIC/BIC, sem teoria econômica, sem teste F incremental).
- ❌ **"Acurácia 92%, ótimo"** (sem precisão/recal por classe, sem análise de erro por segmento, sem curva ROC).
- ❌ **"Modelo calibrado"** (sem Hosmer-Lemeshow ou calibration plot, sem discriminação AUC).
- ❌ **"Não há overfitting"** (sem cross-validation, sem learning curve, sem gap train-vs-test).

### §6. Implementação (vinculado a DA-4, DA-5)

1. **`train_regression` / `train_classifier` / etc.** retornam objeto
   `model_card` em JSON, contendo as 7 perguntas do §2 + referências.
2. **`export_diagnostic_report` renderiza** o `model_card` na seção
   "Decisões Metodológicas" do Markdown/HTML.
3. **`laecon-econometrician-skill`** (M1) tem regras type-checked
   que ABORTAM o save do modelo se qualquer das 7 perguntas estiver
   sem resposta não-vaga.
4. **KB pattern `nps-driver-analysis`** (G6, deadline 2026-07-04)
   serve como template preenchido, demonstrando as 7 perguntas com
   valores reais.

### §7. Princípio de Calibração 20/10 vs 50/1 (PR-1)

> **Origem:** Diretriz explícita do usuário (2026-06-04), em resposta à
> avaliação estrutural PhD do delivery-reviewer. **Target = Level-A
> (padrão global), não PhD, não 4º ano.** Este princípio é transversal
> a LAOS (declarado também em `LAOS/knowledge/padroes-entrega.md`).

**Regra de decisão para tradeoff tempo ↔ qualidade:**

- **+20% tempo → +10% qualidade:** ✅ **adotar**. O usuário aceita
  investir tempo adicional em prol de ganho de qualidade mensurável e
  justificável.
- **+50% tempo → +1% qualidade:** ❌ **rejeitar**. Overhead desproporcional.
  Procurar alternativa mais eficiente ou refatorar a abordagem.
- **Curva contínua:** Calcular `ratio = Δqualidade% / Δtempo%`. Se
  `ratio ≥ 0.5` (1% de qualidade por 2% de tempo), o investimento é
  razoável. Se `ratio < 0.1`, é over-engineering.

**Implicações práticas:**

1. **Anti-PhD:** Evitar rigor acadêmico sem propósito operacional.
   - ❌ Adotar 6-stage Fagan inspection com 4 roles separados quando 1
     reviewer + 1 preflight mecânico cobre 95% dos defeitos.
   - ❌ Implementar IV&V NASA-STD-8739.8 completo para um projeto solo.
   - ❌ Cohen 1960 κ inter-rater quando há apenas 1 revisor.
2. **Anti-4º-ano:** Evitar simplificação excessiva que perde o valor
   entregue.
   - ❌ Modelo = média aritmética porque "já passa no teste".
   - ❌ Review que vira rubber-stamp porque "está tudo bem".
   - ❌ Detalhamento metodológico raso quando a pergunta exige rigor
     econométrico (viola Art. 10 §2).
3. **Target Level-A (global standard):**
   - **Calibração:** igual ou superior ao que 80% dos praticantes
     fariam para o mesmo problema.
   - **Comparação:** benchmark com peers do campo (papers, kaggle
     winners, papers NIPS/ICML/Quirk's para NPS).
   - **Transparência:** toda decisão registrada com trade-off explícito.

**Aplicação ao Art. 10:**

- As 7 perguntas do §2 são **vinculantes** (ratio ≈ ∞ — perguntas têm
  custo baixo, valor alto para auditabilidade).
- A bibliografia com valores numéricos (§3) é **vinculante** (ratio > 1
  — citação explícita vale mais que 10× o tempo de lookup).
- Múltiplas alternativas rejeitadas em tabela (§4) é **vinculante**
  (ratio > 0.5 — análise de trade-off é o que diferencia o modelo).
- **NÃO vinculantes** (ratio < 0.1): reproduzir integrais Gujarati no
  relatório, derivar fórmulas no apêndice, implementar todo paper de
  referência do zero, validar com 10 seeds quando 3 já estabilizam.

**Aplicação transversal ao delivery-reviewer (LAOS):**

- Reforma de 6-stage Fagan → 5-stage Fagan (Stage 0 mecânico + Stages
  1-4 semânticos) com 1 reviewer + 1 preflight script é **Level-A**.
- Cobertura de 5 checks mecânicos (YAML+arithmetic, paths, secrets,
  cross-ref, impl-code) + 4 stages semânticos = ratio ~3.75× (acima
  do limiar 0.5 do PR-1).

### §8. Protocolo de Revisão Metodológica (aprovado LACOUNCIL `2505af1e`, 2026-06-13)

> **Origem:** Proposta `2505af1e-b28a-414c-9af3-88bdc2c7dcab` (maioria,
> 4/4 SIM, 2026-06-13). Motivação: o artigo "Banca Simulada" (Sérgio
> Camargos, 2026) demonstra que decisões sem revisão estruturada
> geram fragilidades detectáveis apenas por examinadores externos.
> Este §8 transforma a revisão de pares em processo mecânico.

Nenhum modelo é considerado completo sem passar pelo **Protocolo de
Revisão Metodológica** — uma avaliação em 5 dimensões aplicada antes
de `export_diagnostic_report`. O gate gera
`artifacts/reviews/methodological-review.md`.

#### As 5 dimensões obrigatórias

| # | Dimensão | Pergunta-gatilho | Fonte inspiradora |
|---|----------|-------------------|-------------------|
| 1 | **Alinhamento** | O método serve ao objetivo declarado? Os dados sustentam o método? | Prompt 2 — Alinhamento interno |
| 2 | **Evidência** | Cada decisão está classificada como A (sustentada), B (parcial), C (sem sustentação)? | Prompt 3 — Afirmações sem lastro |
| 3 | **Referências** | Quais autores/debates estão ausentes? Nível de confiança de que existem? | Prompt 4 — Lacunas de literatura |
| 4 | **Robustez** | Se o modelo falhasse, qual premissa estaria errada? Qual variável ignorada? | Prompt 5 — Pré-mortem |
| 5 | **Autocrítica** | O que eu não sei que não sei? Que crítica um examinador rigoroso faria? | Prompt 1 — Arguição simulada |

#### Formato do output

```markdown
## Revisão Metodológica — [nome do modelo]

### Dimensão 1: Alinhamento
- Objetivo declarado: ...
- Método escolhido: ...
- Veredito: ALINHADO / PARCIAL / DESALINHADO
- Lacuna (se houver): ...

### Dimensão 2: Evidência
| Decisão | Classe | Justificativa |
|---------|--------|---------------|
| Algoritmo X | A | Citação + tabela comparativa |
| Hiperparâmetro Y | B | Range testado mas sem referência |
| Variável Z | C | "Parece relevante" — VIOLAÇÃO |

### Dimensão 3: Referências
| Ausência | Confiança de que existe | Ação |
|----------|------------------------|------|
| [autor] sobre [tópico] | alto/médio/baixo | Verificar no Google Scholar |

### Dimensão 4: Robustez (Pré-mortem)
- Causa principal de falha potencial: ...
- Ponto cego: ...
- Mitigação: ...

### Dimensão 5: Autocrítica
- Crítica que um examinador faria: ...
- O que não sei que não sei: ...
```

#### Implementação

1. **G4 (2026-06-14):** §8 publicado com formato completo.
2. **G6 (2026-07-04):** KB entry `methodological-review-protocol.md`
   com template preenchido usando caso NPS como exemplo.
3. **M1 (2026-07-04):** `export_diagnostic_report` inclui seção
   "Revisão Metodológica" quando o protocolo foi executado.
4. **Auto-documentação:** toda decisão já tomada na LAECON
   (DuckDB, SHAP sobre LIME, sequência de tools) documentada em
   `laecon/own-decisions.md` com as 5 dimensões.

#### Vinculação

- Vinculado a: Art. 10 §1 (hipótese refutável), Art. 3 (interpretabilidade),
  condições DA-4 e DA-5.
- **Vinculado desde G4** (não espera STABLE).

---

> **Nota:** Esta Constitution atingiu status **COMPLETE** em 2026-06-13.
> Os Artigos 4, 6, 7, 9 foram expandidos com conteúdo completo (G4 gate).
> O **Artigo 10** é vinculante desde o BASIC (2026-06-04). O **§8**
> foi adicionado em 2026-06-13 por proposta LACOUNCIL `2505af1e`
> (aprovada 4/4 SIM). As condições vinculantes do Conselho (17 itens:
> 5 DA, 3 DD, 4 AE, 5 DR, ver
> `../LAOS/projects/_meta/capability-evolution/laecon.md`) estão
> refletidas nos artigos correspondentes.
