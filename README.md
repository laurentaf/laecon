# LAECON

> **Econometrics is the spine. ML is the muscle. Likelihood explicit, not black box.**

Capability LAOS para **econometria + machine learning interpretável**, com
likelihood explícita. Posiciona-se como ponte entre `latade` (data
engineering) e `ladesign`/`lan8n` (consumo dos outputs em visual/automação).

**Status atual:** BASIC (scaffold entregue 2026-06-04) — 30 dias para
STABLE (deadline 2026-07-04).
**17 condições vinculantes** do Conselho (5 DA + 3 DD + 4 AE + 5 DR).
Detalhamento em `Condições vinculantes` abaixo.

**Meta-projeto:** `../LAOS/projects/_meta/laecon-capability/project.yaml`
**Tracking:** `../LAOS/projects/_meta/capability-evolution/laecon.md`
**ADR-002:** `../LAOS/projects/_meta/adr/ADR-002-laecon-creation.md`
**Proposta LACOUNCIL:** `cbe2d8ef-f65c-4d0e-8960-ea99527ab39f` (aprovada 2026-06-04, 4/4 SIM, supermaioria)

---

## Capabilities

### Implementadas (BASIC, 2026-06-04)
- **MCP server funcional** com 9 tools (2 implementadas, 7 stubs prontas para M1).
- **MCP tools iniciais:** `health`, `list_supported_operations`,
  `train_regression`, `train_classifier`, `validate_assumptions`,
  `interpret_model`, `evaluate_model`, `predict`,
  `export_diagnostic_report`.
- **Workflow de validação econométrica:** OLS → diagnostics → interpretação → predição → report.

### Planejadas
- **M1 (STABLE, +30d, 2026-07-04):** Todos os 9 tools funcionais. `compare_models`.
- **M2 (+60d, 2026-08-04):** Tree-based models COM SHAP (RF, GBT) — condição DA-2.
  Painel FE/RE (DA-1). Model registry operacional (AE-1). Cross-validation.
- **M3 (+90d, 2026-09-04):** Time series (ARIMA, VAR) + causal inference
  (PSM, IV, DiD) (DA-1). Simulador de drivers de NPS.

---

## Arquitetura

```
laecon/
├── pyproject.toml              # Python project + deps (statsmodels, sklearn, etc.)
├── README.md
├── CONSTITUTION.md             # 9 artigos (skeleton BASIC; completo G4)
├── .gitignore
├── laecon/
│   ├── __init__.py
│   └── mcp/
│       ├── __init__.py
│       └── server.py           # MCP server (BASIC: health + list_supported_operations)
├── kb/                         # Knowledge base (placeholder; completo G6)
│   └── README.md
├── skills/                     # Skills library (placeholder; completo M1)
│   └── README.md
└── tests/                      # Testes (M1)
    └── test_mcp_health.py
```

---

## MCP Tools

| # | Tool | BASIC | M1 (STABLE) | Descrição |
|---|------|-------|-------------|-----------|
| 1 | `health` | ✅ | ✅ | Liveness probe |
| 2 | `list_supported_operations` | ✅ | ✅ | Catálogo de operações |
| 3 | `train_regression` | ⚠️ stub | ✅ | OLS, GLS, WLS, robust SE |
| 4 | `train_classifier` | ⚠️ stub | ✅ | logit, probit, ordered, grouped |
| 5 | `validate_assumptions` | ⚠️ stub | ✅ | Breusch-Pagan, DW, VIF, JB |
| 6 | `interpret_model` | ⚠️ stub | ✅ | marginal effects, odds ratios |
| 7 | `evaluate_model` | ⚠️ stub | ✅ | R², AIC, BIC, AUC, confusion |
| 8 | `predict` | ⚠️ stub | ✅ | predição + IC (requer `model_id`) |
| 9 | `export_diagnostic_report` | ⚠️ stub | ✅ | markdown/HTML com plots |

Stubs retornam `not_implemented_yet` com mensagem clara apontando para
o tracking e a proposta LACOUNCIL.

---

## Uso

### MCP server (para orquestração LAOS)

```bash
cd E:\projects\laecon
uv sync
uv run python -m laecon.mcp.server
```

### Como biblioteca Python

```python
from laecon.mcp.server import health, list_supported_operations

print(health())
# {'status': 'ok', 'capability': 'laecon', 'version': '0.1.0-BASIC', ...}

ops = list_supported_operations()
print(f"Total tools: {len(ops['operations'])}")
print(f"Implemented: {sum(1 for o in ops['operations'] if o['status'] == 'implemented')}")
```

### venv isolado (padrão LAOS)

```bash
cd E:\projects\laecon
uv sync   # cria .venv/ local
# Use `..\laecon\.venv\Scripts\python` para chamadas explícitas
```

---

## Bibliotecas Python instaladas

As principais bibliotecas da área (pedido do usuário), organizadas em
`[main]` (instaladas por padrão) e `[extras]` (opt-in por domínio).

### Main (`uv sync`)

| Categoria | Libs |
|-----------|------|
| MCP | `mcp[cli]` 1.x |
| Numerics | `numpy` 2.x, `scipy` 1.x, `pandas` 3.x |
| Econometrics (espinha) | `statsmodels` 0.14, `linearmodels` 7.0 (DA-1) |
| Classical ML | `scikit-learn` 1.6 |
| Gradient boosting | `xgboost` 3.2, `lightgbm` 4.6, `catboost` 1.2 |
| Interpretabilidade | `shap` 0.48, `lime` 0.2 (DA-2) |
| Hyperopt | `optuna` 4.9 |
| Visualização | `matplotlib` 3.10, `seaborn` 0.13 |
| Utilities | `pydantic` 2.x, `tqdm`, `joblib`, `tabulate`, `pyyaml`, `python-dotenv`, `requests` |

### Extras (`uv sync --extra <name>`)

| Extra | Libs | Marco |
|-------|------|-------|
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy` | M1 |
| `notebooks` | `jupyter`, `jupyterlab`, `ipykernel`, `voila`, `nbconvert` | M1 |
| `causal` | `dowhy`, `causalml`, `econml`, `zepid` | M3 (DA-1) |
| `timeseries` | `statsmodels[tsa]`, `arch`, `pmdarima`, `prophet`, `tslearn` | M3 (DA-1) |
| `viz` | `plotly`, `altair`, `bokeh`, `plotnine`, `wordcloud` | M1 |
| `feature_eng` | `category-encoders`, `feature-engine`, `featuretools`, `tsfresh` | M2 |
| `nlp` | `nltk`, `spacy`, `textblob` | M4+ |
| `validation` | `pandera`, `pytest-mock`, `pytest-asyncio` | M1 (DR-1) |
| `dl` | `torch`, `torchvision`, `pytorch-lightning`, `transformers`, `datasets` | M4+ |
| `bayesian` | `pymc`, `arviz`, `emcee` | M4+ |
| `symbolic` | `sympy`, `patsy` | util |
| `graphs` | `networkx`, `python-igraph` | util |
| `full` | **todas as acima** (inlined) | — |

### Instalação recomendada por milestone

```bash
# BASIC (default) — só main
uv sync

# M1 — STABLE (testing + notebooks + viz + validation)
uv sync --extra dev --extra notebooks --extra viz --extra validation

# M2 — Tree-based com SHAP + feature eng (DA-2)
uv sync --extra feature_eng

# M3 — Time series + causal inference (DA-1)
uv sync --extra causal --extra timeseries

# M4+ — DL + Bayesian + NLP
uv sync --extra dl --extra bayesian --extra nlp

# Tudo de uma vez (cuidado: ~3 GB, 200+ pacotes)
uv sync --extra full
```

### Pacotes deferidos (M3+, não rodam no BASIC)

Por conflitos de pin com pydantic 2.x / altair 5.x já no main, as
seguintes libs ficam fora do `[full]` até M3+:

- `neuralprophet` — exige pydantic 1.x; só funciona em Python < 3.13
- `great-expectations` — pina `altair<5.0`; conflito com `viz`
- `causalimpact` — não priorizado, redundante com `dowhy+econml`

Quando M3 abrir a frente causal, a estratégia é:
1. Subir pydantic 2.x e propagar para todo o ecossistema.
2. Esperar gx migrar para altair 5.x (já em roadmap upstream).
3. Adicionar neuralprophet (que migrou para pydantic 2.x) como
   upgrade opcional.

---

## Fontes de Grounding

| Fonte | Tipo | Uso |
|-------|------|-----|
| Gujarati & Porter, *Basic Econometrics 5th ed* (2009) | Livro, 946 pp. | Base teórica econométrica |
| Larson & Goungetas, *"Modeling the drivers of Net Promoter Score"* (Quirk's, 2013) | Artigo | Driver analysis de NPS (ordered logit) |
| J. Scott Long, *Regression Models for Categorical and Limited Dependent Variables* | Livro | Programação de likelihood ordered/grouped |

Fontes em `E:\projects\_commomdata\` (convenção `_commomdata` documentada
em `../LAOS/knowledge/data-conventions.md`).

---

## Convenções

- **Dados compartilhados:** `E:\projects\_commomdata\` é a convenção
  transversal para fontes de grounding reutilizáveis (livros, papers,
  datasets públicos). Documentada em `../LAOS/knowledge/data-conventions.md`.
- **venv isolado:** `../laecon/.venv/` gerenciado por `uv sync` (não
  compartilha venv com LAOS/LATADE/LACOUNCIL).
- **Naming de modelo:** `<dataset_slug>_<algorithm>_<YYYYMMDD>_<hash[:6]>`
  (Constitution Art. 6, condição AE-1). Exemplo:
  `nps_2025q4_ordered_logit_20260604_a3f9c1`.
- **Reports em português** para outputs LAOS-internos; inglês quando
  consumido por capability externa (ex: ladesign para cliente externo).
- **17 condições vinculantes** do Conselho (proposta LACOUNCIL
  `cbe2d8ef-...`) bloqueiam promoção a STABLE em G7. Destas, **DA-4 e
  DA-5 (Detalhamento Metodológico Extremo) são vinculantes desde o
  BASIC** via Constitution Art. 10 — não esperam G4. Detalhes em
  `../LAOS/projects/_meta/capability-evolution/laecon.md` e no
  `Constitution.md` Art. 10 (com as 7 perguntas, referências com
  valores numéricos e formato de relatório).

---

## Princípios Fundadores

1. **Likelihood explícita, não black box.** Todo modelo tem likelihood
   conhecida e documentada. Tree-based models (RF, GBT) só entram com
   SHAP explícito (condição DA-2).
2. **Interpretabilidade antes de acurácia.** Coefs com SE, p-values,
   IC, efeitos marginais antes de métricas de scoring.
3. **Econometria é a espinha, ML é o músculo.** Modelagem estatística
   vem primeiro; ML preditivo complementa.
4. **Local-first, DuckDB como engine.** Reusa runtime de LATADE.
   Suficiente para o escopo. Sem Spark/Dask.
5. **Anti-overfitting explícito.** Cross-validation, regularização
   documentadas. Toda métrica de scoring vem com baseline.
6. **DataFrame empty guards** (P0 `padroes-entrega.md`, condição DR-1).
   Toda função que opera sobre dados verifica emptiness e retorna
   mensagem amigável, não `IndexError` ou `ValueError`.
7. **Detalhamento metodológico extremo** (Constitution Art. 10, condições
   DA-4 e DA-5, **vinculante desde o BASIC**). Toda decisão
   metodológica — escolha de algoritmo, hiperparâmetro, conjunto de
   variáveis — responde 7 perguntas obrigatórias: (i) hipótese de
   negócio, (ii) por que ESTE algoritmo vs ≥2 alternativas rejeitadas,
   (iii) por que ESTES hiperparâmetros, (iv) por que ESTAS variáveis,
   (v) quão bom é o resultado esperado (com autor e valor de
   referência), (vi) quão ruim é o aceitável (threshold de abortar),
   (vii) quando retreinar. `export_diagnostic_report` tem seção
   obrigatória "Decisões Metodológicas" com tabelas + bibliografia
   citada (Gujarati, Long, Breiman, Hosmer-Lemeshow, etc.). Ver
   `Constitution Art. 10` para o protocolo completo.

---

## Tracking

- **Meta-projeto:** `../LAOS/projects/_meta/laecon-capability/project.yaml`
- **Evolução BASIC→STABLE:** `../LAOS/projects/_meta/capability-evolution/laecon.md`
- **ADR-002:** `../LAOS/projects/_meta/adr/ADR-002-laecon-creation.md`
- **Convenção `_commomdata`:** `../LAOS/knowledge/data-conventions.md`

---

## Licença

MIT (a confirmar com usuário).
