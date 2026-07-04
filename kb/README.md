# KB — Knowledge Base da capability LAECON

**Status:** G6 (2026-07-04)
**Vinculado a:** Constitution Art. 10 (Detalhamento Metodológico Extremo)
               + condições DA-4, DA-5, DD-2, DR-1

Este diretório armazena o **conhecimento de domínio** da capability:
padrões preenchidos, exemplos canônicos, troubleshooting, e templates
para as 7 perguntas metodológicas obrigatórias (Constitution Art. 10 §2).

---

## Estrutura

```
kb/
├── README.md                      ← este arquivo
├── patterns/                      ← patterns canônicos (templates preenchidos)
│   ├── nps-driver-analysis.md     ✅ G6 — caso de uso principal
│   │                                  (Larson & Goungetas + Long 1997)
│   ├── panel-fe-re.md             ← M2 — DA-1
│   └── arima-var-ecm.md           ← M3 — DA-1
├── references/                    ← referências com valores numéricos
│   ├── gujarati-porter.md         ✅ R², F, VIF, DW, BP test, OLS diagnostics
│   ├── hosmer-lemeshow.md         ✅ H-L test, AUC, McFadden R², logit vs probit
│   ├── long-1997.md               ✅ AME/MER/MEM, odds ratio, ordered logit, grouped logit
│   ├── breiman-2001-rf.md         ✅ mtry, ntree, OOB, importance, RF vs OLS
│   ├── friedman-2001-gbm.md       ✅ lr×n_estimators, XGBoost/LightGBM/CatBoost
│   ├── shap-lime.md               ✅ SHAP axioms, plots, LIME, comparison
│   ├── cross-validation.md        ✅ k-fold, LOOCV, stratified, nested, bias-variance
│   └── model-selection.md         ✅ Decision flowchart: outcome type → goal → model
├── troubleshooting/               ← M1+ — problemas comuns e diagnóstico (TODO)
│   ├── empty-dataframe.md         ← guarda DR-1 (M1)
│   ├── convergence-issues.md      ← não-convergência em logit/ARIMA (M1)
│   └── pydantic-version-conflicts.md  ← deferral de neuralprophet/gx (M1)
├── review-protocol.md             ✅ G6 — protocolo de revisão (Art. 10 §8)
└── templates/                     ← M1+ — templates de relatório (TODO)
    ├── diagnostic-report.md       ← Art. 7 + Art. 10 §4 (M1)
    └── model-card.json            ← output de `train_*` (DA-4) (M1)

> **Nota:** `troubleshooting/` e `templates/` são planejados para M1+. Sua ausência não bloqueia G6.
```

---

## Como usar este KB (a partir de M1)

1. Antes de rodar qualquer `train_*`, consultar `kb/patterns/<caso>.md`
   para ver o template preenchido e identificar lacunas.
2. Preencher as 7 perguntas (Constitution Art. 10 §2) usando o template
   `kb/templates/model-card.json`.
3. Conferir valores numéricos em `kb/references/<autor>.md`.
4. Se aparecer problema novo, documentar em `kb/troubleshooting/`.

---

## Cataloga (vinculado às 17 condições)

| KB entry | Condição atendida | Marco |
|----------|-------------------|-------|
| `patterns/nps-driver-analysis.md` | DA-4 (exemplo completo) + DA-5 (relatório) | G6 |
| `references/gujarati-porter.md` | DA-4 (valores de referência oficiais) | M1 |
| `references/hosmer-lemeshow.md` | DA-4 (logit goodness-of-fit) | M1 |
| `references/long-1997.md` | DA-4 (AME/MER/MEM, ordered/grouped logit) | M1 |
| `references/breiman-2001-rf.md` | DA-4 (RF hyperparams) + DA-2 (interpretabilidade) | M2 |
| `references/friedman-2001-gbm.md` | DA-4 (GBM hyperparams, boosting theory) | M2 |
| `references/shap-lime.md` | DA-2 (tree interpretability) | M2 |
| `references/cross-validation.md` | DA-4 (model selection methodology) | M1 |
| `references/model-selection.md` | DA-4 (decision flowchart, sample size rules) | M1 |
| `templates/model-card.json` | DA-4 (enforcement type-checked) | M1 |
| `templates/diagnostic-report.md` | DA-5 (seção Decisões Metodológicas) | M1 |
| `troubleshooting/empty-dataframe.md` | DR-1 (P0 padroes-entrega.md) | M1 |
| `review-protocol.md` | DA-4, DA-5, DR-1 (revisão metodológica 5 dim.) | G6 |

---

## Protocolo de Revisão

O `kb/review-protocol.md` implementa o Art. 10 §8 da Constitution.
5 dimensões de revisão (D1-D5) que toda modelagem deve passar antes
de ser considerada completa. Consumido pelo `delivery-reviewer` em G7.

---

## Learning materials

Materiais de estudo para novos contributors da capability:

- **Econometria base:** Gujarati & Porter (2009), caps. 1-8, 13-15
- **Modelos categóricos:** Long (1997), caps. 1-7
- **NPS drivers:** Larson & Goungetas (2013), Quirk's
- **ML interpretável:** Molnar (2022). *Interpretable Machine Learning*.
  Disponível em https://christophm.github.io/interpretable-ml-book/
- **Boosting:** Friedman (2001). "Greedy Function Approximation." *Annals of Statistics*.
- **Random Forests:** Breiman (2001). *Machine Learning*, 45(1), 5-32.
