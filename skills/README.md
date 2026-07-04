# LAECON Skills

Skills library para a capability LAECON. Skills iniciais serão entregues
em M1 (deadline 2026-07-04, +30 dias).

## Skills planejadas

- **`laecon-econometrician-skill`** (M1) — interpretação, validação de
  pressupostos, boas práticas econométricas. Equivalente a um
  econometrista sênior revisando outputs antes de entregar ao cliente.
  Cobre: Gujarati staples (heteroced, autocorr, multicolin, normality),
  marginal effects interpretation, model selection (AIC/BIC/likelihood-ratio).

- **`laecon-nps-driver-skill`** (M1) — caso canônico NPS: ordered logit,
  simulação de drivers, comunicação executiva. Baseado no artigo da
  Quirk's (Larson & Goungetas, 2013). Cobre: estimar modelo, extrair
  probabilidades por outcome level, simular "what-if" de drivers,
  gerar relatório executivo.

- **`laecon-shap-explainer-skill`** (M2, junto com `compute_shap` tool
  — condição DA-2 anti-black-box) — interpretação de tree-based models
  via SHAP. Cobre: SHAP summary plot, SHAP dependence plot, force plot,
  interaction values.

- **`laecon-time-series-skill`** (M3, DA-1) — séries temporais: ARIMA,
  VAR, estacionariedade (ADF, KPSS), cointegração (Engle-Granger,
  Johansen), modelos ECM.

- **`laecon-causal-inference-skill`** (M3, DA-1) — inferência causal:
  PSM (propensity score matching), IV (instrumental variables, 2SLS),
  DiD (difference-in-differences), RDD (regression discontinuity).

- **`laecon-panel-data-skill`** (M2, DA-1) — dados em painel: fixed
  effects, random effects, Hausman test, modelos dinâmicos.

## Convenção de skill

Cada skill segue o padrão da skill library da capability LADESIGN
(ver `../ladesign/skills/`). Formato:

1. SKILL.md com frontmatter YAML
2. Workflow passo-a-passo
3. Exemplos de uso
4. Referências (Gujarati cap., papers)
