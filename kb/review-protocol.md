# Protocolo de Revisão Metodológica

**Status:** G6 (2026-07-04)
**Vinculado a:** Constitution Art. 10 §8
**Condições vinculantes:** DA-4, DA-5, DR-1

## Objetivo

Garantir que toda modelagem econométrica/ML na capability LAECON passa por
revisão metodológica antes de ser considerada completa. Este protocolo
implementa o Art. 10 §8 da Constitution.

## 5 Dimensões de Revisão

### D1 — Especificação

- [ ] A variável dependente é adequada ao modelo escolhido?
- [ ] As variáveis independentes têm justificativa teórica?
- [ ] A função de ligação é apropriada?
- [ ] Há interações ou termos não-lineares justificados?

### D2 — Diagnóstico

- [ ] Pressupostos do modelo foram testados?
- [ ] Heterocedasticidade? (Breusch-Pagan, White)
- [ ] Multicolinearidade? (VIF < 5)
- [ ] Especificação? (linktest, RESET)
- [ ] Outliers/influência? (Cook's D, leverage)

### D3 — Interpretação

- [ ] Coeficientes estão na escala correta (log-odds, odds ratio, AME)?
- [ ] IC 95% reportados?
- [ ] Efeitos marginais calculados para modelos não-lineares?
- [ ] Simulações quando relevante?

### D4 — Reprodutibilidade

- [ ] seed aleatória fixada?
- [ ] Dados de treino/teste separados?
- [ ] Cross-validation realizada?
- [ ] Código e dados disponíveis?

### D5 — Documentação

- [ ] 7 perguntas metodológicas (Art. 10 §2) respondidas?
- [ ] Seção "Decisões Metodológicas" (DA-5) preenchida?
- [ ] Bibliografia com valores numéricos de referência?
- [ ] Model card preenchido?

## Fluxo de Revisão

```
1. Autor do modelo preenche as 7 perguntas (model card)
2. Aplica as 5 dimensões como checklist
3. Se todas as dimensões OK → modelo aprovado
4. Se alguma dimensão falha → retorna com correções
5. Na segunda falha → escalar para revisão por par
```

## Integração

- Este protocolo é consumido pelo `delivery-reviewer` em G7
- Verificado mecanicamente via preflight_check.py (cross-reference D1-D5)
- Implementado programaticamente nos MCP tools em M1
