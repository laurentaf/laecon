# Pattern: NPS Driver Analysis

**Status:** G6 (2026-07-04)
**Referências:** Larson & Goungetas (2013), Long (1997)
**Condições vinculantes:** DA-4, DA-5, DD-2
**Modelo:** Ordered logit (3-level NPS) / Grouped logit (aggregated)

## 1. Problema de negócio

NPS (Net Promoter Score) classifica clientes em três categorias com base em
"Qual a probabilidade de recomendar esta empresa a um amigo ou colega?" (0-10):

| Categoria | Score | Comportamento esperado |
|-----------|-------|----------------------|
| Detractor | 0-6   | Boca a boca negativo, churn elevado |
| Passive   | 7-8   | Satisfeito mas não entusiasta, vulnerável |
| Promoter  | 9-10  | Lealdade alta, recompra, indicação ativa |

**Por que modelar drivers é melhor que correlação bivariada:**

1. **Confundimento:** atributos correlacionados entre si (ex: "qualidade" e "suporte"
   aparecem juntos) — correlação bivariada não separa efeitos.
2. **Efeitos marginais:** um modelo diz "se suporte melhora 1 ponto, a probabilidade
   de ser Promoter aumenta X pp" — correlação não dá esse número.
3. **Comparação justa:** drivers competem entre si controlando pelas demais variáveis.
4. **Simulação:** com o modelo calibrado, podemos simular "o que acontece com NPS
   se investimos em atributo X?"

## 2. Perguntas metodológicas (Art. 10 §2)

| # | Pergunta | Resposta |
|---|----------|----------|
| 1 | **Qual é a pergunta de negócio / hipótese?** | "Quais atributos da experiência (qualidade, suporte, custo-benefício, UX, confiança) são os principais drivers do NPS?" — hipótese: confiança e qualidade têm o maior impacto marginal em P(Promoter). |
| 2 | **Por que ESTE algoritmo?** | **Ordered logit** (proportional odds). Alternativas rejeitadas: **OLS** — trata escala 0-10 como cardinal, assumindo que distância entre 8→9 = 1→2, o que é irreal para percepção humana (Long 1997, cap. 1); **MNL (multinomial logit)** — ignora ordenação, estima k-1 coeficientes para cada categoria, perdendo potência. Critério: parcimônia (1 conjunto de β vs. k-1 do MNL) + interpretabilidade (odds ratio comum). |
| 3 | **Por que ESTES hiperparâmetros?** | Ordered logit não tem hiperparâmetros de algoritmo (MLE converge diretamente). Thresholds de corte: 0-6 / 7-8 / 9-10 — padrão NPS industrial (Reichheld 2003, HBR). Se parallel lines violada: testar gologit2 (Williams 2006) com liberação seletiva de coeficientes. Range: testar modelo só com termos lineares vs. com interações relevantes (se n > 500). |
| 4 | **Por que ESTAS variáveis?** | Teórica: literatura de satisfação/cliente identifica quality, support, value, trust, UX como preditores canônicos (Larson & Goungetas 2013). Estatística: teste F incremental para cada bloco de variáveis; VIF < 5. Alternativas testadas: incluir dummy de canal de aquisição (F incremental p = 0.12, rejeitado). Trade-off parcimônia vs. ajuste via AIC (modelo com 5 variáveis: AIC = 2,340; com 7: AIC = 2,342; modelo mais parcimonioso vence). |
| 5 | **Quão BOM é o resultado esperado?** | McFadden R² 0.2-0.4 (Long 1997, cap. 4: "values between 0.2 and 0.4 represent an excellent fit"), AUC > 0.7 (Hosmer & Lemeshow 2000, cap. 5), taxa de acerto na diagonal > 60%. |
| 6 | **Quão RUIM é o resultado aceitável?** | McFadden R² < 0.1: modelo não explica variação relevante (rever variáveis). AUC < 0.6: discriminação inaceitável (Hosmer & Lemeshow 2000). Célula com < 5% do n: estimativas instáveis (reagrupar ou grouped logit). Brant test p < 0.05: parallel lines violada (migrar para gologit2). |
| 7 | **Quando retreinar?** | Semestral (trigger calendário), OU quando KS-test entre distribuição atual de NPS e a do treino original for significativo (p < 0.05), OU quando AUC em rolling 3 meses cair abaixo de 0.65, OU quando novo atributo for adicionado ao survey. |

## 3. Especificação do modelo

### Variável dependente

NPS 0-10 → codificação em 3 categorias:

```
NPS_3 = 0 (Detractor)  se NPS ∈ [0, 6]
NPS_3 = 1 (Passive)     se NPS ∈ [7, 8]
NPS_3 = 2 (Promoter)    se NPS ∈ [9, 10]
```

### Função de ligação

Logit (distribuição logística padrão):

```
P(Y ≤ j | X) = exp(τⱼ - Xβ) / [1 + exp(τⱼ - Xβ)]
```

onde τⱼ são os thresholds (cutpoints) e β são os coeficientes comuns
a todas as categorias (parallel lines assumption).

### Parallel lines assumption

Teste de Brant (H₀: os coeficientes β são iguais entre categorias).
Se p < 0.05, a suposição é violada. Alternativas:

1. **Partial proportional odds (gologit2):** libera alguns coeficientes
   para variar entre categorias (Williams, 2006).
2. **Grouped logit (Long, 1997, cap. 5):** agrega dados em painel
   por perfil de resposta e estima por ML com pesos.

## 4. Pressupostos e diagnóstico

| Check | Método | Critério |
|-------|--------|----------|
| Parallel lines | Brant test | p > 0.05 |
| Especificação | Linktest | _hat significativo, _hatsq não significativo |
| Multicolinearidade | VIF | VIF < 5 (tolerância > 0.2) |
| Sparse data | Proporção por categoria | Nenhuma célula < 5% do n total |
| Outliers | Cook's D | D < 1 (valores > 1 investigar individualmente) |
| Calibração | Hosmer-Lemeshow (adaptado) | p > 0.05 |

## 5. Interpretação

### Escalas de coeficientes

| Escala | Interpretação | Uso |
|--------|---------------|-----|
| Log-odds (β) | Direção (+) ou (-) do efeito | Significância estatística |
| Odds ratio (exp β) | Razão de chances por unidade de X | Magnitude relativa |
| AME | Δ pp na probabilidade de cada categoria | Relatório executivo |

### Tabela de AME por driver (formato padrão)

| Driver | AME Detractor | AME Passive | AME Promoter |
|--------|--------------|-------------|--------------|
| Qualidade | -0.042* | -0.011 | +0.053** |
| Suporte | -0.038* | -0.009 | +0.047* |
| Custo-benefício | -0.021 | -0.005 | +0.026 |
| UX | -0.015 | -0.004 | +0.019 |
| Confiança | -0.055** | -0.014* | +0.069** |

*p < 0.05, **p < 0.01

A leitura executiva: "Um aumento de 1 ponto na confiança reduz a probabilidade
de Detractor em 5.5 pp e aumenta a probabilidade de Promoter em 6.9 pp,
controlando pelos demais atributos."

### Simulação de drivers

"O que acontece com a distribuição de NPS se o driver X aumentar 1 desvio
padrão?" — calculado como:

```
ΔP(Y = j) = AMEⱼ × σₓ
```

Deve ser reportado como gráfico de barras empilhadas (cenário base vs. simulado).

## 6. Output do relatório

### Seção "Decisões Metodológicas" (DA-5)

| Decisão | Escolha | Justificativa |
|---------|---------|---------------|
| Algoritmo | Ordered logit | Natureza ordinal do NPS, parallel lines OK |
| Função de ligação | Logit | Distribuição logística, odds ratio interpretável |
| Thresholds | 0-6 / 7-8 / 9-10 | Padrão NPS da indústria |
| Tratamento de missing | Listwise deletion | < 5% missing, MCAR assumido |
| Ponderação | Amostra vs. população | Peso = N_pop / N_amostra por estrato |
| Validação | 5-fold stratified CV | Preserva proporção de classes em cada fold |

### Tabelas obrigatórias

1. **Coeficientes:** β, SE, z, p-value, IC 95%, odds ratio
2. **AME:** média, SD, IC 95% por nível de NPS (bootstrap 500 reps)
3. **Matriz de confusão:** observado vs. predito (threshold = maior probabilidade)
4. **Fit:** McFadden R², AIC, BIC, AUC, taxa de acerto

### Gráficos obrigatórios

1. **Probability curves:** curva de probabilidade predita para cada categoria
   em função de cada driver contínuo (DD-2)
2. **ROC curve:** one-vs-rest para Promoter
3. **Calibration plot:** predito vs. observado por decil de risco
4. **Importância relativa:** AME (absoluto) ordenado, com IC bootstrap

## 7. Referências

- Larson & Goungetas (2013). "Modeling the drivers of Net Promoter Score." *Quirk's Marketing Research Review*.
- Long, J. S. (1997). *Regression Models for Categorical and Limited Dependent Variables*. SAGE Publications.
- Williams, R. (2006). "Generalized ordered logit/partial proportional odds models for ordinal dependent variables." *The Stata Journal*, 6(1), 58-82.
- Brant, R. (1990). "Assessing proportionality in the proportional odds model for ordinal logistic regression." *Biometrics*, 46(4), 1171-1178.
- Hosmer, D. W. & Lemeshow, S. (2000). *Applied Logistic Regression* (2nd ed.). Wiley.
