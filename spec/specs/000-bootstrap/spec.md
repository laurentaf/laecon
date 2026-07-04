# SPEC-000-BOOTSTRAP: LAECON Capability

**Status:** ACEITO
**Version:** 1.0
**Authors:** Laurent (economista, data architect)
**Owner:** Laurent

---

## 1. Executive Summary

Criar a capability LAECON para o LAOS, oferecendo modelagem econométrica e ML interpretável com likelihood explícita. A capability é econometrics-first (Gujarati & Porter 2009), com ML interpretável como complemento (SHAP obrigatório para tree models).

## 2. Contexto

O LAOS tinha lacuna em modelagem preditiva/ML: LATADE cobre data engineering (SQL, medallion), LADESIGN cobre visual, LAN8N cobre automação. Nenhuma cobria regressão, GLM, ordered/grouped logit, validação de pressupostos, efeitos marginais.

O usuário (economista Unicamp) declarou necessidade de evoluir em ML/data science, fornecendo Gujarati & Porter (2009) e o artigo da Quirk's (Larson & Goungetas 2013, NPS driver analysis) como base teórica.

## 3. Decisão inicial

- Naming: **laecon** (LA-ECON, econometria + ML). Não `laml` (genérico), não `lacausal` (restrito demais), não `lamod` (conflito com latade.modeling).
- Status: BASIC com 30 dias para STABLE (deadline 2026-07-04).
- 17 condições vinculantes do Conselho catalogadas.
- Local: `../laecon/`
- Tracking: `../LAOS/projects/_meta/capability-evolution/laecon.md`
- ADR: `../LAOS/projects/_meta/adr/ADR-002-laecon-creation.md`

## 4. Critérios de pronto

1. G1-G3 entregues: MCP server funcional, registry entry, routing.
2. G4 entregue: Constitution completa (10 artigos, 926 linhas).
3. G5 entregue: SDD scaffold completo.
4. G6: KB domain mínimo (index + 1 pattern NPS).
5. G7: delivery-reviewer sign-off.
6. Todos os 9 MCP tools funcionais em M1 (STABLE).
7. 17 condições vinculantes validadas.