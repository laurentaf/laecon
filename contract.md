# Contract — LAECON Capability

**Brief:** Criar a capability LAECON (../laecon) para o LAOS, oferecendo modelagem econométrica e machine learning interpretável, com likelihood explícita, validação de pressupostos, efeitos marginais, e suporte a casos de uso como análise de drivers de NPS (ordered/grouped logit). Capacita o usuário (economista Unicamp) a fazer modelagem preditiva e inferência causal diretamente em LAOS.

**Needs:** improvement, governance, investigation, research

**Deliverables:**
- Capability repo ../laecon/ com MCP server, Constitution, KB, pyproject, README
- Registry entries (capabilities.yaml + needs-to-capabilities.yaml)
- Tracking and ADR in projects/_meta/
- Preflight checker scripts/ e reforma do delivery-reviewer
- Knowledge entries (data-conventions.md, padroes-entrega.md updates)

**Capabilities used:** lacouncil (governance), context7/docs, exa/research, latade (data source)

**Repo:** ../laecon/ (github.com/laurentaf/laecon)

**Status:** BASIC → STABLE (+30d, deadline 2026-07-04)
**17 condições vinculantes:** 5 DA, 3 DD, 4 AE, 5 DR (detalhes em ../LAOS/projects/_meta/capability-evolution/laecon.md)