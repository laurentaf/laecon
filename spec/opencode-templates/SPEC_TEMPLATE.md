# SPEC-NNN: {Component Name}

**Status:** RASCUNHO
**Version:** 1.0
**Authors:** {name} (role)
**ADRs-âncora:** ADR-NNN
**GSD referenced:** §X.Y
**HARNESS:** HARNESS-NNN levels N
**Owner:** {name}

---

## 1. Executive Summary

{In 1 paragraph: what this component delivers, for whom, what business problem it solves.}

## 2. User Stories

### US-1
As a {role}, I need {result} because {reason}.

## 3. Acceptance Criteria

- [ ] Criterion 1 is verifiable
- [ ] Criterion 2 is measurable
- [ ] [NEEDS CLARIFICATION: question]

## 4. Sources (Inputs)

| Table | Expected Schema | Filter |
|-------|-----------------|--------|
| `source_table` | as per SPEC-NNN | `condition` |

## 5. Destination (Output)

### 5.1 Location
- Catalog: `{catalog}`
- Schema: `{schema}`
- Table: `{table_name}`

### 5.2 DDL
```sql
CREATE TABLE IF NOT EXISTS {full_table_name} (
    {columns}
);
```

## 6. Business Rules

### R-LOCAL-1 — {Rule Name}
{Rule description referencing GSD §X.Y}

## 7. Refresh Strategy

- Mode: {full overwrite|delete-insert|merge|append}
- Reprocessable: {yes/no}

## 8. Operational Window

- Frequency: {daily|weekly|monthly} at {time}
- Upstream dependency: {component} ready by {time}
- Output SLA: {time}

## 9. Validations

### 9.1 Pre-execution
- [ ] {check}

### 9.2 Post-execution
- [ ] {check}
- [ ] HARNESS-NNN level N returns OK

## 10. Edge Cases

- {edge case 1}
- [NEEDS CLARIFICATION: {question}]

## 11. Code Path

- File: `src/{path}/{file}.py`