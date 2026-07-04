# HARNESS-NNN: {Reconciliation Name}

**Status:** RASCUNHO
**Systems:** {pipeline} vs {baseline}
**Owner:** {name}

---

## 1. Objective

Prove pipeline X produces numbers equivalent to source Y within tolerances.

## 2. Baseline

- Source: {system}
- Table: `{table}`
- Load: {frequency}

## 3. Levels

| Level | Metric | Tolerance | Frequency |
|------:|--------|----------:|-----------|
| 1 | sum(metric) | ±0.5% | every run |
| 2 | sum(metric) by group | ±1.0% | weekly |
| 3 | sample rows | visual | monthly |

## 4. Execution Table

```sql
CREATE TABLE harness.reconciliation_log (
    execution_date TIMESTAMP,
    metric STRING,
    value_pipeline DECIMAL(18,4),
    value_baseline DECIMAL(18,4),
    diff_pct DECIMAL(7,4),
    status STRING
);
```

## 5. Treatment

| Status | Action |
|--------|--------|
| OK | Proceed |
| ALERT | Investigate |
| CRITICAL | Block |