# PLAN — {Feature Name} (SPEC-NNN)

**Status:** RASCUNHO
**SPEC parent:** SPEC-NNN
**ADRs-âncora:** ADR-NNN

---

## 1. Technical Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Storage | {tech} | ADR-NNN |
| Compute | {tech} | {reason} |

## 2. Pre-Implementation Gates

### Simplicity Gate (Article VII)
- [ ] Using ≤ 3 medallion layers?
- [ ] No speculative future-proofing?

### Anti-Abstraction Gate (Article VIII)
- [ ] Using DuckDB/Delta directly?
- [ ] No custom framework?

### Integration-First Gate (Article IX)
- [ ] Contracts between layers defined?
- [ ] HARNESS-NNN referenced?

## 3. Technical Decisions

### 3.1 {Decision}
- Decision: {what}
- Rationale: ADR-NNN

## 4. File Structure

```
src/{path}/
├── {file}.py
└── {file}_test.py
```

## 5. Test Plan

1. Contract tests
2. Integration tests
3. Unit tests

## 6. Complexity Tracking

{Empty if all gates pass.}