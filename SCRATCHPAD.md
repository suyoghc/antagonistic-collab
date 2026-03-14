# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-14

Divergence-driven experiment selection implemented (D12). RULEX re-validation running in background.

### Status of M3

| Ground Truth | Winner | Correct? |
|---|---|---|
| GCM | Exemplar_Agent (0.4334) | Yes |
| SUSTAIN | Clustering_Agent (0.4432) | Yes |
| RULEX | Exemplar_Agent (0.4417) | No — round-robin, **re-running with divergence-driven** |

### What changed (D12):
- Batch-mode arbitration now picks proposal with highest structure divergence
- `compute_divergence_map()` uses all 11 STRUCTURE_REGISTRY structures
- Falls back to critique count on ties
- 115 tests passing, ruff clean

### Awaiting:
- RULEX re-validation result (`runs/True_RULEX_LLM_gpt-4o_COLLAB_Exemplar-Rule-Clustering_04/`)

### Next after RULEX result:
1. **Longer debates (5+ cycles)** — with better selection, check RMSE gap growth
2. **Critique quality assessment** — check the "my model can also predict that" pattern
3. **Remaining Codex items** — P1 (Phase 5), P2 (reject path), P3 (--demo flag)
4. **Show concrete predictions in divergence ranking** — agent-facing display of what divergence means

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
