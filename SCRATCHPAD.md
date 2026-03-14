# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-14

Implementing concrete model predictions in divergence ranking. The goal: agents see per-model predicted accuracies per structure so they can propose structures that favor their model.

### Context
- Divergence-driven selection (D12) works but agents don't propose strategically
- Rule_Agent proposes Type_II (lowest divergence) every cycle because it sounds interesting for rule learning
- The fix: show "GCM: 0.60, RULEX: 0.95, SUSTAIN: 0.55" per structure so agents know their advantage

### Implementation plan
1. Modify `compute_divergence_map()` to include per-model mean accuracy in each structure entry
2. Update `_divergence_context()` to show per-model predictions alongside divergence scores
3. TDD: tests first, then implementation

### Status of M3

| Ground Truth | Winner | Correct? |
|---|---|---|
| GCM | Exemplar_Agent (0.4334) | Yes |
| SUSTAIN | Clustering_Agent (0.4432) | Yes |
| RULEX | Exemplar_Agent (0.3872) | No — agents don't propose RULEX-favorable structures |

### Run inventory:
- `runs/True_GCM_LLM_gpt-4o_COLLAB_Exemplar-Rule-Clustering_03/` — GCM correct
- `runs/True_SUSTAIN_LLM_gpt-4o_COLLAB_Exemplar-Rule-Clustering_02/` — SUSTAIN correct
- `runs/True_RULEX_LLM_gpt-4o_COLLAB_Exemplar-Rule-Clustering_03/` — RULEX incorrect (round-robin)
- `runs/True_RULEX_LLM_gpt-4o_COLLAB_Exemplar-Rule-Clustering_04/` — RULEX incorrect (divergence-driven, gap narrowed)

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
