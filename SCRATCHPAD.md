# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-13

M3 multi-model validation complete (2 of 3 pass). Key finding: the debate doesn't influence quantitative outcomes yet — round-robin selection + model-computed predictions means the LLM debate is cosmetic.

### Status of M3

| Ground Truth | Winner | Correct? |
|---|---|---|
| GCM | Exemplar_Agent (0.4334) | Yes |
| SUSTAIN | Clustering_Agent (0.4432) | Yes |
| RULEX | Exemplar_Agent (0.4417) | No — structure selection unfavorable |

### Key insight: the debate needs teeth

The adversarial debate machinery (critique, interpretation, theory revision) generates interesting text but has zero causal connection to quantitative scoring. For it to matter:

1. **Divergence-driven experiment selection** — replace round-robin with a moderator that picks the most discriminating experiment
2. **Phase 5 implementation** — critiques should revise proposals before execution
3. **LLM moderator** — evaluate proposals+critiques to select experiments that resolve open disputes

### Where to pick up next session:
1. **Make debate influence outcomes** — implement divergence-driven experiment selection (highest priority, unblocks RULEX validation)
2. **Longer debates (5+ cycles)** — with better selection, check RMSE gap growth
3. **Critique quality assessment** — check the "my model can also predict that" pattern
4. **Remaining Codex items** — P1 (Phase 5), P2 (reject path), P3 (--demo flag)

### Run inventory (post-LOO):
- `runs/True_GCM_LLM_gpt-4o_COLLAB_Exemplar-Rule-Clustering_03/` — GCM correct
- `runs/True_SUSTAIN_LLM_gpt-4o_COLLAB_Exemplar-Rule-Clustering_02/` — SUSTAIN correct
- `runs/True_RULEX_LLM_gpt-4o_COLLAB_Exemplar-Rule-Clustering_03/` — RULEX incorrect

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
