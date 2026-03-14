# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-14

Testing whether updated proposal prompt + concrete predictions fix RULEX validation. Run _07 in progress.

### Progress table (RULEX as ground truth)

| Run | Changes | Gap | Winner |
|---|---|---|---|
| _03 | round-robin | 16.7% | Exemplar (wrong) |
| _04 | divergence-driven | 5.5% | Exemplar (wrong) |
| _05 | + concrete preds | crashed (w_i param) | — |
| _06 | + param filter | 8.5% | Exemplar (wrong) |
| _07 | + updated prompt | pending | pending |

### What changed in each run:
- **_04**: Moderator picks highest-divergence structure from proposals
- **_05/06**: Agents see per-model accuracy in divergence ranking
- **_07**: Prompt explicitly tells agents to pick structures where their model has highest predicted accuracy

### Key observation:
Rule_Agent won cycle 0 of run _06 on `linear_separable_4d` (RMSE 0.312 vs 0.342). The system can identify the correct model on favorable structures. The challenge is getting enough favorable structures selected across 3 cycles.

### If _07 still fails:
- Consider whether this is a fundamental GCM flexibility issue — GCM is genuinely more flexible than RULEX on most structures
- May need 5+ cycles to accumulate enough evidence on RULEX-favorable structures
- Or need to investigate whether the LOO accuracy metric for RULEX is computed correctly (stochastic model with seed=42 may not represent expected behavior)

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
