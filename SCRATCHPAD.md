# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-14

5-cycle validation runs completed. GCM and SUSTAIN converge correctly. RULEX fails — need to diagnose and fix.

### 5-cycle validation results

| Ground Truth | Winner | RMSE | 2nd Place | RMSE | Gap | Correct? |
|---|---|---|---|---|---|---|
| **GCM** | Exemplar_Agent | 0.342 | Rule_Agent | 0.403 | 15.1% | YES |
| **SUSTAIN** | Clustering_Agent | 0.344 | Rule_Agent | 0.507 | 32.2% | YES |
| **RULEX** | Clustering_Agent | 0.500 | Exemplar_Agent | 0.501 | 0.2% | NO |

### RULEX failure diagnosis

**Symptom:** All 3 agents converge to RMSE ~0.50 (random guessing). No separation after 5 cycles.

**Root causes identified:**
1. **Structure repetition** — 4/5 RULEX experiments used Type_VI, where RULEX is weakest (no simple rule exists). No diversity in structure selection.
2. **Clustering_Agent dominance** — Won 12/15 experiment selections across all 3 runs. Divergence-driven selection favors complex structures (Type_VI, five_four) which Clustering_Agent consistently proposes.
3. **Rule_Agent passivity** — Never won a single experiment selection across any run (0/15). Its proposals for simple structures (Type_I, Type_III) have lower divergence scores and lose.

**Proposed fix: Structure diversity penalty**
- Track which structures have been tested in prior cycles
- Penalize (or heavily discount) re-testing the same structure
- Forces exploration of RULEX-favorable structures (Type_I, Type_II, linear_separable)

### Other observations
- Critique quality: agents revise theories "progressively" but critiques don't prevent structure repetition
- RMSE gap widens for GCM (good) but collapses for RULEX (bad)
- Clustering_Agent dominates experiment selection regardless of ground truth

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
