# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — 2026-03-14

Implemented debate-as-hypothesis-generator architecture (D19). Three-phase refactor:
- **Phase A:** Full-pool EIG over 55 candidates replaces agent proposals. Agents shift to interpreting results and generating hypotheses.
- **Phase B:** Learning curves as second evidence channel. GCM=gradual, RULEX=sudden, SUSTAIN=stepwise (on Type_I, all models are fast — patterns clearer on harder structures).
- **Phase C:** Novel structure generation. LLM agents can propose new structures during interpretation; validated and added to next cycle's EIG pool.

Legacy 9-phase flow preserved as `--mode legacy` (default). New flow: `--mode full_pool`.

**Next:** 5-cycle validation runs with `--mode full_pool` to compare convergence with legacy mode.

### 5-cycle validation results (pre-diversity-penalty)

| Ground Truth | Winner | RMSE | 2nd Place | RMSE | Gap | Correct? |
|---|---|---|---|---|---|---|
| **GCM** | Exemplar_Agent | 0.342 | Rule_Agent | 0.403 | 15.1% | YES |
| **SUSTAIN** | Clustering_Agent | 0.344 | Rule_Agent | 0.507 | 32.2% | YES |
| **RULEX** | Clustering_Agent | 0.500 | Exemplar_Agent | 0.501 | 0.2% | NO |

### Diversity penalty analysis

**What the heuristic does:**
- Two-tier penalty on previously-tested structures
- Exact structure+condition repeat: 2x decay per prior use
- Same structure, different condition: 1.5x decay per prior use
- Effect: forces exploration of diverse structures over 5 cycles

**What it fixes:** Structure repetition. Without it, Type_VI is selected 4/5 times. With it, 5 cycles should test ~5 different structures.

**What it does NOT fix — the GCM flexibility problem:**

Divergence map analysis shows RULEX and GCM have **low pairwise divergence** across all structures:

| Structure | GCM-RULEX divergence | RULEX best? |
|---|---|---|
| Type_I | 0.341 | Tied (both 1.0) |
| linear_separable_4d | 0.276 | Tied (~0.80) |
| linear_separable_2d | 0.229 | Tied (~0.65) |
| Type_III–V | 0.325 | Tied (~0.50) |
| Type_VI | 0.400 | No (SUSTAIN wins) |
| five_four | 0.331 | No (GCM wins) |

The high divergence on structures like linear_separable_2d (0.619) is between **RULEX vs SUSTAIN**, not RULEX vs GCM. GCM mimics RULEX by assigning high attention weights to the diagnostic dimension. This is a genuine property of GCM's flexibility — not a bug.

**Implication:** Even with perfect structure diversity, RULEX may not win because GCM approximates its predictions on every structure. The diversity penalty is necessary but likely insufficient. The Bayesian information-gain approach (TASKS.md) may help by selecting structures that maximize GCM-RULEX divergence specifically, but the fundamental issue is that these models are hard to distinguish with current category structures.

**Possible paths forward:**
1. Wait for RULEX re-run with diversity penalty — see if it helps at all
2. Consider whether RULEX-vs-GCM indistinguishability is a scientifically valid finding (it is, per Nosofsky 1991 — GCM can approximate many rule-like patterns)
3. Add structures specifically designed to differentiate GCM from RULEX (e.g., random categories where rules don't exist but attention helps)
4. Use learning curves instead of final accuracy — GCM predicts gradual learning, RULEX predicts sudden rule discovery

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
