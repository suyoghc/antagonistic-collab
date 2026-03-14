# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — M4: Analysis & write-up

M3 (validate convergence) is complete. Full_pool mode correct in all 3 conditions with learning curves solving the GCM-RULEX problem (2.4% → 68% gap).

**Available validation data (in `runs/`):**
- `validation_full_pool_GCM/` — 5 cycles, Exemplar wins (RMSE 0.161)
- `validation_full_pool_SUSTAIN/` — 5 cycles, Clustering wins (RMSE 0.270)
- `validation_full_pool_RULEX/` — 5 cycles, Rule wins (RMSE 0.119)
- `validation_legacy_GCM/` — 5 cycles, Exemplar wins (RMSE 0.255)
- `validation_legacy_SUSTAIN/` — 5 cycles, Clustering wins (RMSE 0.361)
- `validation_legacy_RULEX/` — 5 cycles, Rule wins (RMSE 0.429)

**M4 analysis questions:**
1. What structures/conditions does EIG select? Is there a pattern?
2. Do novel structures proposed by agents actually improve discrimination?
3. How quickly does the Bayesian posterior collapse? Monotonic?
4. Do interpretations/critiques improve over cycles or stay formulaic?
5. How much variance is there across replicate runs?

**Status:** Full_pool mode validated end-to-end (2-cycle real run with Princeton/GPT-4o). All integration gaps closed (D23): learning curves wired into execution + Bayesian update, novel structures validated + registered, curve context in interpretation debate, temporary structures in curve computation.

### 2-cycle full_pool validation (GCM ground truth)
- Cycle 0: EIG selected `five_four / fast_presentation` (highest EIG)
- Cycle 1: EIG selected `Type_I / low_attention` (different structure)
- Exemplar_Agent RMSE=0.139, Rule_Agent=0.352, Clustering_Agent=0.298
- Correct agent wins decisively

**Status update — 2026-03-14 evening:**
- 5-cycle comparative validation COMPLETE. Full_pool mode correct in all 3 conditions.
- Novel structure prompting COMPLETE (D24). Agents propose valid structures in every cycle.
- D25 crash fix: summary_for_agent on non-string predictions.
- All M3 tasks complete. 207 tests passing.

### 5-cycle comparative validation results

| Ground Truth | Mode | Winner | RMSE | 2nd | RMSE | Gap | Correct? |
|---|---|---|---|---|---|---|---|
| GCM | full_pool | Exemplar | 0.161 | Clustering | 0.242 | 34% | YES |
| GCM | legacy | Exemplar | 0.255 | Clustering | 0.404 | 37% | YES |
| SUSTAIN | full_pool | Clustering | 0.270 | Rule | 0.465 | 42% | YES |
| SUSTAIN | legacy | Clustering | 0.361 | Rule/Exemplar | 0.546 | 34% | YES |
| RULEX | full_pool | Rule | 0.119 | Clustering | 0.366 | 68% | YES |
| RULEX | legacy | Rule | 0.429 | Exemplar | 0.440 | 2.4% | YES |

**Key finding:** Learning curves solved the GCM-RULEX discrimination problem. Gap went from 2.4% to 68%.

**Next:**
- Update CHATLOG.md with session 13 summary
- Consider M4 goals: write-up, analysis of novel structures, cross-condition patterns

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
