# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — M5: Close Debate Feedback Loops — COMPLETE

### M5 implementation (all 4 features done)

| Feature | Status | Commit | Tests |
|---|---|---|---|
| 7.1 Parameter revision persistence | DONE | `1d12fde` | 6 |
| 7.4 Structured claim ledger | DONE | `4625d53` | 8 |
| 7.2 Critique-as-falsification | DONE | `84852bb` | 5 |
| 7.3 Debate-informed EIG weighting | DONE | `f61eec4` | 5 |

### M5 validation (2026-03-15, GPT-4o via Princeton)

| Ground Truth | Winner | RMSE | Posterior | Correct? |
|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.1836 | 1.0000 | ✓ |
| SUSTAIN | Clustering_Agent | 0.2687 | 1.0000 | ✓ |
| RULEX | Rule_Agent | 0.1580 | 1.0000 | ✓ |

### Replication variance (4× GCM runs)

| Run | Exemplar RMSE | Clustering RMSE | Rule RMSE |
|---|---|---|---|
| Initial | 0.1836 | 0.2123 | 0.3280 |
| Rep 1 | 0.1587 | 0.2424 | 0.3379 |
| Rep 2 | 0.1832 | 0.2571 | 0.3628 |
| Rep 3 | 0.2082 | 0.2590 | 0.3558 |
| **Std Dev** | **0.0177** | **0.0189** | **0.0153** |

**Key result:** RMSE variance is now non-zero (was 0.000 pre-M5). Debate causally affects outcomes through parameter revision persistence.

### M5 feature activity
- ~45 FALSE CLAIMs detected across 6 runs (critique-as-falsification)
- 1 verified claim (Rule_Agent predicted 0.600, actual 0.544)
- Agents consistently overclaim model accuracy (predicting 0.65–0.90 when actual 0.10–0.48)
- RULEX posterior trajectory: started P(Exemplar)=1.0 for cycles 0-1, flipped to P(Rule)=0.9998 at cycle 2

### Key reference
- **12 Theses on LLM-Mediated Scientific Debate** — synthesis of all findings, in LESSONS_LEARNED.md "Synthesis" section

### Next steps
- ARBITER/CRUCIBLE integration (M6a-M6e roadmap exists)
- New domain extensions
- Cross-LLM replication with M5 features (compare variance across GPT-4o/Sonnet/Opus)

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
