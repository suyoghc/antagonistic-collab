# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — M4: Analysis & write-up

### Analysis status (all complete)

| Analysis | Status | Location |
|---|---|---|
| EIG selection patterns | DONE | LESSONS 9.1 |
| Posterior convergence speed | DONE | LESSONS 9.2 |
| Novel structure generation | DONE | LESSONS 9.3 |
| Legacy vs full_pool comparison | DONE | LESSONS 9.4 |
| Theory revision patterns | DONE | LESSONS 9.5 |
| Interpretation debate quality audit | DONE | LESSONS 9.6 |

### Remaining M4 tasks

1. **Replication runs** — run each condition 3× (full_pool mode, 5 cycles) to get variance estimates on RMSE gaps. Requires API credits.
2. **Write-up** — structured report (intro, methods, results, discussion). Can draft from existing single-run data; add replication CIs later.

### 10 key findings (for write-up)

1. Learning curves solved the GCM-RULEX discrimination problem (2.4% → 68% gap)
2. Bayesian EIG beats LLM proposals for experiment selection
3. LLMs add value for interpretation, not selection (architecture thesis)
4. Posterior convergence is fast but not instant for hard model pairs (RULEX: 2-cycle lag)
5. Novel structures are narratively interesting but statistically unselected (0/21 chosen by EIG)
6. Interpretation debate quality is weak (posteriors as proxy, no cumulative learning)
7. Theory revision follows Lakatos-compatible patterns (correct stable, incorrect revise progressively)
8. GCM's flexibility is genuine, not a bug (approximates rule-like behavior via attention weights)
9. Defensive type coercion essential for LLM-in-the-loop systems (3 crashes from same pattern)
10. The specification gap is the fundamental bottleneck (LLMs can't close concept→code loop)

### Validation data (in `runs/`)

| Run | Winner | RMSE | Gap |
|---|---|---|---|
| validation_full_pool_GCM | Exemplar | 0.161 | 34% |
| validation_full_pool_SUSTAIN | Clustering | 0.270 | 42% |
| validation_full_pool_RULEX | Rule | 0.119 | 68% |
| validation_legacy_GCM | Exemplar | 0.255 | 37% |
| validation_legacy_SUSTAIN | Clustering | 0.361 | 34% |
| validation_legacy_RULEX | Rule | 0.429 | 2.4% |

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
