# Current State (M14)

## Scientific conclusion

The computational pipeline (Bayesian EIG + model predictions + learning curves) is
**causally sufficient** to identify the correct model on synthetic benchmarks with
fully-specified models. Debate adds interpretive value but does not improve
identification accuracy.

This was established by M13 ablation: 18/18 correct across 3 debate conditions ×
3 ground truths × 2 selection strategies. No-debate achieved the best RMSE (0.055)
and gap (87.6%) while running 3-4x faster.

**Open question:** Does debate help when models are *misspecified*? (See [ROADMAP.md](ROADMAP.md))

## M14 results (GPT-4o, 5 cycles, full debate + computation feedback loop)

| Ground Truth | Winner | RMSE | Gap | Claims | Resolved |
|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.087 | 78% | 44 | 15 |
| SUSTAIN | Clustering_Agent | 0.061 | 88% | 36 | 15 |
| RULEX | Rule_Agent | 0.233 | 39% | 40 | 15 |

All 3 feedback interventions fire: fuzzy structure matching (94 matches),
claim-directed experiment selection (12 firings), parameter validation (7/33
rejected), claim auto-resolution (45 resolved, 39 falsified vs 6 confirmed).

## Key findings across milestones

- **Learning curves are the key discriminator.** RULEX gap: 2.4% (item-level only, M5) → 68% (with learning curves, M6). GCM approximates RULEX's final accuracy but not its sudden learning dynamics.
- **LLM-agnostic.** Correct model wins in 9/9 cross-LLM runs: GPT-4o, Claude Sonnet, Claude Opus (M4).
- **Falsification-dominated.** 39/45 claims falsified vs 6 confirmed (M14). The system converges by ruling out wrong theories, not confirming the right one.
- **Param validation is the strongest M14 intervention.** Blocked 21% of proposed revisions. Prevents RMSE degradation of 0.02–0.10 per blocked revision.

## What's next

See [ROADMAP.md](ROADMAP.md) — M15 (model misspecification), M16 (open design space).

## Full history

See [Notes/archive/INDEX.md](Notes/archive/INDEX.md) for development docs across 14 milestones.
