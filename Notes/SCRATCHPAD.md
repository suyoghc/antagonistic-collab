# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — M6: ARBITER Integration — COMPLETE

### M6 implementation (all 5 features + 1 bugfix)

| Feature | Status | Commit | Tests |
|---|---|---|---|
| M6a: MetaAgentConfig | DONE | `3109d14` | 8 |
| M6b: Crux Negotiation (6 sub-commits) | DONE | `dfc6ed2`–`f49818c` | 32 |
| M6e: Conflict Map | DONE | `7fb5de3` | 6 |
| M6d: Pre-registration Output | DONE | `d7b8ca6` | 4 |
| M6c: HITL Checkpoints | DONE | `be91b7b` | 4 |
| Bugfix: dict new_predictions | DONE | `2a57937` | 2 |

Total: 56 new tests (231 → 287), 11 commits.

### M6 live validation (2026-03-15, GPT-4o via Princeton, all M6 features enabled)

| Ground Truth | Winner | RMSE | Gap% | Posterior | Cruxes | Claims | Time |
|---|---|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.1512 | 36.4% | 1.0000 | 4/34 | 14F/1C/26U | 431s |
| SUSTAIN | Clustering_Agent | 0.2700 | 45.6% | 1.0000 | 7/32 | 15F/0C/24U | 439s |
| RULEX | Rule_Agent | 0.1187 | 67.6% | 1.0000 | 4/35 | 15F/0C/26U | 467s |

3/3 correct. F=falsified, C=confirmed, U=untested.

### What the M6 validation reveals

**The system is a falsification engine.** 44 claims falsified, 1 confirmed, 76 untested. Convergence occurs by ruling out wrong theories, not by confirming the right one. Popper-compatible.

**Crux negotiation is genuinely selective with real LLMs.** 15% acceptance rate (vs 100% in mock). Accepted cruxes cluster around real theoretical fault lines: exemplars vs rules, presentation order effects, attention allocation.

**Posterior collapse is the main bottleneck.** GCM and SUSTAIN lock to correct model on cycle 0. RULEX takes until cycle 2. After that, EIG≈0 and remaining cycles are uninformative. Crux boost can't overcome zero EIG.

**Winning theories need fewer revisions.** Rule_Agent made 0 revisions and won RULEX by 67.6%. Clustering_Agent made 3 futile revisions in the same run. Lakatos-compatible: robust cores resist falsification.

**RULEX is the most scientifically interesting case.** Non-monotonic posterior trajectory — system initially backs Exemplar_Agent, self-corrects by cycle 2 when Type_I structures disambiguate. GCM and SUSTAIN converge immediately.

### Next steps
- Address posterior collapse: tempering, entropy-based re-exploration, or multi-hypothesis tracking
- New cognitive domains (memory retrieval, decision making)
- AutoRA integration for real data
- Longer runs (10+ cycles) to test cumulative reasoning
- Claim-responsive debate: agents should address prior falsified claims

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
