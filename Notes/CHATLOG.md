# Chat Log

Human-readable summary of each Claude Code session on this project.

---

## Session 1 — 2026-03-11 ~23:00

**Transcript:** `4b9a651e-e17f-4b50-84b7-820aacf6103e.jsonl`
**Commits:** `5e3d726`, `7d2e854`

**What we did:**
- Initial code review of the antagonistic collaboration framework
- Fixed first round of bugs: critique provenance (sprayed to all proposals), category structure KeyError fallback, off-by-one in transcript filenames, non-deterministic divergence maps, 0.0 RMSE display, format crash on non-numeric accuracy, packaging issues
- Fixed packaging layout (moved source into `antagonistic_collab/` subdirectory), replaced regex JSON parser with brace-depth parser, guarded `--cycles 0` edge case

**Key discussion:**
- Identified that the codebase had multiple crash-level bugs preventing any successful run
- Decided to fix bugs bottom-up before attempting to run debates

---

## Session 2 — 2026-03-12 ~12:00–14:30

**Transcript:** `425a9658-4c99-45f8-9fe5-358a346504a8.jsonl`
**Commits:** `e41ae99`, `f4198a5`

**What we did:**
- Fixed numpy serialization (int64 keys broke JSON)
- Rewrote tests to exercise production code paths instead of mocking internals
- Fixed 16 additional bugs across 5 modules: NaN in correlations, missing dict keys, shared-reference mutation in theory revisions, empty API responses, negative critique indices, EOFError on stdin
- Cleaned all ruff lint warnings

**Key discussion:**
- Shifted from "make it not crash" to "make data integrity correct"
- Tests expanded from basic to 57 passing

---

## Session 3 — 2026-03-12 ~16:00–16:36

**Transcript:** `ae8634fd-2da7-42fe-afc4-f201b765a573.jsonl`
**Commits:** `296112f`

**What we did:**
- Added Princeton AI Sandbox as alternative LLM backend (`--backend princeton`)
- Initial implementation used Azure OpenAI SDK
- 7 regression tests for both Anthropic and OpenAI code paths

**Key discussion:**
- Needed GPT-4o access through Princeton's compute allocation
- Decided on a `--backend` CLI flag rather than auto-detection

---

## Session 4 — 2026-03-12 ~20:00–22:28

**Transcript:** `c6ea8aa1-d5a0-4e8d-9efc-7ac7fd8f960e.jsonl`
**Commits:** `2624d9f`, `be4f44b`, `8f837d6`

**What we did:**
- Fixed Princeton backend: switched from AzureOpenAI to Portkey gateway (api.portkey.ai)
- Added Markdown reports: per-cycle `.md` transcripts + `summary.md` with leaderboard and theory trajectories
- Added auto-naming for output dirs (`runs/True_{model}_LLM_{llm}_COLLAB_{agents}_{NN}/`)
- Fixed cumulative transcript bug (each cycle's JSON was accumulating all prior messages)
- Fixed duplicate JSON in cycle markdown
- Fixed empty prediction leaderboard (agents weren't using `mean_accuracy` key)
- Ran first batch debates (4 runs, `_01` through `_04`), discovered batch-mode bias
- Planned the moderator rotation fix (D5)

**Key discussion:**
- First successful end-to-end 3-cycle debate (`_04`)
- Discovered that all 3 experiments were proposed by Exemplar_Agent — batch mode always picks `approve 0`
- Analyzed the run and designed the round-robin + critique tiebreaker strategy

---

## Session 5 — 2026-03-12 22:28 → 2026-03-13 ~07:16 (current)

**Transcript:** `dea55001-6188-40c3-bf20-d30fdae94c19.jsonl`
**Commits:** `c193d0a`

**What we did:**
- Implemented batch-mode rotation fix (TDD: 3 failing tests → implementation → all 73 pass)
- Deleted buggy runs from prior sessions
- Saved Princeton API key to `.env`
- Re-ran 3-cycle debate with rotation fix — confirmed each agent gets one experiment per cycle
- Analyzed convergence: discovered that all experiments return `mean_accuracy = 0.550` regardless of design
- Diagnosed root cause: LLM designs aren't parsed into stimuli/labels, generator always falls back to Shepard Type II with fixed params/seed
- Discussed 4 solution options (A: structure library, B: parse LLM designs, C: param variation, D: pre-computed menu)
- Created DECISIONS.md, TASKS.md, CHATLOG.md to establish project documentation

**Key discussion:**
- The debate machinery works (rotation, critique, theory revision) but produces no signal because data never varies
- Leaning toward Option A+C (structure library + condition-to-param mapping) as next fix
- This is the main blocker for meaningful convergence

---

## Session 6 — 2026-03-13

**What we did:**
- Implemented Phase 2: model-sensitive synthetic data (D7)
- Added `STRUCTURE_REGISTRY` (11 structures) and `CONDITION_EFFECTS` (5 conditions) to `debate_protocol.py`
- Rewrote `_synthetic_runner()`: looks up structure by name, applies condition overrides, uses md5-based per-experiment seeds, returns metadata
- Fixed scoring filter in `runner.py`: item_accuracies merged into actual dict for per-item scoring
- Updated experiment proposal prompt with structure/condition menus
- Updated prediction prompt with item-level guidance
- Enhanced `_divergence_context()` with ranked structure list
- 9 new TDD regression tests, 82 total passing, ruff clean
- Ran 3-cycle validation debate (Princeton/GPT-4o): data now varies (0.666, 0.473, 0.651 across cycles)
- Discovered next blocker: Clustering_Agent beats Exemplar_Agent despite GCM being ground truth — agents guess predictions instead of running models (D8)
- Logged findings to LESSONS_LEARNED.md Phase 2 (2.1–2.3)

**Key discussion:**
- Phase 2 data pipeline fix works: different structures/conditions/cycles produce different data
- But prediction scoring is confounded by LLM calibration — agents need to call `model.predict()` for predictions
- Agents also underuse divergence ranking (nobody picked 5-4 despite highest divergence)
- Next: agents call their models for predictions, then re-validate with 4-cycle run

---

## Session 7 — 2026-03-13

**Commits:** `dba1489`, `7b0b787`

**What we did:**
- Implemented Phase 3: model-computed predictions (D8)
- Added `compute_model_predictions()` to `DebateProtocol` — runs agent's model on structure with condition overrides, returns per-item P(correct label)
- Modified `run_execution()`: LLM provides reasoning+confidence, system computes predictions via `model.predict()`
- Fixed P1 bug from Codex review: `param_overrides` from LLM response were ignored and `default_params` recorded instead of actual params
- Queued 4 remaining Codex findings (P1–P3) in TASKS.md
- Ran 3-cycle validation debate (Princeton/GPT-4o, `--true-model GCM`)
- **Result: Exemplar_Agent wins with RMSE=0.0776, 3.6x lower than Rule_Agent (0.2755) and 4.5x lower than Clustering_Agent (0.3528)**
- Logged findings to LESSONS_LEARNED.md Phase 3 (3.1–3.4)
- 91 tests passing, ruff clean

**Key discussion:**
- Phase 3 fixes the core blocker from Phase 2: RMSE now measures model fit, not LLM calibration
- The correct agent wins decisively — monotonic separation across all 3 cycles
- LLM reasoning is still valuable for interpretation; the fix was separating semantics (LLM) from numerics (model)
- 5-4 structure still never selected despite highest divergence ranking
- Milestone M3 core hypothesis confirmed; ready for multi-model validation (SUSTAIN, RULEX as ground truth)
- Reviewed 5 findings from Codex review: fixed P1 param_overrides bug, queued remaining 4 (P1–P3)
- Created FEATURES.md cataloging 23 scientifically meaningful design choices across 6 categories
- Discussed implications of agents ignoring divergence ranking; queued concrete prediction display feature

---

## Session 8 — 2026-03-13

**Commits:** `8927fb9`, `b11e6d9`, `dbcf83e`

**What we did:**
- Comprehensive code review across all 6 modules → fixed 12 bugs in 3 rounds (D9, D10, D11)
  - **D9 (8 bugs):** Format crash, design_spec validation, GCM r=0 and bias, SUSTAIN zero-lambdas, to_json dirs, NaN guard, condition warning
  - **D10 (2 bugs):** SUSTAIN predict_learning_curve uses test items, call_agent retry with backoff
  - **D11:** Leave-one-out cross-validation in compute_model_predictions and compute_divergence_map
- Ran multi-model validation (pre-LOO): SUSTAIN won all 3 conditions — diagnosed self-prediction bias
- Implemented LOO fix, re-ran all 3 validations:
  - GCM → Exemplar_Agent wins (0.4334) ✓
  - SUSTAIN → Clustering_Agent wins (0.4432) ✓
  - RULEX → Exemplar_Agent wins (0.4417), Rule_Agent 2nd (0.5153) ✗
- Documented Phase 4 findings in LESSONS_LEARNED.md (4.1–4.5)
- 21 new regression tests (112 total), ruff clean

**Key findings:**
- Self-prediction bias: GCM matching item to itself (distance=0) produced over-confident predictions that couldn't match noisy data. LOO fixes this.
- RULEX failure: round-robin selects Type_VI (worst structure for RULEX) in cycle 0 every time. Need divergence-driven selection.
- **The debate doesn't influence outcomes yet** — predictions are model-computed, selection is round-robin, critiques don't revise proposals. The LLM debate is currently cosmetic. Making it matter is the next architectural challenge.

---

## Session 9 — 2026-03-14

**Commits:** `19e8756`, `e8e9ce2`, `5e66a70`, `1677fec`, `1d4b903`, `1029ddb`, `7ade636`

**What we did:**
- Fixed 2 remaining test failures from divergence-driven selection (D12), merged to main
- Implemented concrete model predictions in divergence ranking — agents now see per-model predicted accuracy per structure
- Discovered and fixed crash: LLM agents propose param_overrides with invented parameter names (e.g., `w_i`). Added `inspect.signature` filtering.
- Updated proposal prompt to direct agents toward structures where their model has highest accuracy
- Ran 4 RULEX validation runs (runs _04 through _07) with progressive improvements
- Documented Phase 5 findings in LESSONS_LEARNED.md (5.1–5.4)
- 122 tests passing, ruff clean

**Key findings:**
- Divergence-driven selection works: picks higher-divergence structures (0.444–0.619 vs arbitrary)
- Concrete predictions + updated prompt changed agent behavior: Rule_Agent proposed Type_I for first time
- RULEX gap narrowed from 16.7% to 4.7% across 4 runs, but GCM still wins — appears to be a genuine model flexibility difference
- This may be a real scientific finding: GCM is more parsimonious even for rule-generated data

**Key discussion:**
- User asked whether to use Della — not needed, compute is on API side
- Central insight: information design for LLM agents matters as much as algorithm design. Showing the right numbers in the right format changes behavior.

---

## Session 10 — 2026-03-14

**Commits:** `d499221`, `c69a5e2`, `16e34ea`, `6b27a21`, `e94c4db`, `d148d2b`

**What we did:**
- Implemented Phase 5 (Design Revision) — agents now revise proposals based on critiques (D15)
- Fixed moderator reject path (P2, D16) — rejection loops back to proposal→critique→revision→arbitration, capped at 3 attempts
- Fixed --demo flag order-sensitivity (P3) — `"--demo" in sys.argv` works regardless of position
- Implemented two-tier structure diversity penalty (D17) — exact repeats: 2x decay, same-structure-different-condition: 1.5x decay
- Ran 5-cycle validation debates for all 3 ground truth models:
  - GCM → Exemplar_Agent wins (15.1% gap) ✓
  - SUSTAIN → Clustering_Agent wins (32.2% gap) ✓
  - RULEX → Rule_Agent wins after diversity fix (1.8% gap) ✓
- Investigated GCM flexibility confound — concluded it's a genuine scientific finding (Nosofsky 1991), not a system bug
- Assessed critique quality over 5 cycles — critiques remain substantive but formulaic; don't drive structure diversity
- Documented Phase 6 findings in LESSONS_LEARNED.md (6.1–6.5)
- 146 tests passing, ruff clean

**Key findings:**
- Structure repetition pathology: without diversity penalty, Type_VI selected 4/5 cycles, causing all agents to converge to ~0.50 RMSE
- GCM-RULEX divergence is inherently low (0.16–0.40) across all 11 structures — GCM mimics rule-like behavior via attention weights
- The 1.8% RULEX gap vs 15–32% for GCM/SUSTAIN reflects genuine model flexibility asymmetry, not a system failure
- Critique quality: agents revise theories progressively but critiques don't influence moderator's structure selection

**Key discussion:**
- User directed: refine heuristic now, log Bayesian information-gain approach as preferred long-term replacement in TASKS.md
- Diversity penalty is necessary but insufficient for hard model pairs (GCM vs RULEX)

---

## Session 11 — 2026-03-14

**Commits:** `1242b29`, `32e1284`, `68bc770`

**What we did:**
- Implemented the debate-as-hypothesis-generator architecture (D19) in three TDD phases:
  - **Phase A (12 tests):** Full-pool EIG selection over 55 candidates, interpretation debate with structured JSON output (hypothesis, confounds, novel structures), interpretation critique, `--mode full_pool|legacy` flag
  - **Phase B (9 tests):** Learning curves as second evidence channel — `compute_learning_curve_predictions()`, `extract_curve_features()` (gradual/sudden/stepwise classification), curve evidence in Bayesian posterior updates
  - **Phase C (10 tests):** Novel structure generation — `validate_novel_structure()`, `temporary_structures` dict, integration with EIG pool
- All 189 tests passing, ruff clean at each phase
- Updated TASKS.md, DECISIONS.md, SCRATCHPAD.md

**Key findings:**
- On simple structures (Type_I), all models learn too fast for gradual/sudden/stepwise patterns to cleanly differentiate — harder structures needed for curve-based discrimination
- Full-pool EIG eliminates LLM calls from experiment selection, shifting agents to where they add real value: interpreting results and generating hypotheses
- Novel structure validation ensures LLM-proposed structures are executable (4-32 items, ≤8 dims, ≥2 categories)

**Key discussion:**
- Architecture redesign motivated by D18 (EIG already does selection better than LLMs) and D17 (1.8% GCM-RULEX gap needs a second evidence channel)
- Legacy flow preserved as default for backward compatibility

---

## Session 12 — 2026-03-14

**Commits:** `27d4809`, `06c2f8f`, `6c32451`, `908cb36`, `b9837a3`, `d8841fd`, `6adcdf1`

**What we did:**
- Fixed full_pool mode phase desync bug (D20): `advance_phase()` uses `current_phase` for transitions, but after divergence mapping current_phase was `EXPERIMENT_PROPOSAL`, not `HUMAN_ARBITRATION`. Fix: `skip_to_phase(HUMAN_ARBITRATION)` before advancing from EIG selection result.
- Added `TestFullPoolIntegration`: 2-cycle end-to-end integration test with mocked LLM
- Validated with real 2-cycle Princeton/GPT-4o run (`--mode full_pool --true-model GCM`):
  - Cycle 0: EIG selected `five_four / fast_presentation`
  - Cycle 1: EIG selected `Type_I / low_attention`
  - Exemplar_Agent wins (RMSE 0.139 vs 0.352 Rule, 0.298 Clustering) ✓
- Documented Phase 7 findings in LESSONS_LEARNED.md (7.1–7.4)
- Fixed 8 bugs from two Codex review rounds:
  - **D21 (4 bugs):** Override fallback drops condition params, scalar addresses_critiques crash, invalid approval escapes retry loop, SUSTAIN partial block duplicate label
  - **D22 (4 fixes):** RULEX ground truth non-deterministic (missing seed=42), RULEX predict_learning_curve drops seed, redundant model_predicted/predicted keys, .gitignore gap for .md transcripts
- 198 tests passing, ruff clean

**Key findings:**
- Full_pool mode works end-to-end with 36% fewer LLM calls per cycle than legacy mode
- EIG is naturally self-diversifying through posterior updates — no heuristic penalty needed
- Phase state machine desyncs are subtle integration bugs that unit tests miss; need integration tests
- RULEX non-determinism was a significant correctness issue — same experiment could produce different ground truth between runs

**Key discussion:**
- Codex review dismissed: module-level globals (valid but large refactor), code in `__init__.py` (standard re-exports). Deferred: GCM.fit() self-prediction bias (latent, unused in pipeline).
- Next steps: wire learning curves into execution, feed novel structures back into EIG pool, 5-cycle comparative validation

---

## Session 13 — 2026-03-14

**Commits:** `b663fc1`, `e4c05a2`, `03fd898`, `6958557`, `65dbc98`, `5e37ac1`, `f3fd0d7`

**What we did:**
- Implemented D23: wired learning curves into `run_execution()` (compute + Bayesian update), novel structures into `run_interpretation_debate()` (validate + register), curve context into interpretation prompts, temporary structures into `compute_learning_curve_predictions()`
- Implemented D24: few-shot novel structure examples in interpretation debate prompt (constraints, example, strategic guidance)
- Fixed D25: `summary_for_agent()` crash on non-string `new_predictions` — coerce to `str()` before joining
- Ran 5-cycle comparative validation: 6 runs (3 ground truths × 2 modes), all correct winners
- Updated LESSONS_LEARNED.md Phase 8 (8.1–8.4), DECISIONS.md (D23–D25), TASKS.md, SCRATCHPAD.md, PLANNING.md
- 9 new regression tests (207 total), ruff clean

**Key findings:**

| Ground Truth | full_pool Gap | legacy Gap |
|---|---|---|
| GCM | 34% | 37% |
| SUSTAIN | 42% | 34% |
| RULEX | **68%** | **2.4%** |

- **Learning curves solved the GCM-RULEX discrimination problem** (2.4% → 68% gap)
- Full_pool mode correct in all 3 conditions with consistently larger RMSE gaps
- Novel structures proposed by agents in every cycle with few-shot prompting
- D25 crash: same pattern as D21 — LLM outputs have unpredictable types, defensive coercion needed

**Key discussion:**
- M3 milestone complete: all convergence validation tasks done
- Architecture thesis confirmed: LLMs add value for interpretation/hypothesis generation, not experiment selection
- Multiple evidence channels (accuracy + curve shape) beat single-channel for model comparison
- Ready for M4: analysis, write-up, cross-condition patterns

---

## Session 14 — 2026-03-14

**Commits:** `1c31f37`

**What we did:**
- Completed interpretation debate quality audit (LESSONS 9.6): audited 30 debate cycles across 6 validation runs
- Compiled 10 key findings across the entire project
- Cleaned SCRATCHPAD.md (removed resolved M3 working notes, added findings summary for write-up)
- Updated all tracking files

**Key findings from quality audit (9.6):**
- Data citation: WEAK — agents cite posteriors as proxy, rarely reference item-level data
- Critique quality: MIXED — mechanism-aware but numerically ungrounded
- Behavioral adaptation: LIMITED — same 2-3 talking points repeat across all 5 cycles
- Novel structure rationale: POOR — not rooted in actual model divergence
- Adversarial critique forcing function helps in later cycles (only evidence of improvement)

**Write-up:** Drafted REPORT.md — structured report with abstract, intro, methods, results (3.1–3.9), discussion (4.1–4.6), conclusion, references.

**Replication runs:** Ran 3× each condition (full_pool mode). Key finding: RMSE values identical across all replicates (zero variance). Pipeline fully deterministic — EIG selection, synthetic data, model predictions all seeded. LLM debate varies but is epiphenomenal to RMSE (LESSONS 9.7). Updated REPORT.md Section 3.5 and limitations.

**Status:** All M4 tasks complete. Milestone M4 done.

---

## Session 15 — 2026-03-15

**Commits:** `1c31f37`, `e4d9484`, `07856d7`, `e2cc0e9`, `2736ac0`, `26fcb3c`, `d6cde99`, `6076c20`, `3fd2be2`

**What we did:**
- Completed interpretation debate quality audit (LESSONS 9.6)
- Drafted REPORT.md (structured write-up with abstract, intro, methods, results 3.1–3.10, discussion, conclusion)
- Ran replication runs: discovered zero variance (LESSONS 9.7)
- Cross-LLM comparison: GPT-4o vs Claude Sonnet vs Claude Opus — 9 runs, all correct (D26, LESSONS 9.8)
- Moved docs to Notes/, removed CLAUDE.md from repo, updated README.md
- Analyzed debate feedback loops: 4 of 6 broken (parameter revisions, param_overrides, hypotheses, critique content)
- Removed contributor names pending confirmation

**Cross-LLM results (LESSONS 9.8):**

| Ground Truth | GPT-4o | Sonnet | Opus |
|---|---|---|---|
| GCM | Exemplar (0.159) | Exemplar (0.159) | Exemplar (0.143) |
| SUSTAIN | Clustering (0.270) | Clustering (0.270) | Clustering (0.270) |
| RULEX | Rule (0.158) | Rule (0.148) | Rule (0.213) |

Correct model wins in 9/9 runs. Framework is LLM-agnostic.

**Key discussion:**
- Debate is epiphenomenal to RMSE — convergence driven entirely by Bayesian machinery
- 4 broken feedback loops identified (parameter persistence, param_overrides, hypotheses, critique content)
- Agent isolation analysis: epistemic coupling without rhetorical coupling
- param_overrides is the one surviving LLM→RMSE path, explaining small cross-LLM variation

**Status:** M4 complete. Next: close debate feedback loops (parameter persistence as priority #1).

---

## Session 16 — 2026-03-15

**Commits:** `1d12fde`, `4625d53`, `84852bb`, `f61eec4`

**What we did:**
- Implemented all 4 M5 features (close debate feedback loops):
  - 7.1: Parameter revision persistence (`sync_params_from_theory()`)
  - 7.4: Structured claim ledger (`DebateClaim` dataclass + ledger methods)
  - 7.2: Critique-as-falsification (`verify_prediction_claim()`)
  - 7.3: Debate-informed EIG weighting (focus pair boosting in `select_from_pool()`)
- 24 new tests (207 → 231 total), all passing, ruff clean
- Added 8 "Compound Engineering" lessons to CLAUDE.md
- Compared with ARBITER/CRUCIBLE repo (kachergis/ARBITER), produced M6 integration roadmap
- Ran M5 validation: 5-cycle runs for all 3 ground truths (GPT-4o via Princeton)
  - GCM → Exemplar_Agent (RMSE 0.1836) ✓
  - SUSTAIN → Clustering_Agent (RMSE 0.2687) ✓
  - RULEX → Rule_Agent (RMSE 0.1580) ✓
- Ran 4× GCM replication: RMSE std=0.018 (was 0.000 pre-M5)
- ~45 FALSE CLAIMs detected across validation runs

**Key findings:**
- **Debate now causally affects RMSE** — parameter revision persistence closes the theory→prediction loop
- Replication variance non-zero for the first time in the project's history
- Agents consistently overclaim model accuracy in critiques (predicting 0.65–0.90 when actual 0.10–0.48)
- Correct winners preserved in all runs despite the new feedback paths
- RULEX posterior shows interesting 2-cycle lag then flip (P(Exemplar)=1.0 → P(Rule)=1.0)

**Key discussion:**
- ARBITER/CRUCIBLE uses pure LLM (no computational models), role-specialized agents, HITL checkpoints, pre-registration output — complementary to this repo's Bayesian+computation approach
- M6 integration roadmap: role specialization, crux negotiation, HITL checkpoints, pre-registration, multi-framework arbitration

**Status:** M5 complete and validated. 231 tests, all 3 ground truths correct, non-zero RMSE variance.

---

## Session 17 — 2026-03-15

**Commits:** `3109d14`, `dfc6ed2`, `201ff06`, `35485d8`, `fde4949`, `e1920a4`, `f49818c`, `7fb5de3`, `d7b8ca6`, `be91b7b`, `2a57937`

**What we did:**
- Implemented all 5 M6 ARBITER features via TDD (54 new tests):
  - M6a: MetaAgentConfig — role-specialized Integrator & Critic meta-agents
  - M6b: Crux Negotiation — 6 sub-commits covering Crux dataclass, identification, negotiation, finalization, EIG boosting, and run_cycle wiring
  - M6e: Conflict Map — category field on claims, conflict_map_summary()
  - M6d: Pre-registration Output — prediction tables + adjudication criteria
  - M6c: HITL Checkpoints — optional breakpoints at key decision points
- Fixed dict new_predictions crash in summary_for_agent (2 regression tests, commit `2a57937`)
- Ran M6 live validation with GPT-4o via Princeton AI Sandbox, all M6 features enabled:
  - GCM → Exemplar_Agent (RMSE 0.1512, gap 36.4%) ✓
  - SUSTAIN → Clustering_Agent (RMSE 0.2700, gap 45.6%) ✓
  - RULEX → Rule_Agent (RMSE 0.1187, gap 67.6%) ✓ (re-run after bugfix)
- Analyzed results in depth

**M6 live validation findings:**
- **Falsification engine**: 44 claims falsified, 1 confirmed, 76 untested across all 3 runs. System converges by ruling out wrong theories, not confirming the right one.
- **Crux selectivity**: 15% acceptance rate with real LLMs (vs 100% in mock). Accepted cruxes target real theoretical fault lines.
- **Posterior collapse**: EIG≈0 after cycle 0–1 in GCM/SUSTAIN. Main architectural bottleneck — later cycles are uninformative.
- **Winner revision asymmetry**: Rule_Agent made 0 revisions and won RULEX by 67.6%. Losing agents revise futilely. Lakatos-compatible pattern.
- **RULEX self-correction**: Non-monotonic posterior trajectory — initially favored Exemplar_Agent, flipped by cycle 2. Most scientifically interesting case.
- **Meta-agents substantive**: Critic consistently identifies weakest argument; Integrator synthesizes across theories. Neither overrides Bayesian machinery.

**Key discussion:**
- ARBITER architecture is now operational: role specialization, crux negotiation, conflict tracking, pre-registration
- Posterior collapse identified as the primary bottleneck for M7 (D29)
- Updated all project documentation (TASKS, SCRATCHPAD, DECISIONS, CHATLOG, REPORT, LESSONS_LEARNED, FEATURES, README)

**Status:** M6 complete and validated. 287 tests, 3/3 correct, ARBITER features producing meaningful debate dynamics.

---

## Session 18 — 2026-03-15

**Commits:** (pending)

**What we did:**
- Implemented M7: Likelihood Tempering to address posterior collapse (D30)
- Added `learning_rate` (tau) parameter to the entire Bayesian selection pipeline:
  - `ModelPosterior.update()` — core tempering + input validation (0 < lr ≤ 1)
  - `compute_eig()` — applied in simulated posterior updates
  - `select_from_pool()`, `select_experiment()` — threaded to `compute_eig()`
  - `update_posterior_from_experiment()` — threaded to `posterior.update()`, recorded in history
  - `runner.py` — `_LEARNING_RATE = 1.0` global, wired through 3 call sites, `--learning-rate` CLI flag
  - `__main__.py` — `--learning-rate` in `_build_argparser()`
- TDD: wrote 9 failing tests first, then implemented (Red → Green → Refactor)
- 296 tests passing (287 + 9 new), ruff clean

**Tests added (TestLikelihoodTempering):**
1. Tempered update slower than untempered (entropy comparison)
2. Tempered update preserves ordering (same winner)
3. Backward compatibility at tau=1.0
4. EIG changes with learning_rate
5. EIG nonzero after tempered updates (core property)
6. History records learning_rate
7. select_from_pool threads parameter
8. CLI --learning-rate parsed
9. Input validation (rejects 0, negative, >1)

**Key discussion:**
- Posterior collapse was the #1 finding from M6 live validation (D29)
- Likelihood tempering is well-established: Grünwald (2012), Bissiri et al. (2016), Miller & Dunson (2019)
- Default tau=1.0 preserves all existing behavior — no regressions
- Recommended tau=0.1–0.3 for synthetic data with known generative models

**Status:** M7 implementation complete. Pending live validation with `--learning-rate 0.2`.

---

*This log is maintained manually. Update it at the end of each session.*
