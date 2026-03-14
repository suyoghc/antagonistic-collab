# Lessons Learned (Chronology)

Findings that emerged from building and running the antagonistic collaboration framework. Organized by development phase. Each lesson documents what we expected, what actually happened, and what it implies.

---

## Phase 1: First end-to-end runs (2026-03-11 → 2026-03-13)

After fixing ~30 bugs across 5 modules and getting the 9-phase debate protocol running end-to-end, we ran multiple 3-cycle batch debates with GPT-4o (via Princeton/Portkey). The debate machinery worked — agents proposed experiments, critiqued each other, revised theories, made predictions. But the system failed to produce meaningful scientific convergence. Five issues surfaced, each revealing something about the architecture of LLM-mediated science.

### 1.1 The specification gap

**Expected:** LLM agents would propose experiment designs that the synthetic data generator could execute, producing model-sensitive data.

**Actual:** Agents produced scientifically sophisticated but computationally inert experiment specifications. A typical proposal:

> "Non-linearly-separable categories with family resemblance structure, 3 conditions (Exemplar Training, Rule Induction, Clustering Induction), 120 subjects"

The synthetic data generator needed `{"stimuli": np.array, "labels": np.array}` — a precise numerical specification of category structure. No agent ever produced output in this format. The generator silently fell back to a default (Shepard Type II) every time.

**Implication:** There is a fundamental disconnect between natural-language scientific reasoning and formal computational specification. LLMs can reason fluently about experimental design at the conceptual level — choosing between within- vs. between-subjects designs, identifying diagnostic category structures, specifying dependent variables — but they cannot close the loop from concept to executable code without an explicit translation layer. This is likely a general challenge for any system where LLMs interface with formal computational backends (simulation, lab automation, code generation). The system needs either (a) constrained choice from a pre-validated menu, or (b) a structured intermediate representation that the LLM can produce and the backend can consume.

### 1.2 Eloquent but ungrounded debate

**Expected:** Agents would update their theoretical commitments in response to experimental data, with the ground-truth model's agent gradually accumulating better predictions.

**Actual:** Every experiment returned `mean_accuracy = 0.550` regardless of the proposed design. Despite this, agents wrote detailed, plausible-sounding interpretations:

> "The lower-than-expected accuracy suggests that participants may not have been relying solely on exemplar-based categorization. The discrepancy between predictions and data implies that there could be factors or processes not adequately captured by the current model."

This interpretation is entirely confabulated. The data didn't reflect the proposed experiment at all — it was the same default structure every time. Yet the agent produced a coherent narrative that reads as genuine scientific reasoning.

**Implication:** LLMs will generate plausible explanatory narratives for any data, including data that is effectively random relative to the question being asked. The debate *looked* like science — agents critiqued proposals, revised theories, explained prediction errors — but none of it was grounded in actual data variation. This is a vivid demonstration of the difference between fluent reasoning and empirically grounded reasoning. Without discriminating data, the entire interpretive apparatus runs on empty. The form of science was present; the function was absent.

### 1.3 The constrain-vs-parse design tension

**Expected:** We would need to choose between letting agents design experiments freely (maximizing scientific creativity) and constraining them to a fixed menu (maximizing execution reliability).

**Actual:** Freeform design failed completely (see 1.1). The agents invented a different specification format in every cycle — exemplar lists, cluster center specifications, condition descriptions, formal logic — none of which matched the generator's expected input. Parsing all possible formats would be intractable.

We identified four possible solutions:
- **(A)** Constrain agents to choose from the existing category structure library (Shepard I–VI, 5-4, rule+exception, linear-separable)
- **(B)** Build a parser for LLM-generated specifications (brittle, format varies every time)
- **(C)** Map experimental conditions to model parameter perturbations (same structure, different data)
- **(D)** Pre-compute a divergence-driven experiment menu (agents debate which pre-computed experiment to run)

**Implication:** This is a real design tension that likely applies to any LLM-in-the-loop scientific system. Constraining the action space sacrifices expressiveness but guarantees executability. The optimal point probably involves structured choice with room for parameterization — agents pick a category structure from a validated set but can specify conditions (e.g., "high cognitive load") that map to parameter variations. This preserves some scientific agency while ensuring the computational backend can execute the design. The key insight is that the *translation layer* between natural language and formal specification is where systems like this break down, and it must be designed explicitly rather than assumed.

### 1.4 Circular critique without discriminating data

**Expected:** Adversarial critique would sharpen experiment designs across rounds, with agents identifying genuine weaknesses in each other's proposals.

**Actual:** Every agent's critique of every proposal followed the same template: "My model can also predict that pattern, so your experiment is not diagnostic." Examples:

- Exemplar_Agent on a clustering proposal: "GCM can also predict order effects through differential attention weight adjustment."
- Rule_Agent on an exemplar proposal: "RULEX can handle non-linearly-separable categories by storing exceptions."
- Clustering_Agent on a rule proposal: "SUSTAIN's cluster recruitment can be impaired by cognitive load, similar to rule disruption."

This critique is logically valid — these models *can* in principle account for many of the same phenomena through different mechanisms. But without data that actually distinguishes the models, the critique never resolves. It becomes an unfalsifiable loop: every model claims to explain everything, no experiment can adjudicate, and the debate doesn't progress.

**Implication:** Argumentation alone does not produce scientific convergence. Adversarial pressure is necessary but not sufficient — it requires empirical traction. When models are flexible enough to post-hoc accommodate any result (as these cognitive models are), the critique phase needs to be paired with data that creates asymmetric predictions. This connects to the philosophical problem of underdetermination: when multiple theories are compatible with the same evidence, no amount of argument between them will resolve the dispute without new, discriminating evidence. The fix is not better argumentation but better experiments — specifically, experiments that the models make *different* quantitative predictions about, which requires the data generator to actually vary across experiments.

### 1.5 Scoring misalignment

**Expected:** The prediction leaderboard would track which agent's model best explains the data, with the ground-truth model's agent (GCM, in our runs) accumulating the lowest RMSE.

**Actual:** Agents predicted rich multivariate patterns — generalization gradients, reaction times, cluster recruitment dynamics, per-condition accuracies — but the scoring function compared only `mean_accuracy`. The leaderboard ranked agents by how close their single-number prediction was to 0.55:

| Agent | Mean RMSE | True model? |
|-------|-----------|-------------|
| Rule_Agent | 0.125 | No |
| Exemplar_Agent | 0.160 | **Yes (GCM)** |
| Clustering_Agent | 0.187 | No |

The "best" predictor was the theoretically *wrong* model. Rule_Agent won simply by guessing closer to 0.55, not by making theoretically grounded predictions. Meanwhile, the models already compute per-item classification probabilities that diverge substantially across category structures — this information was generated but thrown away by the scalar scoring.

**Implication:** The metric determines what the system optimizes for, and a scalar metric on flat data incentivizes lucky guessing over theoretical insight. This is an alignment problem in miniature. The fix has two parts: (a) make the data vary so that predictions are non-trivially constrained, and (b) score on the dimensions where models actually diverge — per-item accuracy patterns rather than grand means. The models already produce the richer output; the scoring just needs to use it.

---

## Phase 2: Model-sensitive synthetic data (2026-03-13)

Fixed the data pipeline so different experiments produce different data. Added a structure registry (11 structures), condition effects (5 conditions with per-model param overrides), rewrote `_synthetic_runner()` to use structure lookup and md5-based per-experiment seeds, merged `item_accuracies` into scoring, and updated prompts with structure/condition menus and item-level prediction guidance. 9 new tests, 82 total passing.

### 2.1 Data variation works, but the wrong agent wins

**Expected:** With model-sensitive data, the ground-truth model's agent (GCM/Exemplar_Agent) should accumulate the best prediction RMSE over 3 cycles.

**Actual:** Data now genuinely varies across experiments:

| Cycle | Structure | Condition | mean_accuracy |
|-------|-----------|-----------|---------------|
| 0 | Type_VI | baseline | 0.666 |
| 1 | Type_II | low_attention | 0.473 |
| 2 | Type_VI | baseline | 0.651 |

But the final leaderboard after 3 cycles:

| Agent | Mean RMSE | True model? |
|-------|-----------|-------------|
| Clustering_Agent | 0.1135 | No |
| Exemplar_Agent | 0.1483 | **Yes (GCM)** |
| Rule_Agent | 0.2021 | No |

Clustering_Agent beat Exemplar_Agent despite GCM being the ground truth. The reason: agents don't actually *run* their models to generate predictions. The LLM *reasons about* what its model would predict, then writes down numbers. Clustering_Agent happened to make more conservative, closer-to-mean predictions that scored better. The RMSE reflects how well the LLM approximates each model's behavior, not the model's actual fit to data.

**Implication:** There is a critical gap between "the model that generated the data" and "the agent that best predicts the data." The LLM's numerical intuition about its model is not the same as running the model. This means prediction accuracy is confounded by the LLM's calibration — a well-calibrated guesser can beat the theoretically correct model. The fix is to have agents actually call `model.predict()` with their stated parameters and use those outputs as predictions, removing the LLM's numerical guessing from the scoring loop entirely.

### 2.2 More cycles don't fix LLM-guessed predictions

**Expected:** A 4-cycle run (vs 3) would give the true model's agent more opportunities to separate from competitors.

**Actual:** 4-cycle run with GCM as ground truth:

| Cycle | mean_accuracy |
|-------|---------------|
| 0 | 0.666 |
| 1 | 0.667 |
| 2 | 0.537 |
| 3 | 0.806 |

Final leaderboard comparison:

| Agent | 3-cycle RMSE | 4-cycle RMSE |
|-------|-------------|-------------|
| Clustering_Agent | 0.1135 | 0.1626 |
| Exemplar_Agent | 0.1483 | 0.1662 |
| Rule_Agent | 0.2021 | 0.1829 |

The gap between Clustering and Exemplar shrank from 0.035 to 0.004 — essentially a tie at 4 cycles. Rule_Agent is consistently worst in both runs, which is correct (RULEX should lose when GCM generates the data). But Exemplar_Agent never overtakes Clustering_Agent.

**Implication:** Adding cycles produces diminishing returns when predictions are LLM-guessed. The noise floor is set by how well the LLM approximates each model's numerical behavior, not by the underlying model's fit to data. Conservative, closer-to-mean guesses will always compete with (or beat) the theoretically correct model's agent. More data can't overcome the fact that the scoring mechanism isn't measuring model fit — it's measuring LLM calibration. The fix must be structural: wire agents to call `model.predict()` so RMSE reflects actual model behavior.

### 2.3 Rule_Agent separates correctly even with LLM guessing

**Expected:** All three agents would be similarly confounded by LLM guessing noise.

**Actual:** Rule_Agent consistently ranked worst across both runs (0.2021 at 3 cycles, 0.1829 at 4 cycles). The ordering Rule_Agent > Exemplar_Agent > Clustering_Agent held in both runs (worst to best RMSE).

**Implication:** The system *partially* works even with LLM-guessed predictions. The wrong model (RULEX) does lose. The problem is specifically at the top of the leaderboard — the true model's agent can't separate from a close competitor (SUSTAIN) because the LLM's guesses are too similar. This suggests that model-calling will help most for distinguishing between models that make similar-but-not-identical predictions. The coarse signal (rule models are wrong for GCM-generated data) already comes through.

### 2.4 Agents underuse the divergence ranking

**Expected:** Agents would pick structures from the top of the divergence ranking (5-4 had the highest divergence at 0.556).

**Actual:** Across both runs (7 total experiments), no agent ever picked 5-4. Agents overwhelmingly chose Type_VI and Type_II. They appeared to pick structures based on narrative familiarity rather than consulting the quantitative divergence ranking.

**Implication:** Providing information is not the same as influencing behavior. The divergence ranking was in the prompt, but agents defaulted to structures they could argue about most fluently. Stronger nudging — e.g., requiring agents to justify why they didn't pick the highest-divergence structure, or defaulting to the top-ranked structure — may be needed.

### 2.5 Conditions are being used but not strategically

**Expected:** Agents would use conditions (low_attention, high_noise, etc.) to create maximally diagnostic experiments.

**Actual:** Rule_Agent picked `low_attention` for Type_II (a reasonable choice — testing whether verbal load disrupts rule discovery). Exemplar_Agent picked `high_attention` for Type_VI in cycle 2. But the condition choices appeared to be narrative-driven ("cognitive load disrupts rule learning") rather than grounded in knowing what the parameter changes actually do to model predictions.

**Implication:** Agents don't know what `low_attention` does to model parameters. They reason about it as a psychological manipulation, not as a parameter perturbation. This is fine for ecological validity (real scientists also reason about manipulations conceptually) but limits the system's ability to find maximally discriminating experiments. A future enhancement could show agents what each condition does to each model's predictions on the selected structure.

---

## Phase 3: Model-computed predictions (2026-03-13)

Replaced LLM-guessed predictions with model-computed ones. Added `compute_model_predictions()` to `DebateProtocol` — during the EXECUTION phase, each agent's model is run on the approved experiment structure with condition overrides. The LLM still provides reasoning and confidence, but the numbers come from `model.predict()`. Fixed a P1 bug (from Codex review) where `param_overrides` from the LLM response were ignored and `default_params` were recorded instead of actual params used.

### 3.1 The correct agent now wins decisively

**Expected:** With model-computed predictions, the ground-truth model's agent (GCM/Exemplar_Agent) should have the lowest RMSE.

**Actual:** 3-cycle run with GCM as ground truth:

| Cycle | Structure | Condition | mean_accuracy |
|-------|-----------|-----------|---------------|
| 0 | rule_plus_exception_1exc | baseline | 0.639 |
| 1 | linear_separable_4d | baseline | 0.933 |
| 2 | Type_II | low_attention | 0.521 |

Final leaderboard:

| Agent | Mean RMSE | True model? |
|-------|-----------|-------------|
| **Exemplar_Agent** | **0.0776** | **Yes (GCM)** |
| Rule_Agent | 0.2755 | No |
| Clustering_Agent | 0.3528 | No |

Compare to the best Phase 2 run (LLM-guessed, 4 cycles):

| Agent | Phase 2 RMSE | Phase 3 RMSE |
|-------|-------------|-------------|
| Exemplar_Agent | 0.1662 | **0.0776** |
| Rule_Agent | 0.1829 | 0.2755 |
| Clustering_Agent | 0.1626 | 0.3528 |

Phase 2: all agents within ~0.02 RMSE, wrong agent (Clustering) wins.
Phase 3: 3.6x gap between first and second place, correct agent wins.

**Implication:** Removing LLM numerical guessing from the scoring loop is the single most impactful change for convergence validity. The system now measures model fit, not LLM calibration. The true model's agent wins because its model literally generated the data — as it should. This unblocks M3 convergence validation.

### 3.2 RMSE spread reveals model-specific signatures

**Expected:** Models would produce somewhat different predictions, giving a meaningful RMSE spread.

**Actual:** The RMSE spread is dramatic (0.0776 to 0.3528, a 4.5x range). In Phase 2, the spread was only 0.1626 to 0.1829 (1.1x range). Model-computed predictions amplify genuine differences between models rather than compressing them through LLM averaging.

Per-cycle prediction scores show the separation clearly:

| Cycle | Exemplar RMSE | Rule RMSE | Clustering RMSE |
|-------|-------------|---------|---------------|
| 0 | low | high | high |
| 1 | low | high | high |
| 2 | low | high | high |

Exemplar_Agent was consistently closest to the data every cycle, not just on average. This is the monotonic separation pattern that Phase 2 never produced.

**Implication:** The system now produces the kind of cumulative evidence that real scientific debates depend on — each experiment adds signal, and the gap widens rather than fluctuating randomly. This is the core behavior needed for M3/M4 multi-model validation.

### 3.3 LLM reasoning is qualitatively correct even when numbers were wrong

**Expected:** With model-computed predictions, the LLM's reasoning would become irrelevant.

**Actual:** The LLM reasoning is still informative for the transcript. Exemplar_Agent correctly explained *why* GCM would perform well ("similarity to stored exemplars handles non-linear separability"), and Rule_Agent correctly identified *why* RULEX would struggle under low attention ("conjunction rules require verbal working memory"). The reasoning was qualitatively sound even when the LLM's guessed numbers (Phase 2) were poorly calibrated.

**Implication:** The LLM adds genuine interpretive value — it explains model behavior in natural language, identifies mechanisms, and connects predictions to theory. The fix wasn't to remove the LLM from the loop, but to separate what the LLM is good at (reasoning, interpretation) from what it's bad at (precise numerical prediction). This division of labor — LLM for semantics, model for numerics — is likely a general design principle for LLM-in-the-loop scientific systems.

### 3.4 Structure diversity improved but 5-4 still unused

**Expected:** Agents would explore a wider range of structures.

**Actual:** Agents picked rule_plus_exception_1exc (cycle 0), linear_separable_4d (cycle 1), and Type_II (cycle 2) — three different structures, more diverse than Phase 2 (which used only Type_VI and Type_II). But 5-4 (highest divergence at 0.556) was still never selected across all runs.

**Implication:** Structure diversity is improving naturally as agents see different data each cycle, but the divergence ranking is still not driving choices. The 5-4 structure may need to be more prominently featured or its advantages explained more concretely in the prompt.

---

## Phase 4: Multi-model validation with LOO (2026-03-13)

Comprehensive code review fixed 12 bugs (D9–D10). Then ran multi-model validation: 3-cycle debates with each model (GCM, SUSTAIN, RULEX) as ground truth. Initial runs revealed a self-prediction bias — GCM's predictions were inflated by self-matching (distance=0, similarity=1.0). Implemented leave-one-out cross-validation (D11) and re-ran all three.

### 4.1 Self-prediction bias masks model differences

**Expected:** With model-computed predictions and per-item scoring, the correct agent should win regardless of which model is ground truth.

**Actual (pre-LOO):** SUSTAIN/Clustering_Agent won in all three conditions:

| Ground Truth | Winner | RMSE | Correct agent's RMSE | Correct? |
|---|---|---|---|---|
| GCM | Clustering_Agent | 0.1051 | 0.1323 (2nd) | No |
| SUSTAIN | Clustering_Agent | 0.0306 | 0.0306 (1st) | Yes |
| RULEX | Clustering_Agent | 0.1741 | 0.2223 (2nd) | No |

**Root cause:** `compute_model_predictions()` trains and tests on the same items. For GCM, item i is its own nearest exemplar: distance=0, similarity=exp(0)=1.0. This self-match dominates the Luce choice rule, producing near-binary predictions (~0.79 vs ~0.21) that are identical for every item in a structure. These over-confident predictions have higher RMSE against noisy synthetic data (which has binomial sampling variance from N subjects). SUSTAIN's cluster-based competition produces softer, more intermediate predictions that accidentally fit noise better.

**Implication:** Testing on the training set is a well-known methodological error, but it manifests in a non-obvious way here. The bias doesn't crash the system or produce NaN — it produces valid-looking predictions that systematically disadvantage exemplar models. The fix is leave-one-out cross-validation, which is standard in the GCM literature (Nosofsky 1986). This is a reminder that methodological correctness matters even in synthetic evaluation pipelines.

### 4.2 Leave-one-out restores correct model identification (2 of 3)

**Expected:** With LOO, the correct agent should win when their model is ground truth.

**Actual (post-LOO):**

| Ground Truth | 1st | RMSE | 2nd | RMSE | 3rd | RMSE | Correct? |
|---|---|---|---|---|---|---|---|
| **GCM** | **Exemplar_Agent** | **0.4334** | Rule_Agent | 0.4930 | Clustering_Agent | 0.5536 | **Yes** |
| **SUSTAIN** | **Clustering_Agent** | **0.4432** | Exemplar_Agent | 0.5591 | Rule_Agent | 0.6640 | **Yes** |
| **RULEX** | Exemplar_Agent | 0.4417 | Rule_Agent | 0.5153 | Clustering_Agent | 0.5802 | **No** |

GCM and SUSTAIN work correctly. RULEX fails — Exemplar_Agent outperforms Rule_Agent on RULEX-generated data.

**Implication:** LOO fixes the self-prediction bias and makes GCM competitive (it was systematically disadvantaged before). RMSE values are higher across the board (0.43–0.66 vs 0.03–0.28) because LOO makes prediction harder — this is expected and healthy. The correct agent winning for GCM and SUSTAIN validates the core framework. The RULEX failure needs separate diagnosis.

### 4.3 RULEX fails because experiment selection doesn't favor it

**Expected:** Rule_Agent would win when RULEX generates the data, because RULEX's rule-based predictions should match RULEX-generated data.

**Actual:** In the RULEX ground-truth run, the three experiments tested Type_VI, Type_II, and a linear separable structure. Type_VI is an all-exceptions structure with no simple rule — this is RULEX's *weakest* category type. RULEX has to memorize all items as exceptions, which it does stochastically and poorly. Meanwhile GCM handles Type_VI well via summed similarity to multiple exemplars. Only Type_II (conjunctive rule) was favorable for RULEX, but that was only 1 of 3 experiments.

**Structures tested by ground truth:**

| Ground Truth | Cycle 0 | Cycle 1 | Cycle 2 |
|---|---|---|---|
| GCM | Type_VI (high_attn) | Type_II (verbal load) | Linear Sep |
| SUSTAIN | Type_VI (high_attn) | Type_II (verbal load) | Multimodal |
| RULEX | Type_VI | Type_II (verbal load) | Linear Sep |

All three runs tested similar structures (agents make similar proposals). Type_VI appears in every cycle 0 because Exemplar_Agent always goes first (round-robin) and always proposes Type_VI.

**Implication:** The round-robin experiment selection is the bottleneck. When the correct model is disadvantaged on 2 of 3 structures (as RULEX is on Type_VI and linear separable), it can't accumulate a winning RMSE in just 3 cycles. This is where **the debate should matter but doesn't** — an intelligent moderator would select experiments that maximize discriminability between models, rather than cycling through agents in order.

### 4.4 The debate doesn't influence outcomes (yet)

**Expected:** The adversarial debate process (critique, interpretation, theory revision) would improve experiment selection and lead to better model identification.

**Actual:** In the current architecture, the debate contributes zero signal to the quantitative outcome:

1. **Predictions are model-computed** (D8) — RMSE comes from `model.predict()`, not LLM reasoning
2. **Experiment selection is round-robin** (D5) — each agent gets one experiment regardless of critique quality
3. **Phase 5 (Design Revision) is a placeholder** — critiques don't modify proposals
4. **Divergence ranking is ignored** — agents pick structures by narrative familiarity, not quantitative divergence

The debate generates interesting text (qualitative reasoning, mechanism-level explanations, theory revisions) but none of it enters the scoring pipeline. The system is effectively: "run 3 experiments with round-robin selection, score each model on each experiment, rank by mean RMSE."

**Implication:** This is the central architectural gap. The debate machinery has scientific value (forcing agents to articulate mechanisms, identify predictions, revise commitments) but no causal connection to the quantitative evaluation. For the debate to matter, at least one of these must change:

- **Experiment selection should use divergence information** — the moderator should pick the most discriminating experiment, not round-robin. This is where critique quality could matter: a good critique identifies why a proposed experiment fails to discriminate.
- **Phase 5 should revise proposals** — critiques should modify experiment designs before execution. If Rule_Agent's critique of a Type_VI proposal is "my model can't be tested here," the proposal should shift to a more diagnostic structure.
- **The moderator should be an LLM** — instead of round-robin, an LLM moderator could evaluate proposals + critiques and select the experiment most likely to resolve open disputes. This closes the loop between qualitative debate and quantitative evaluation.

### 4.5 Model flexibility is a confound

**Expected:** Each model would predict well on its own ground-truth data and poorly on others'.

**Actual:** RMSE gaps between agents are relatively small (GCM: 0.43 vs 0.55; SUSTAIN: 0.44 vs 0.56). All three models achieve similar accuracy on most structures because:

- **GCM** is flexible: with enough exemplars and attention weights, it approximates any decision boundary.
- **SUSTAIN** interpolates between exemplar and prototype: cluster recruitment adapts to structure.
- **RULEX** has exception storage: even when rules fail, exception memorization provides a fallback.

The models are more similar than different on most category structures. Discrimination requires *specifically chosen* structures where the models make divergent predictions — and the current system doesn't optimize for this.

**Implication:** Model flexibility is a fundamental challenge for adversarial collaboration between cognitive models. When all models can accommodate most data patterns (albeit through different mechanisms), the discriminating experiments are rare and must be deliberately sought. This mirrors the real scientific debate between these models (Nosofsky & Johansen 2000, Love et al. 2004): decades of research have focused on finding the specific conditions where the models disagree. The system needs to replicate this strategic experiment selection, not just run arbitrary experiments.

---

## Phase 5: Divergence-driven experiment selection (2026-03-14)

Replaced round-robin batch-mode arbitration with divergence-driven selection (D12). The moderator now picks the proposal whose category structure has the highest max pairwise divergence between models, falling back to critique count on ties. Also expanded `compute_divergence_map()` to use all 11 `STRUCTURE_REGISTRY` structures.

### 5.1 Divergence-driven selection works, but agents don't propose strategically

**Expected:** With the moderator selecting the most diagnostic structure from proposals, RULEX would have a better chance of winning when it's the ground truth model — the moderator would avoid RULEX-unfavorable structures.

**Actual:** 3-cycle RULEX validation with divergence-driven selection:

| Cycle | Structure Selected | Divergence | Exemplar RMSE | Rule RMSE | Clustering RMSE |
|---|---|---|---|---|---|
| 0 | Type_VI | 0.444 | 0.4896 | 0.5354 | **0.4514** |
| 1 | linear_separable_4d | 0.513 | **0.4151** | 0.4414 | 0.5488 |
| 2 | linear_separable_4d | 0.513 | **0.2569** | 0.2484 | 0.6172 |
| **Final** | | | **0.3872** | 0.4084 | 0.5720 |

RULEX still lost, but the gap narrowed from 16.7% (0.4417 vs 0.5153 in Phase 4) to 5.5% (0.3872 vs 0.4084). Notably, Rule_Agent had the lowest single-experiment RMSE in cycle 2 (0.2484 vs 0.2569).

The divergence-driven selection did its job — it picked higher-divergence structures (0.444, 0.513, 0.513) instead of low-divergence ones. But the problem shifted from the moderator to the agents:

- **Rule_Agent proposed Type_II in every cycle** (divergence = 0.160, the lowest of all 11 structures)
- **No agent ever proposed Type_I** (simple rule, RULEX's strongest case)
- **`linear_separable_2d`** (highest divergence at 0.619) was never proposed by any agent

**Implication:** The bottleneck has moved from moderator selection to agent proposal quality. Agents don't understand which structures favor their model. They choose structures by narrative appeal ("Type_II is interesting for rule learning because of XOR") rather than by quantitative advantage. The divergence ranking tells agents *which structures are most diagnostic* but not *which model wins* on each structure. An agent seeing "Type_I — divergence 0.341" doesn't know whether Type_I favors RULEX, GCM, or SUSTAIN.

### 5.2 The information gap: divergence ≠ advantage

**Expected:** Agents would use the divergence ranking to choose structures that discriminate well.

**Actual:** The divergence ranking shows max pairwise divergence (e.g., "linear_separable_2d — 0.619") but not the direction. An agent needs to know: "On Type_I, RULEX predicts 0.95 while GCM predicts 0.60 — this structure favors me." Without this, the ranking is a list of numbers with no actionable meaning for a specific agent.

The fix is to show concrete per-model predictions in the divergence context:
```
Type_I — GCM: 0.60, RULEX: 0.95, SUSTAIN: 0.55 (max divergence: 0.40)
→ Rule_Agent would immediately see this is their strongest structure
```

**Implication:** Information design matters. Providing the right abstraction level is crucial for LLM agents. Raw divergence scores are too abstract; per-model predictions are actionable. This is analogous to the real scientific problem: a researcher choosing experiments needs to know their model's predictions, not just that "the models differ." The fix is straightforward — the data is already computed in `compute_divergence_map()`, it just needs to be surfaced per-model in the prompt.

### 5.3 Updated prompt changes agent behavior

**Expected:** Explicitly telling agents to choose structures where their model has the highest predicted accuracy would change their proposals.

**Actual:** With the updated prompt, Rule_Agent proposed Type_I for the first time (run _07), and Exemplar_Agent proposed five_four. This is the first time agents chose structures based on quantitative advantage rather than narrative familiarity. However, the divergence-driven moderator still selected linear_separable_4d (proposed by Clustering_Agent) because it had the highest divergence (0.513).

### 5.4 GCM flexibility may be a fundamental limit

**Expected:** With enough improvements to experiment selection, Rule_Agent (RULEX) should eventually win when RULEX generates the data.

**Actual:** Across four runs with progressive improvements, the gap narrowed but GCM always won:

| Run | Changes | Gap |
|---|---|---|
| _03 | Round-robin | 16.7% |
| _04 | Divergence-driven | 5.5% |
| _06 | + Concrete predictions | 8.5% |
| _07 | + Updated prompt | 4.7% |

GCM outperforms RULEX on RULEX-generated data even on `linear_separable_4d` — a structure where RULEX has the highest raw accuracy (0.80 vs 0.75 for GCM). The issue: LOO accuracy measures binary classification, but RMSE measures probability calibration. GCM's probability estimates may be better calibrated than RULEX's even when RULEX's binary accuracy is higher.

**Implication:** This may reflect a genuine scientific finding: GCM is a more flexible model that can approximate rule-based behavior through attention weight optimization, while RULEX cannot approximate exemplar behavior as well. This asymmetry is well-documented in the categorization literature (Nosofsky & Johansen 2000). The system may be correctly identifying that GCM is the more parsimonious explanation even for data generated by a rule-based process — which is itself an interesting result about model flexibility and underdetermination in cognitive science.
