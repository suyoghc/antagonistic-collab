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

---

## Phase 6: 5-cycle validation with structure diversity (2026-03-14)

Ran 5-cycle validation debates for all 3 ground truth models. Discovered structure repetition pathology: divergence-driven selection picked Type_VI 4/5 times in the RULEX run. Implemented two-tier structure diversity penalty (D17) and re-ran RULEX validation.

### 6.1 GCM and SUSTAIN converge correctly at 5 cycles

**Expected:** With 5 cycles instead of 3, the correct agent should win with a wider RMSE gap.

**Actual:**

| Ground Truth | Winner | RMSE | 2nd Place | RMSE | Gap |
|---|---|---|---|---|---|
| GCM | Exemplar_Agent | 0.342 | Rule_Agent | 0.403 | 15.1% |
| SUSTAIN | Clustering_Agent | 0.344 | Rule_Agent | 0.507 | 32.2% |

Both show clear separation. SUSTAIN has the widest gap (32.2%), suggesting clustering models are easier to identify — their predictions are more distinctive because cluster recruitment is stochastic and structure-dependent.

**Implication:** The framework works well for GCM and SUSTAIN identification. 5 cycles provides sufficient signal. Clustering_Agent dominated experiment selection (12/15 across all runs) but this didn't prevent correct identification — the scoring is robust to proposal dominance because predictions are model-computed, not agent-selected.

### 6.2 Structure repetition defeats RULEX

**Expected:** Divergence-driven selection would test diverse structures, giving RULEX a fair chance.

**Actual (pre-diversity-penalty):** RULEX 5-cycle run:

| Cycle | Structure | Condition | Rule RMSE | Exemplar RMSE | Clustering RMSE |
|---|---|---|---|---|---|
| 0 | Type_VI | baseline | — | 0.376 | — |
| 1 | Type_VI | high_attention | 0.466 | 0.444 | 0.548 |
| 2 | Type_VI | high_attention | 0.472 | 0.470 | 0.521 |
| 3 | Type_VI | high_attention | 0.494 | 0.493 | 0.510 |
| 4 | five_four | high_attention | 0.504 | 0.500 | 0.500 |
| **Final** | | | **0.504** | **0.501** | **0.500** |

All agents converge to RMSE ~0.50 (random guessing). No separation after 5 cycles. 4/5 experiments used Type_VI — a structure where RULEX is weakest.

**Root cause:** Raw divergence doesn't decay with repeated testing. Type_VI has the third-highest divergence (0.444) and wins the selection every cycle because no penalty exists for retesting the same structure. Clustering_Agent always proposes Type_VI, and the moderator always selects it.

**Implication:** Optimal experiment design requires not just high discriminability but also *coverage* — testing the same structure repeatedly yields diminishing marginal information. This is a fundamental principle in adaptive experimental design (Myung & Pitt 2009). The system needed an explicit mechanism to enforce diversity.

### 6.3 Two-tier diversity penalty fixes RULEX

**Expected:** Penalizing previously-tested structures would force exploration of structures where RULEX has an advantage.

**Actual (post-diversity-penalty):**

| Metric | Before | After | Change |
|---|---|---|---|
| Rule_Agent RMSE | 0.504 (3rd) | **0.433 (1st)** | Now wins |
| Exemplar_Agent RMSE | 0.501 (2nd) | 0.441 (2nd) | Improved |
| Clustering_Agent RMSE | 0.500 (1st) | 0.515 (3rd) | Correctly worst |
| Gap (1st vs 2nd) | 0.2% | 1.8% | 9x wider |
| Unique structures | 2 | 4+ | Much more diverse |
| Rule_Agent experiments | 0/5 | 1/5 | First time ever |

The penalty halves a structure's effective divergence for each prior use (exact condition repeat: 2x decay, same structure new condition: 1.5x decay). Over 5 cycles, this forces testing Type_VI, five_four, Type_I, Type_V, and others rather than Type_VI four times.

**Implication:** A simple heuristic (exponential decay on repeated structures) dramatically improved RULEX identification. The gap is still small (1.8%) compared to GCM (15.1%) and SUSTAIN (32.2%), which reflects the genuine GCM flexibility confound — not a system limitation. A more principled approach (Bayesian expected information gain) could further improve selection, but the heuristic demonstrates that structure diversity is the key missing ingredient.

### 6.4 The GCM flexibility confound is a real scientific finding

**Expected:** With enough system improvements, Rule_Agent should win with a gap comparable to Exemplar_Agent's (15%).

**Actual:** Analysis of the divergence map reveals that GCM and RULEX have low pairwise divergence across ALL structures:

| Structure | GCM-RULEX divergence | RULEX advantage? |
|---|---|---|
| Type_I | 0.341 | Tied (both ~1.0) |
| linear_separable_4d | 0.276 | Tied (~0.80) |
| linear_separable_2d | 0.229 | Tied (~0.65) |
| Type_III–V | 0.325 | Tied (~0.50) |
| Type_VI | 0.400 | No (SUSTAIN wins) |

The high divergence scores on structures like linear_separable_2d (0.619) are between RULEX/GCM vs SUSTAIN, not between RULEX vs GCM. GCM mimics RULEX by assigning high attention weights to the diagnostic dimension.

**Implication:** The 1.8% RULEX gap is not a system bug — it reflects a genuine property of these models. GCM is a more flexible model that can approximate rule-based behavior through dimensional attention, while RULEX cannot approximate exemplar behavior as well. This asymmetry is well-documented (Nosofsky 1991, Nosofsky & Johansen 2000). The system is correctly detecting this: it identifies the correct model but with low confidence, which is the scientifically honest result. A stronger discriminating test would require category structures specifically designed to maximize GCM-RULEX divergence, or alternative metrics like learning curves (where RULEX predicts sudden rule discovery vs GCM's gradual exemplar accumulation).

### 6.5 Critique quality is stable but formulaic

**Expected:** Critique quality might degrade over cycles (circular "my model can also predict that" patterns).

**Actual:** Critiques remained substantive through all 5 cycles. Each critique cited model-specific mechanisms and made quantitative claims. However, the critique template was consistent throughout:

> "[Other model] can also account for [phenomenon] because [mechanism], so [your experiment] is not diagnostic."

This is logically valid but formulaic. Agents never identified genuinely novel discriminating conditions — they critiqued within a fixed repertoire of "my model can do that too" arguments. Theory revisions were marked "progressive" but typically added auxiliary assumptions rather than fundamental changes.

**Implication:** The adversarial critique phase generates useful qualitative content but doesn't drive experiment selection or resolve disputes. The cycle 5 audit correctly detected "no convergence collapse" but also no genuine convergence. Critiques need to be causally connected to experiment design — a good critique should suggest a *better* experiment, not just argue that the proposed experiment is non-diagnostic. This could be achieved by having the moderator consider critiques when selecting experiments, or by having a "critique-driven revision" phase where critiques explicitly propose alternative structures.

---

## Phase 7: Full-pool EIG mode validation (2026-03-14)

Implemented the debate-as-hypothesis-generator architecture (D19): full-pool Bayesian EIG over all 55 structure×condition candidates replaces LLM proposal/critique/revision/arbitration phases. Agents shift to interpreting results, generating hypotheses, detecting confounds, and proposing novel structures. Learning curves implemented as second evidence channel. Novel structure validation added. Fixed a phase state machine desync bug (D20) discovered during integration testing.

### 7.1 Phase state machine desync in new mode

**Expected:** Adding `--mode full_pool` to `run_cycle()` would work by simply calling different functions for the selection and interpretation phases.

**Actual:** The cycle counter never incremented. The root cause was subtle: `advance_phase()` uses `self.current_phase` (not the result's phase) to determine the next state. After divergence mapping, `current_phase = EXPERIMENT_PROPOSAL`. In legacy mode, phases 3-6 advance through PROPOSAL → CRITIQUE → REVISION → ARBITRATION → EXECUTION. In full_pool mode, we called `advance_phase()` directly from EXPERIMENT_PROPOSAL, which transitioned to ADVERSARIAL_CRITIQUE — not EXECUTION. The state machine never reached AUDIT, so `advance_cycle()` never fired.

Unit tests didn't catch this because they mocked at function level (e.g., testing `run_full_pool_selection` in isolation). Only an integration test exercising the full `run_cycle()` flow revealed the desync.

**Fix:** `skip_to_phase(Phase.HUMAN_ARBITRATION)` before advancing from the EIG selection result. This restores the correct transition chain: HUMAN_ARBITRATION → EXECUTION → INTERPRETATION → AUDIT → advance_cycle().

**Implication:** When a pipeline has a state machine controlling phase transitions, adding alternative paths (like full_pool mode) requires careful attention to the state machine invariants. The transition map is implicit — each phase assumes the previous one was the expected predecessor. Bypassing intermediate phases without updating the state creates silent desync bugs that manifest as downstream failures (in this case, cycle counter stuck at 0). Integration tests are essential for multi-phase orchestration; unit tests on individual phases give false confidence.

### 7.2 EIG selects diverse structures without heuristic penalty

**Expected:** Full-pool EIG would need some form of diversity mechanism (like the D17 heuristic penalty) to avoid repeating structures.

**Actual:** In the 2-cycle validation run, EIG selected `five_four / fast_presentation` (cycle 0) and `Type_I / low_attention` (cycle 1) — two different structures without any diversity penalty. The Bayesian posterior update after cycle 0 (P(Exemplar)=1.0) shifted the EIG landscape so that structures distinguishing the remaining two models became more informative.

**Implication:** Bayesian EIG is naturally self-diversifying through posterior updates. Once an experiment resolves one comparison (e.g., GCM vs SUSTAIN), the posterior shifts so that the next EIG computation favors experiments that resolve the remaining uncertainty (e.g., GCM vs RULEX). This is qualitatively different from the heuristic penalty, which decays based on usage count regardless of what was learned. The heuristic is a blunt approximation of what EIG does principally. However, the 2-cycle run is too short to confirm this — 5-cycle runs are needed to see if EIG avoids pathological repetition over longer horizons.

### 7.3 Full-pool mode produces correct convergence with fewer LLM calls

**Expected:** Full_pool mode should produce similar or better convergence to legacy mode while eliminating LLM calls for experiment selection.

**Actual:** 2-cycle validation with GCM as ground truth:

| Agent | RMSE | Rank |
|---|---|---|
| **Exemplar_Agent** | **0.139** | **1st** |
| Clustering_Agent | 0.298 | 2nd |
| Rule_Agent | 0.352 | 3rd |

The correct agent wins decisively. LLM calls per cycle were reduced:

| Phase | Legacy mode calls | Full_pool mode calls |
|---|---|---|
| Commitment (cycle 0 only) | 3 | 3 |
| Divergence mapping | 3 | 3 |
| Experiment proposal | 3 | 0 |
| Adversarial critique (2 rounds) | 6 | 0 |
| Design revision | 3 | 0 |
| Execution predictions | 3 | 3 |
| Interpretation / debate | 3 | 3 |
| Interpretation critique | 0 | 3 |
| Audit | 1 | 1 |
| **Total per cycle** | **25** | **16** |

Full_pool mode uses 36% fewer LLM calls per cycle while producing correct convergence. The removed calls (proposal, critique, revision) were the ones where LLMs performed worst (see Phase 1 lessons: agents underuse divergence ranking, propose narratively familiar structures). The added calls (interpretation critique) are where LLMs add genuine value — challenging each other's causal explanations.

**Implication:** The reorganization confirms the core thesis of D19: LLMs add value for qualitative scientific reasoning (interpretation, hypothesis generation, confound detection) but not for quantitative experiment selection (which EIG does better, faster, and cheaper). The debate still matters — it just matters in a different place in the pipeline. This is a concrete instantiation of the "LLM for semantics, model for numerics" principle from Phase 3 (lesson 3.3), now extended to "LLM for interpretation, Bayesian for selection."

### 7.4 Interpretation debate produces structured, actionable output

**Expected:** Agents would produce valid JSON with interpretation, confounds, hypotheses, and optional novel structures.

**Actual:** All 3 agents produced valid structured JSON in every cycle. Example interpretation (Exemplar_Agent, cycle 0):

> "Results support my model's predictions. Under fast presentation conditions, participants appear to rely more on exemplar-based strategies, potentially due to limited time for forming complex abstractions."

Confounds flagged included "small sample size" and "condition may not discriminate sufficiently." Hypotheses were forward-looking: "Next we should test a harder structure."

No novel structures were proposed in the 2-cycle run — agents defaulted to `null` for the `novel_structure` field. This may require stronger prompting or examples in the interpretation prompt.

**Implication:** The structured JSON format works well for extracting machine-readable outputs from natural language reasoning. The interpretation critique phase produced substantive challenges — agents disputed each other's causal claims and offered alternative explanations grounded in their theoretical frameworks. However, novel structure generation may need explicit few-shot examples showing what a valid novel structure looks like (stimuli array, labels array) to trigger creative proposals. The current prompt describes the format but doesn't demonstrate it.

---

## Phase 8 — Full integration: learning curves + novel structures + 5-cycle validation (Session 13)

### 8.1 Learning curves as second evidence channel work as designed

**Expected:** Adding learning curve RMSE to Bayesian posterior updates would improve model discrimination, especially for the hard GCM-RULEX pair.

**Actual:** Full_pool mode with learning curves produces dramatically better RMSE separation than legacy mode:

| Ground Truth | full_pool Winner RMSE | full_pool Gap | legacy Winner RMSE | legacy Gap |
|---|---|---|---|---|
| **GCM** | 0.1606 | 34% | 0.2551 | 37% |
| **SUSTAIN** | 0.2701 | 42% | 0.3605 | 34% |
| **RULEX** | 0.1187 | 68% | 0.4294 | 2.4% |

The RULEX result is the headline finding: the GCM-RULEX discrimination problem (1.8–2.4% gap in legacy mode) is completely resolved in full_pool mode (68% gap). This confirms the Phase B hypothesis from D19 — curve *shape* (gradual vs sudden) provides evidence orthogonal to final accuracy, and the GCM-RULEX pair that looks similar in accuracy diverges sharply in learning dynamics.

**Implication:** Learning curves are essential for discriminating models with similar asymptotic accuracy but different learning mechanisms. This is a concrete demonstration that multiple evidence channels in Bayesian updating outperform single-channel accuracy-only approaches.

### 8.2 Novel structure generation works with few-shot prompting

**Expected:** With few-shot examples (D24) and validation (D23), agents would propose valid novel structures during interpretation.

**Actual:** Agents proposed novel structures in every cycle of the 5-cycle full_pool runs. Structures registered include: `overlapping_features`, `rule_vs_similarity`, `complex_multimodal`, `scatter_grouped`, `subgroup_multimodal`, `complex_conjunction`, `order_effects_challenge`. Validation correctly rejected malformed proposals.

**Implication:** Few-shot examples are necessary and sufficient for triggering creative structure proposals from LLM agents. The validation gate (4-32 items, ≤8 dims, ≥2 categories) prevents garbage structures from entering the EIG pool. Whether these novel structures actually improve discrimination beyond the 11 registry structures requires further analysis.

### 8.3 Full_pool mode outperforms legacy mode across all ground truths

**Expected:** Full_pool mode (EIG + learning curves + interpretation debate) would converge faster or with larger gaps than legacy mode (9-phase LLM proposal flow).

**Actual:** Full_pool produces correct winners in all 3 conditions with consistently larger RMSE gaps:

| Ground Truth | full_pool Gap | legacy Gap | Improvement |
|---|---|---|---|
| **GCM** | 34% | 37% | Similar |
| **SUSTAIN** | 42% | 34% | +8pp |
| **RULEX** | 68% | 2.4% | +66pp |

GCM discrimination is comparable between modes (both get it right easily). SUSTAIN discrimination is moderately better in full_pool mode. RULEX discrimination is transformatively better — from nearly indistinguishable to clear separation.

**Implication:** The architecture redesign (D19) is validated. Moving experiment selection from LLM agents to Bayesian EIG, and adding learning curves as a second evidence channel, produces a strictly better system for all 3 ground truth conditions. The debate still matters for interpretation and hypothesis generation — but not for experiment selection.

### 8.4 D25 crash: non-string new_predictions in summary_for_agent

**Expected:** Full_pool runs would complete without crashes.

**Actual:** All 3 initial full_pool runs crashed at cycle 0 → audit phase. `summary_for_agent()` called `'; '.join(...)` on `new_predictions` that contained dicts from LLM revision output.

**Fix:** Coerce to `str()` before joining. This is the same class of bug as D21 (scalar addresses_critiques): LLM outputs have unpredictable types, and any code that assumes string-typed fields will eventually crash.

**Implication:** Every `join()`, format string, or type-sensitive operation on LLM-derived data should defensively coerce inputs. This pattern has now appeared 3 times (D21 scalar, D25 dict predictions, earlier format crash on 'N/A').

---

## Phase 9 — M4 Analysis: Cross-run patterns from 5-cycle validation (Session 13)

### 9.1 EIG structure selection patterns

**full_pool mode** — EIG selects from 55+ candidates (11 structures × 5 conditions + novel structures):

| Ground Truth | Cycle 0 | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|---|
| GCM | five_four/fast | five_four/fast | five_four/fast | five_four/fast | five_four/fast |
| SUSTAIN | five_four/fast | Type_I/fast | five_four/baseline | five_four/baseline | five_four/baseline |
| RULEX | five_four/fast | Type_I/low_attn | Type_I/low_attn | Type_I/low_attn | Type_I/low_attn |

**Observations:**
- `five_four / fast_presentation` is the universally highest-EIG experiment in cycle 0 — it has the most items (9) and the most complex category boundary, producing maximal model disagreement
- GCM ground truth: EIG locks onto five_four/fast for all 5 cycles because posterior collapses to P(Exemplar)=1.0 after cycle 0 and EIG remains highest there
- RULEX ground truth: EIG shifts from five_four to Type_I after cycle 0 — once exemplar is initially favored, Type_I/low_attention is the most discriminating follow-up (simple rule structure where RULEX excels)
- SUSTAIN ground truth: mixed selection shows EIG exploring the structure space more

**Legacy mode** — LLM agents propose experiments:

| Ground Truth | Cycle 0 | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|---|
| GCM | 5-4/baseline | linear_sep_4d/high_attn | rule_plus_exc/baseline | 5-4/high_attn | Type_VI/baseline |
| SUSTAIN | 5-4/high_attn | 5-4/high_attn | complex_cat/high_attn | 5-4/baseline | Type_VI/baseline |
| RULEX | complex_cat | 5-4 | SUSTAIN_multimodal | RULEX_verbal | SUSTAIN_Type_V |

**Key contrast:** Legacy mode gets more structure diversity (agents propose different structures each cycle) but this diversity is not strategically optimal — it's driven by agents' narrative preferences, not information gain. Full_pool mode's "boring" repeated selection of the same high-EIG structure actually produces better discrimination.

### 9.2 Posterior convergence speed

**Full_pool mode** posterior convergence (log scale):

| Ground Truth | Cycle 0 → P(correct) | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|---|
| GCM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SUSTAIN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RULEX | 5.5e-5 (wrong!) | 5.5e-5 | **0.9998** | 1.0000 | 1.0000 |

**Critical finding:** RULEX posterior starts *wrong* — Exemplar_Agent is initially favored (P=1.0) because GCM fits five_four better than RULEX. The posterior flips in cycle 2 when EIG shifts to Type_I/low_attention, a structure where RULEX dramatically outperforms GCM (RMSE 0.06 vs 0.35). By cycle 3, Rule_Agent has P=1.0.

This is the Bayesian system working exactly as designed: initial evidence can mislead, but subsequent experiments chosen to maximize information gain eventually find the correct model. The 2-cycle lag for RULEX explains why short (2-cycle) runs may not converge for hard model pairs.

GCM and SUSTAIN converge immediately (cycle 0) because these models are easily distinguishable on five_four.

### 9.3 Novel structure generation in practice

Across 3 full_pool runs (15 total cycles), agents proposed **21 novel structures**:

| Category | Examples | Count |
|---|---|---|
| Random/unstructured | random_assignment, randomized_no_rule, random_category_assignment | 5 |
| Complex conjunctive | complex_conjunction, complex_conjunctive, noisy_xor | 4 |
| Multimodal/subgroup | multimodal_subgroups, overlapping_clusters, multi_modal_split | 5 |
| Attention/order-based | order_dependency_test, nonverbal_complex | 3 |
| Other | noisy_or, staggered_overlap, asymmetric_complex, nonlinear_family_resemblance | 4 |

**Important caveat:** None of these novel structures were *selected* by EIG — the Bayesian selector consistently chose registry structures (five_four, Type_I) over agent-proposed ones. This suggests either:
1. The novel structures don't actually maximize information gain (likely — LLMs propose narratively interesting structures, not statistically optimal ones)
2. The 11 registry structures already span the relevant space well
3. Novel structures need multiple cycles to accumulate enough EIG advantage

Whether agent-proposed structures add value beyond the registry remains an open question. They may be more useful for *longer* runs (10+ cycles) where registry structures become exhausted.

### 9.4 Legacy vs full_pool: where LLM proposals hurt

In legacy mode, Exemplar_Agent proposed 10/15 experiments across the 3 runs. This is the "proposal bias" from D5 (batch mode): the first agent's proposal tends to get approved.

More critically, the legacy proposals often **miss the most discriminating structures**:
- Legacy RULEX run never tested Type_I (the single structure where RULEX dominates GCM)
- Legacy SUSTAIN run tested five_four 3 times with high_attention — repeating the same non-diagnostic condition
- Legacy GCM run tested 5 different structures but spent cycle 4 on Type_VI (hardest structure, lowest discriminability)

**Lesson:** LLM agents propose experiments that tell a good story ("let's test complex categories!") rather than experiments that maximize statistical discrimination. This is the core reason full_pool mode outperforms legacy mode.

### 9.5 Theory revision patterns

| Theory | Revisions when TRUE model | Revisions when NOT true |
|---|---|---|
| GCM (Exemplar) | 0 (stays stable) | 1–4 (progressive) |
| RULEX (Rule) | 1 | 0–1 (stays stable) |
| SUSTAIN (Clustering) | 0 (stays stable) | 2–4 (progressive) |

**Pattern:** Correct theories don't need to revise — their predictions already match the data. Incorrect theories revise progressively (adapting parameters, adjusting claims) but never degeneratively. This is a Lakatos-compatible outcome: theories under pressure accommodate rather than degenerate, which is scientifically healthy behavior.

RULEX is notably resistant to revision even when wrong (0–1 revisions) — consistent with its rigid rule-based structure having fewer free parameters to adjust.

### 9.6 Interpretation debate quality audit

Audited all 30 debate cycles across 6 validation runs (3 full_pool + 3 legacy, 5 cycles each). Four dimensions assessed: data citation accuracy, critique quality, behavioral adaptation, and novel structure rationale.

#### Data citation accuracy: WEAK

Agents reliably cite posterior probabilities ("Bayesian posterior = 1.0000") but rarely reference specific RMSE values, item-level prediction errors, or learning curve shapes. Posteriors serve as a proxy for understanding rather than a summary of it. Example (full_pool GCM, Cycle 1): Rule_Agent cites "posterior probability of 1.0000" but doesn't reference the actual mean accuracy (0.5875) or its own RMSE (0.352).

Rare positive example (legacy SUSTAIN, Cycle 1, Clustering_Agent): "prediction for items 5, 6, 7, and 8 was quite close to the observed high accuracies, suggesting that SUSTAIN captured the correct cluster recruitment for these items. However, predictions for items 0 to 4 were not as accurate." This item-level engagement is the exception, not the norm.

#### Critique quality: MIXED

Critiques are structurally substantive but shallow in model grounding. Best critiques cite specific mechanisms — e.g., (full_pool GCM, Cycle 4): "under the RULEX framework, if participants are faced with the five-four structure, they might resort to memorizing exceptions when simple rule application fails. This mechanism could result in similar categorization accuracy, especially if p_exception is high." This names a parameter (p_exception) and a testable prediction.

More commonly, critiques use generic claims: "model flexibility in fitting attention weights post-hoc means it can often fit a wide range of data" — without specifying which weights diverge or how. The dominant pattern is theory-driven argument (abstract mechanism claims) rather than data-driven argument (specific numerical divergences). Unfalsifiable critiques like "my model can also predict that" remain common.

#### Behavioral adaptation: LIMITED

Agents repeat the same 2–3 talking points across all 5 cycles within a run. Example (full_pool GCM): Exemplar_Agent cycles through (1) exemplars account for individual items, (2) attention weight flexibility, (3) no-information-loss assumption — in cycles 0, 1, 2, 3, and 4. Novel structure proposals show no cumulative learning: agents don't revisit why prior proposals weren't selected or build on them.

Theory revisions are parameter tweaks ("adjust r and tau"), not conceptual changes. No agent ever says "this mechanism is fundamentally wrong." Critiques don't adapt to shifting posteriors — even when posterior is 1.0 for a competitor, agents restate the same objections.

**What does improve:** In later cycles of legacy mode, where adversarial critique + design revision phases force engagement, proposals become more specific. Example (full_pool GCM, Cycle 4): agents propose transfer phases with novel stimuli and dual-task paradigms — concrete manipulations grounded in mechanism predictions. This suggests the critique-and-revise forcing function works, but only after several cycles.

#### Novel structure rationale: POOR

Agents propose structures to "test their advantage" without citing actual prediction divergence. Proposals duplicate existing structures with condition permutations rather than designing for maximum model discrimination. Example (legacy GCM, Cycle 1): all three agents propose variations of the same ~4 structures already tested.

Rare exception (full_pool GCM, Cycle 4): "introduce a transfer phase where novel stimuli are presented — this will discern whether participants rely on similarity to stored exemplars or rule abstraction." This names an observable outcome that distinguishes models, but only appears in late cycles after sustained critique pressure.

#### Summary

| Dimension | Rating | Key issue |
|---|---|---|
| Data citation | Weak | Posteriors as proxy; item-level data rarely cited |
| Critique quality | Mixed | Mechanism-aware but numerically ungrounded |
| Behavioral adaptation | Limited | Same talking points repeat; no cumulative learning |
| Novel structure rationale | Poor | Not rooted in actual model divergence |

**Implication:** The interpretation debate adds value primarily through the forcing function of adversarial critique — it pressures agents to refine proposals in later cycles. But the debate does not produce cumulative scientific reasoning. Agents don't learn from prior cycles' data. Future iterations should enforce numerical citation requirements (e.g., "cite 3 specific item predictions that diverge") and track whether agents update specific claims based on data rather than repeating generic theoretical arguments.

### 9.7 Replication reveals zero variance — debate is epiphenomenal to RMSE

**Expected:** Replication runs (3× per ground truth, full_pool mode) would show some variance in RMSE gaps, allowing confidence interval estimation.

**Actual:** All replicates produced identical RMSE values to 4 decimal places:

| Ground Truth | RMSE (all 3 reps) | Winner |
|---|---|---|
| GCM | 0.1587 | Exemplar_Agent |
| SUSTAIN | 0.2701 | Clustering_Agent |
| RULEX | 0.1580 | Rule_Agent |

The entire quantitative pipeline is deterministic: EIG selection (same prior → same experiment), synthetic data (md5-seeded), model predictions (deterministic). The LLM debate text varies across replicates but does not feed back into RMSE scores.

**Implication:** This is the strongest possible evidence for finding 9.4 (debate doesn't influence outcomes). The debate is literally epiphenomenal to the quantitative result — it can be removed entirely without changing convergence. The value of debate is exclusively in the qualitative layer: human-readable explanations, mechanistic narratives, and hypothesis generation. Future architectures should acknowledge this separation explicitly: use the Bayesian pipeline for convergence, and the LLM debate for communication and interpretation only.

### 9.8 Cross-LLM comparison: GPT-4o vs Claude Sonnet vs Claude Opus

**Expected:** Different LLM backbones might produce different convergence outcomes, since debate quality and parameter proposals vary.

**Actual:** Correct model wins in all 9/9 runs (3 ground truths × 3 LLMs):

| Ground Truth | GPT-4o Winner (RMSE) | Sonnet Winner (RMSE) | Opus Winner (RMSE) |
|---|---|---|---|
| GCM | Exemplar (0.159) | Exemplar (0.159) | Exemplar (0.143) |
| SUSTAIN | Clustering (0.270) | Clustering (0.270) | Clustering (0.270) |
| RULEX | Rule (0.158) | Rule (0.148) | Rule (0.213) |

**Key observations:**

1. **SUSTAIN perfectly deterministic** — identical RMSE (0.270) across all 3 LLMs. The param_overrides proposed by agents had no effect on SUSTAIN's predictions for this structure/condition sequence.

2. **GCM nearly identical** — Opus slightly lower (0.143 vs 0.159). Opus proposed param_overrides that improved GCM's fit marginally.

3. **RULEX shows most variation** — Opus higher RMSE (0.213) vs Sonnet (0.148) and GPT-4o (0.158). Opus's param_overrides for RULEX were less effective, but the correct agent still won with a 42% gap.

4. **The variation source is `param_overrides`** — the only code path where LLM output affects RMSE. During execution, agents propose parameter tweaks that are applied to one prediction. Different LLMs propose different overrides, creating small RMSE differences.

**Implication:** The framework is LLM-agnostic for convergence — the correct model wins regardless of backbone. RMSE varies slightly through param_overrides (the one surviving feedback path from LLM to quantitative pipeline), but not enough to change outcomes. This confirms the architecture thesis: convergence is driven by computation, not by which LLM generates the debate text.
