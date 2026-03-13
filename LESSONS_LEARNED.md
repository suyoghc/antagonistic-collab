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

### 2.2 Agents underuse the divergence ranking

**Expected:** Agents would pick structures from the top of the divergence ranking (5-4 had the highest divergence at 0.556).

**Actual:** No agent picked 5-4. Two out of three experiments used Type_VI, one used Type_II. Agents appeared to pick structures based on their narrative about what's theoretically interesting rather than by consulting the quantitative divergence ranking.

**Implication:** Providing information is not the same as influencing behavior. The divergence ranking was in the prompt, but agents defaulted to structures they could argue about most fluently. Stronger nudging — e.g., requiring agents to justify why they didn't pick the highest-divergence structure, or defaulting to the top-ranked structure — may be needed.

### 2.3 Conditions are being used but not strategically

**Expected:** Agents would use conditions (low_attention, high_noise, etc.) to create maximally diagnostic experiments.

**Actual:** Rule_Agent picked `low_attention` for Type_II (a reasonable choice — testing whether verbal load disrupts rule discovery). Exemplar_Agent picked `high_attention` for Type_VI in cycle 2. But the condition choices appeared to be narrative-driven ("cognitive load disrupts rule learning") rather than grounded in knowing what the parameter changes actually do to model predictions.

**Implication:** Agents don't know what `low_attention` does to model parameters. They reason about it as a psychological manipulation, not as a parameter perturbation. This is fine for ecological validity (real scientists also reason about manipulations conceptually) but limits the system's ability to find maximally discriminating experiments. A future enhancement could show agents what each condition does to each model's predictions on the selected structure.
