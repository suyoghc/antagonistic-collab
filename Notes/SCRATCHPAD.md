# Scratchpad

Working notes, open questions, and in-progress plans. Clean out when work is completed.

---

## Current focus — M7: Likelihood Tempering + Codex Fixes — DONE

All M7 work complete: likelihood tempering, ARBITER toggle, config file, and 5 Codex review bug fixes. 306 tests passing.

### Next steps
- Live validation with `--learning-rate 0.2` to confirm posterior entropy stays above 0 after cycle 0
- New cognitive domains (memory retrieval, decision making)
- AutoRA integration for real data
- Longer runs (10+ cycles) to test cumulative reasoning with tempering
- Claim-responsive debate: agents should address prior falsified claims

---

## Failed approaches (do not repeat)

- **Pre-LOO prediction** — training and testing on same items gives GCM self-similarity bias (D11). Always use LOO.
- **Round-robin experiment selection** — doesn't optimize for discriminability, disadvantages models that are weak on common structures (RULEX on Type_VI).
- **Divergence ranking without per-model predictions** — agents can't interpret abstract divergence scores. They need to see which model wins on each structure.
- **Trusting LLM param overrides** — LLM agents invent parameter names (e.g., `w_i`). Always filter through `inspect.signature`.
- **Pure divergence-driven selection without diversity** — picks the same high-divergence structure every cycle (Type_VI). RULEX never gets tested on favorable structures.
