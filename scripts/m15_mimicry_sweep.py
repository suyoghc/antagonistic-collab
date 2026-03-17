"""
M15 Phase 1 — Parameter sweep for misspecification calibration.

Phase 1a: Mimicry sweep — compute prediction overlap between models at different
parameter settings. Result: no true mimicry exists between GCM/SUSTAIN/RULEX.

Phase 1b: Competition sweep — generate synthetic data from each ground-truth model,
score all three models against it with the correct model misspecified. Measures
how much misspecification narrows the winner's gap.

Grounded in Wagenmakers, Ratcliff, Gomez & Iverson (2004) PBCM framework and
Pitt, Myung & Zhang (2004) on model selection / parameter estimation entanglement.

No LLM calls. Pure computation.

Usage:
    python scripts/m15_mimicry_sweep.py              # both phases
    python scripts/m15_mimicry_sweep.py --phase 1a   # mimicry only
    python scripts/m15_mimicry_sweep.py --phase 1b   # competition only
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from antagonistic_collab.models.gcm import GCM
from antagonistic_collab.models.rulex import RULEX
from antagonistic_collab.models.sustain import SUSTAIN
from antagonistic_collab.models.category_structures import (
    shepard_types,
    five_four_structure,
)

# ---------------------------------------------------------------------------
# Ground-truth parameters (from debate_protocol._synthetic_runner)
# ---------------------------------------------------------------------------

GROUND_TRUTH_PARAMS = {
    "GCM": {"c": 4.0, "r": 1, "gamma": 1.0},
    "SUSTAIN": {"r": 9.01, "beta": 1.252, "d": 16.924, "eta": 0.092},
    "RULEX": {"p_single": 0.5, "p_conj": 0.3, "error_tolerance": 0.1, "seed": 42},
}

# ---------------------------------------------------------------------------
# Parameter grids to sweep
# ---------------------------------------------------------------------------

GCM_C_GRID = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]

SUSTAIN_GRID = [
    {"r": r, "eta": eta}
    for r in [3.0, 5.0, 7.0, 9.01, 12.0]
    for eta in [0.04, 0.06, 0.092, 0.12, 0.15]
]

RULEX_GRID = [
    {"error_tolerance": et, "p_single": ps}
    for et in [0.05, 0.1, 0.15, 0.2, 0.25]
    for ps in [0.3, 0.4, 0.5, 0.7]
]

# ---------------------------------------------------------------------------
# Structures — base registry only (well-studied, known model behavior)
# ---------------------------------------------------------------------------


def load_structures() -> dict[str, dict]:
    """Load Shepard I-VI + five_four."""
    structs = {}
    for type_name, s in shepard_types().items():
        structs[f"Type_{type_name}"] = s
    structs["five_four"] = five_four_structure()
    return structs


# ---------------------------------------------------------------------------
# Prediction computation — LOO per item, matching compute_model_predictions()
# ---------------------------------------------------------------------------


def loo_prediction_vector(model, structure: dict, params: dict) -> np.ndarray:
    """Compute LOO P(correct) for each item. Returns 1D array of length n_items."""
    stimuli = structure["stimuli"]
    labels = structure["labels"]
    n = len(labels)
    preds = np.zeros(n)
    for i in range(n):
        loo_stimuli = np.delete(stimuli, i, axis=0)
        loo_labels = np.delete(labels, i)
        result = model.predict(stimuli[i], loo_stimuli, loo_labels, **params)
        preds[i] = result["probabilities"].get(int(labels[i]), 0.5)
    return preds


def prediction_profile(
    model, structures: dict[str, dict], params: dict
) -> np.ndarray:
    """Concatenated LOO prediction vector across all structures."""
    vectors = []
    for name in sorted(structures.keys()):
        vectors.append(loo_prediction_vector(model, structures[name], params))
    return np.concatenate(vectors)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------------------------------------------------------------------------
# Phase 1b: Competition-based sweep
# Generate synthetic data from ground truth, score all models against it.
# ---------------------------------------------------------------------------

N_SUBJECTS = 30  # Matching _synthetic_runner default


def generate_synthetic_data(
    model, structure: dict, params: dict, n_subjects: int = N_SUBJECTS, seed_str: str = "default"
) -> dict[str, float]:
    """Generate noisy synthetic accuracy data from a ground-truth model.

    Matches _synthetic_runner logic: LOO predictions → binomial noise.
    Returns {item_i: noisy_accuracy} dict.
    """
    stimuli = structure["stimuli"]
    labels = structure["labels"]
    n = len(labels)

    # Deterministic seed from seed_str
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)

    data = {}
    for i in range(n):
        loo_stimuli = np.delete(stimuli, i, axis=0)
        loo_labels = np.delete(labels, i)
        result = model.predict(stimuli[i], loo_stimuli, loo_labels, **params)
        p_correct = result["probabilities"].get(int(labels[i]), 0.5)
        p_correct = np.clip(p_correct, 0.01, 0.99)
        n_correct = rng.binomial(n_subjects, p_correct)
        data[f"item_{i}"] = n_correct / n_subjects
    return data


def score_model_against_data(
    model, structure: dict, params: dict, data: dict[str, float]
) -> float:
    """Score a model's LOO predictions against synthetic data. Returns RMSE."""
    stimuli = structure["stimuli"]
    labels = structure["labels"]
    n = len(labels)

    errors = []
    for i in range(n):
        loo_stimuli = np.delete(stimuli, i, axis=0)
        loo_labels = np.delete(labels, i)
        result = model.predict(stimuli[i], loo_stimuli, loo_labels, **params)
        p_correct = result["probabilities"].get(int(labels[i]), 0.5)
        observed = data[f"item_{i}"]
        errors.append((p_correct - observed) ** 2)
    return float(np.sqrt(np.mean(errors)))


def run_competition(
    structures: dict[str, dict],
    true_model_name: str,
    true_model,
    true_params: dict,
    models: dict[str, tuple],  # {name: (model, params)}
    misspec_params: dict,  # Override params for the correct model's agent
    label: str = "",
) -> dict:
    """Run a full competition: generate data from true model, score all models.

    Returns dict with RMSE per model, winner, and gap.
    """
    # Generate synthetic data across all structures
    all_data = {}
    for sname in sorted(structures.keys()):
        seed_str = f"0_{sname}_baseline"  # cycle 0, baseline condition
        all_data[sname] = generate_synthetic_data(
            true_model, structures[sname], true_params, seed_str=seed_str
        )

    # Score each model
    rmses = {}
    for mname, (model, default_params) in models.items():
        # Correct model uses misspecified params; others use defaults
        if mname == true_model_name:
            params = misspec_params
        else:
            params = default_params

        struct_rmses = []
        for sname in sorted(structures.keys()):
            r = score_model_against_data(model, structures[sname], params, all_data[sname])
            struct_rmses.append(r)
        rmses[mname] = float(np.mean(struct_rmses))

    winner = min(rmses, key=rmses.get)
    best_competitor = min(
        (k for k in rmses if k != true_model_name), key=rmses.get
    )
    gap = (rmses[best_competitor] - rmses[true_model_name]) / rmses[best_competitor] * 100

    return {
        "rmses": rmses,
        "winner": winner,
        "correct_wins": winner == true_model_name,
        "gap": gap,
        "true_rmse": rmses[true_model_name],
        "best_competitor": best_competitor,
        "competitor_rmse": rmses[best_competitor],
    }


def run_phase_1b(structures: dict[str, dict]):
    """Phase 1b: Competition-based sweep."""
    print()
    print("=" * 70)
    print("PHASE 1b: COMPETITION-BASED SWEEP")
    print("Generate synthetic data from each GT, score with misspecified correct model")
    print("=" * 70)
    print()

    gcm = GCM()
    sustain = SUSTAIN()
    rulex = RULEX()

    models = {
        "GCM": (gcm, GROUND_TRUTH_PARAMS["GCM"]),
        "SUSTAIN": (sustain, GROUND_TRUTH_PARAMS["SUSTAIN"]),
        "RULEX": (rulex, GROUND_TRUTH_PARAMS["RULEX"]),
    }

    # --- Baseline: all models at correct params ---
    print("--- Baseline (all models at correct params) ---")
    print(f"{'GT':>8}  {'GCM_rmse':>8}  {'SUS_rmse':>8}  {'RUL_rmse':>8}  {'winner':>10}  {'gap':>6}")
    print("-" * 60)

    for gt_name, (gt_model, gt_params) in models.items():
        result = run_competition(
            structures, gt_name, gt_model, gt_params, models,
            misspec_params=gt_params,  # No misspecification
        )
        r = result["rmses"]
        print(f"{gt_name:>8}  {r['GCM']:>8.4f}  {r['SUSTAIN']:>8.4f}  {r['RULEX']:>8.4f}  "
              f"{result['winner']:>10}  {result['gap']:>+6.1f}%")

    print()

    # --- GCM misspecification sweep ---
    print("--- GCM ground truth: sweep c (misspecify GCM agent) ---")
    print(f"{'c':>6}  {'GCM_rmse':>8}  {'SUS_rmse':>8}  {'RUL_rmse':>8}  {'winner':>10}  {'gap':>7}  note")
    print("-" * 75)

    gcm_comp_results = []
    for c in GCM_C_GRID:
        misspec = {**GROUND_TRUTH_PARAMS["GCM"], "c": c}
        result = run_competition(
            structures, "GCM", gcm, GROUND_TRUTH_PARAMS["GCM"], models,
            misspec_params=misspec,
        )
        r = result["rmses"]
        marker = " <-- GT" if c == GROUND_TRUTH_PARAMS["GCM"]["c"] else ""
        win_marker = " ** FLIPPED **" if not result["correct_wins"] else ""
        gcm_comp_results.append({"c": c, **result})
        print(f"{c:>6.1f}  {r['GCM']:>8.4f}  {r['SUSTAIN']:>8.4f}  {r['RULEX']:>8.4f}  "
              f"{result['winner']:>10}  {result['gap']:>+7.1f}%{marker}{win_marker}")

    print()

    # --- SUSTAIN misspecification sweep ---
    print("--- SUSTAIN ground truth: sweep r × eta (misspecify SUSTAIN agent) ---")
    print(f"{'r':>6}  {'eta':>6}  {'GCM_rmse':>8}  {'SUS_rmse':>8}  {'RUL_rmse':>8}  {'winner':>12}  {'gap':>7}  note")
    print("-" * 85)

    sustain_comp_results = []
    for grid_point in SUSTAIN_GRID:
        misspec = {**GROUND_TRUTH_PARAMS["SUSTAIN"], **grid_point}
        result = run_competition(
            structures, "SUSTAIN", sustain, GROUND_TRUTH_PARAMS["SUSTAIN"], models,
            misspec_params=misspec,
        )
        r = result["rmses"]
        is_gt = (grid_point["r"] == 9.01 and grid_point["eta"] == 0.092)
        marker = " <-- GT" if is_gt else ""
        win_marker = " ** FLIPPED **" if not result["correct_wins"] else ""
        sustain_comp_results.append({**grid_point, **result})
        print(f"{grid_point['r']:>6.2f}  {grid_point['eta']:>6.3f}  {r['GCM']:>8.4f}  {r['SUSTAIN']:>8.4f}  "
              f"{r['RULEX']:>8.4f}  {result['winner']:>12}  {result['gap']:>+7.1f}%{marker}{win_marker}")

    print()

    # --- RULEX misspecification sweep ---
    print("--- RULEX ground truth: sweep error_tolerance × p_single (misspecify RULEX agent) ---")
    print(f"{'err_tol':>7}  {'p_single':>8}  {'GCM_rmse':>8}  {'SUS_rmse':>8}  {'RUL_rmse':>8}  {'winner':>10}  {'gap':>7}  note")
    print("-" * 85)

    rulex_comp_results = []
    for grid_point in RULEX_GRID:
        misspec = {**GROUND_TRUTH_PARAMS["RULEX"], **grid_point}
        result = run_competition(
            structures, "RULEX", rulex, GROUND_TRUTH_PARAMS["RULEX"], models,
            misspec_params=misspec,
        )
        r = result["rmses"]
        is_gt = (grid_point["error_tolerance"] == 0.1 and grid_point["p_single"] == 0.5)
        marker = " <-- GT" if is_gt else ""
        win_marker = " ** FLIPPED **" if not result["correct_wins"] else ""
        rulex_comp_results.append({**grid_point, **result})
        print(f"{grid_point['error_tolerance']:>7.2f}  {grid_point['p_single']:>8.1f}  {r['GCM']:>8.4f}  "
              f"{r['SUSTAIN']:>8.4f}  {r['RULEX']:>8.4f}  {result['winner']:>10}  "
              f"{result['gap']:>+7.1f}%{marker}{win_marker}")

    print()

    # --- Summary ---
    print("=" * 70)
    print("PHASE 1b SUMMARY")
    print("=" * 70)
    print()

    # GCM: find settings with narrowest gap (still winning)
    gcm_winning = [r for r in gcm_comp_results if r["correct_wins"]]
    gcm_flipped = [r for r in gcm_comp_results if not r["correct_wins"]]
    if gcm_winning:
        narrowest = min(gcm_winning, key=lambda r: r["gap"])
        print(f"  GCM narrowest gap (still wins): c={narrowest['c']:.1f}, "
              f"gap={narrowest['gap']:+.1f}%, RMSE={narrowest['true_rmse']:.4f}")
    if gcm_flipped:
        print(f"  GCM FLIPPED at: {', '.join(f'c={r['c']:.1f}' for r in gcm_flipped)}")

    print()

    sustain_winning = [r for r in sustain_comp_results if r["correct_wins"]]
    sustain_flipped = [r for r in sustain_comp_results if not r["correct_wins"]]
    if sustain_winning:
        narrowest = min(sustain_winning, key=lambda r: r["gap"])
        print(f"  SUSTAIN narrowest gap (still wins): r={narrowest['r']:.2f}, "
              f"eta={narrowest['eta']:.3f}, gap={narrowest['gap']:+.1f}%, "
              f"RMSE={narrowest['true_rmse']:.4f}")
    if sustain_flipped:
        print(f"  SUSTAIN FLIPPED at: {', '.join(f'r={r['r']:.2f}/eta={r['eta']:.3f}' for r in sustain_flipped)}")

    print()

    rulex_winning = [r for r in rulex_comp_results if r["correct_wins"]]
    rulex_flipped = [r for r in rulex_comp_results if not r["correct_wins"]]
    if rulex_winning:
        narrowest = min(rulex_winning, key=lambda r: r["gap"])
        print(f"  RULEX narrowest gap (still wins): err_tol={narrowest['error_tolerance']:.2f}, "
              f"p_single={narrowest['p_single']:.1f}, gap={narrowest['gap']:+.1f}%, "
              f"RMSE={narrowest['true_rmse']:.4f}")
    if rulex_flipped:
        print(f"  RULEX FLIPPED at: {', '.join(f'et={r['error_tolerance']:.2f}/ps={r['p_single']:.1f}' for r in rulex_flipped)}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_phase_1a(structures: dict[str, dict]):
    """Phase 1a: Mimicry sweep (original analysis)."""
    struct_names = sorted(structures.keys())
    total_items = sum(len(structures[s]["labels"]) for s in struct_names)
    print(f"Structures: {struct_names}")
    print(f"Total items across structures: {total_items}")
    print()

    gcm = GCM()
    sustain = SUSTAIN()
    rulex = RULEX()

    # --- Step 1: Compute ground-truth profiles ---
    print("=" * 70)
    print("Computing ground-truth prediction profiles...")
    print("=" * 70)
    t0 = time.time()

    gt_profiles = {}
    for model_name, model, params in [
        ("GCM", gcm, GROUND_TRUTH_PARAMS["GCM"]),
        ("SUSTAIN", sustain, GROUND_TRUTH_PARAMS["SUSTAIN"]),
        ("RULEX", rulex, GROUND_TRUTH_PARAMS["RULEX"]),
    ]:
        gt_profiles[model_name] = prediction_profile(model, structures, params)
        mean_acc = np.mean(gt_profiles[model_name])
        print(f"  {model_name} (ground truth): mean P(correct) = {mean_acc:.3f}")

    print(f"  ({time.time() - t0:.1f}s)")
    print()

    # --- Step 2: Sweep each model and compare to all ground truths ---

    # GCM sweep
    print("=" * 70)
    print("GCM parameter sweep: c")
    print("=" * 70)
    print(f"{'c':>6}  {'mean_acc':>8}  {'vs GCM_gt':>9}  {'vs SUS_gt':>9}  {'vs RUL_gt':>9}  closest")
    print("-" * 70)

    gcm_results = []
    for c in GCM_C_GRID:
        params = {**GROUND_TRUTH_PARAMS["GCM"], "c": c}
        profile = prediction_profile(gcm, structures, params)
        mean_acc = np.mean(profile)
        d_gcm = rmse(profile, gt_profiles["GCM"])
        d_sus = rmse(profile, gt_profiles["SUSTAIN"])
        d_rul = rmse(profile, gt_profiles["RULEX"])

        distances = {"GCM": d_gcm, "SUSTAIN": d_sus, "RULEX": d_rul}
        # Closest non-self model
        competitors = {k: v for k, v in distances.items() if k != "GCM"}
        closest = min(competitors, key=competitors.get)
        closest_d = competitors[closest]

        gcm_results.append({
            "c": c, "mean_acc": mean_acc,
            "d_gcm": d_gcm, "d_sus": d_sus, "d_rul": d_rul,
            "closest": closest, "closest_d": closest_d,
        })
        marker = " <-- GT" if c == GROUND_TRUTH_PARAMS["GCM"]["c"] else ""
        print(f"{c:>6.1f}  {mean_acc:>8.3f}  {d_gcm:>9.4f}  {d_sus:>9.4f}  {d_rul:>9.4f}  {closest} ({closest_d:.4f}){marker}")

    print()

    # SUSTAIN sweep
    print("=" * 70)
    print("SUSTAIN parameter sweep: r × eta")
    print("=" * 70)
    print(f"{'r':>6}  {'eta':>6}  {'mean_acc':>8}  {'vs GCM_gt':>9}  {'vs SUS_gt':>9}  {'vs RUL_gt':>9}  closest")
    print("-" * 70)

    sustain_results = []
    for grid_point in SUSTAIN_GRID:
        params = {**GROUND_TRUTH_PARAMS["SUSTAIN"], **grid_point}
        profile = prediction_profile(sustain, structures, params)
        mean_acc = np.mean(profile)
        d_gcm = rmse(profile, gt_profiles["GCM"])
        d_sus = rmse(profile, gt_profiles["SUSTAIN"])
        d_rul = rmse(profile, gt_profiles["RULEX"])

        distances = {"GCM": d_gcm, "SUSTAIN": d_sus, "RULEX": d_rul}
        competitors = {k: v for k, v in distances.items() if k != "SUSTAIN"}
        closest = min(competitors, key=competitors.get)
        closest_d = competitors[closest]

        is_gt = (grid_point["r"] == 9.01 and grid_point["eta"] == 0.092)
        marker = " <-- GT" if is_gt else ""
        sustain_results.append({
            **grid_point, "mean_acc": mean_acc,
            "d_gcm": d_gcm, "d_sus": d_sus, "d_rul": d_rul,
            "closest": closest, "closest_d": closest_d,
        })
        print(f"{grid_point['r']:>6.2f}  {grid_point['eta']:>6.3f}  {mean_acc:>8.3f}  {d_gcm:>9.4f}  {d_sus:>9.4f}  {d_rul:>9.4f}  {closest} ({closest_d:.4f}){marker}")

    print()

    # RULEX sweep
    print("=" * 70)
    print("RULEX parameter sweep: error_tolerance × p_single")
    print("=" * 70)
    print(f"{'err_tol':>7}  {'p_single':>8}  {'mean_acc':>8}  {'vs GCM_gt':>9}  {'vs SUS_gt':>9}  {'vs RUL_gt':>9}  closest")
    print("-" * 70)

    rulex_results = []
    for grid_point in RULEX_GRID:
        params = {**GROUND_TRUTH_PARAMS["RULEX"], **grid_point}
        profile = prediction_profile(rulex, structures, params)
        mean_acc = np.mean(profile)
        d_gcm = rmse(profile, gt_profiles["GCM"])
        d_sus = rmse(profile, gt_profiles["SUSTAIN"])
        d_rul = rmse(profile, gt_profiles["RULEX"])

        distances = {"GCM": d_gcm, "SUSTAIN": d_sus, "RULEX": d_rul}
        competitors = {k: v for k, v in distances.items() if k != "RULEX"}
        closest = min(competitors, key=competitors.get)
        closest_d = competitors[closest]

        is_gt = (grid_point["error_tolerance"] == 0.1 and grid_point["p_single"] == 0.5)
        marker = " <-- GT" if is_gt else ""
        rulex_results.append({
            **grid_point, "mean_acc": mean_acc,
            "d_gcm": d_gcm, "d_sus": d_sus, "d_rul": d_rul,
            "closest": closest, "closest_d": closest_d,
        })
        print(f"{grid_point['error_tolerance']:>7.2f}  {grid_point['p_single']:>8.1f}  {mean_acc:>8.3f}  {d_gcm:>9.4f}  {d_sus:>9.4f}  {d_rul:>9.4f}  {closest} ({closest_d:.4f}){marker}")

    print()

    # --- Step 3: Identify best mimicry settings ---
    print("=" * 70)
    print("MIMICRY SUMMARY — Settings where each model most resembles a competitor")
    print("=" * 70)
    print()

    # GCM: find c where GCM is closest to each competitor
    for target in ["SUSTAIN", "RULEX"]:
        key = f"d_{target[:3].lower()}"
        best = min(gcm_results, key=lambda r: r[key])
        print(f"  GCM → {target}: c={best['c']:.1f}, RMSE={best[key]:.4f} "
              f"(vs self at GT: {best['d_gcm']:.4f})")

    print()

    # SUSTAIN: find (r, eta) where SUSTAIN is closest to each competitor
    for target in ["GCM", "RULEX"]:
        key = f"d_{target[:3].lower()}"
        best = min(sustain_results, key=lambda r: r[key])
        print(f"  SUSTAIN → {target}: r={best['r']:.2f}, eta={best['eta']:.3f}, "
              f"RMSE={best[key]:.4f} (vs self at GT: {best['d_sus']:.4f})")

    print()

    # RULEX: find (error_tolerance, p_single) where RULEX is closest to each competitor
    for target in ["GCM", "SUSTAIN"]:
        key = f"d_{target[:3].lower()}"
        best = min(rulex_results, key=lambda r: r[key])
        print(f"  RULEX → {target}: err_tol={best['error_tolerance']:.2f}, "
              f"p_single={best['p_single']:.1f}, RMSE={best[key]:.4f} "
              f"(vs self at GT: {best['d_rul']:.4f})")

    print()
    print("=" * 70)
    print("RECOMMENDED MISSPECIFICATION LEVELS FOR PHASE 2")
    print("=" * 70)
    print()
    print("Choose settings where the misspecified model is closer to a competitor")
    print("than to its own ground truth — these are the mimicry regions where")
    print("parameter recovery is genuinely hard.")
    print()

    # For each model, find settings where distance to a competitor < distance to self GT
    print("GCM mimicry candidates (d_competitor < d_self_gt):")
    for r in gcm_results:
        if r["c"] == GROUND_TRUTH_PARAMS["GCM"]["c"]:
            continue
        if r["closest_d"] < r["d_gcm"]:
            print(f"  c={r['c']:.1f}: closer to {r['closest']} ({r['closest_d']:.4f}) "
                  f"than to GCM_gt ({r['d_gcm']:.4f})")

    print()
    print("SUSTAIN mimicry candidates (d_competitor < d_self_gt):")
    for r in sustain_results:
        if r["r"] == 9.01 and r["eta"] == 0.092:
            continue
        if r["closest_d"] < r["d_sus"]:
            print(f"  r={r['r']:.2f}, eta={r['eta']:.3f}: closer to {r['closest']} "
                  f"({r['closest_d']:.4f}) than to SUSTAIN_gt ({r['d_sus']:.4f})")

    print()
    print("RULEX mimicry candidates (d_competitor < d_self_gt):")
    for r in rulex_results:
        if r["error_tolerance"] == 0.1 and r["p_single"] == 0.5:
            continue
        if r["closest_d"] < r["d_rul"]:
            print(f"  err_tol={r['error_tolerance']:.2f}, p_single={r['p_single']:.1f}: "
                  f"closer to {r['closest']} ({r['closest_d']:.4f}) "
                  f"than to RULEX_gt ({r['d_rul']:.4f})")

    print()


def main():
    parser = argparse.ArgumentParser(description="M15 Phase 1 parameter sweep")
    parser.add_argument("--phase", choices=["1a", "1b"], default=None,
                        help="Run specific phase (default: both)")
    args = parser.parse_args()

    structures = load_structures()

    if args.phase is None or args.phase == "1a":
        run_phase_1a(structures)
    if args.phase is None or args.phase == "1b":
        run_phase_1b(structures)


if __name__ == "__main__":
    main()
