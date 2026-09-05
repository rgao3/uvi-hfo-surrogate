"""Build geochemical-consistency and reusable-predictor artifacts."""

from __future__ import annotations

import json
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
W2 = ROOT / "predictive_modeling"
WORK = ROOT / "consistency_and_benchmark"
ART = ROOT / "uvi_surrogate" / "artifacts"
DATA = ROOT / "data" / "U_HFO_ML_Dataset_Final.csv"

BASE = ["Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s", "Hfo_w"]
LOG_COLS = ["U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s", "Hfo_w"]
FEATURES = BASE + ["log10_" + c for c in LOG_COLS]
SPECIES = [
    "Hfo_sOUO2+", "(Hfo_sO)2UO2", "(Hfo_sO)2UO2CO3-2",
    "Hfo_wOUO2+", "(Hfo_wO)2UO2", "(Hfo_wO)2UO2CO3-2",
]
RNG = 42


def add_features(frame):
    d = frame.copy()
    for c in LOG_COLS:
        d["log10_" + c] = np.log10(d[c].clip(lower=1e-12))
    return d


def finite_sample_q(scores, alpha=0.10):
    n = len(scores)
    return float(np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"))


def fit_or_load(path, X, y, params):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    model = XGBRegressor(tree_method="hist", n_jobs=2, random_state=RNG, verbosity=0, **params)
    model.fit(X, y)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return model


def exact_slice(df, settings, vary):
    mask = np.ones(len(df), dtype=bool)
    for c, v in settings.items():
        if c != vary:
            mask &= np.isclose(df[c].to_numpy(float), v, rtol=1e-7, atol=1e-14)
    return df.loc[mask].sort_values(vary).drop_duplicates(vary)


def main():
    WORK.mkdir(exist_ok=True)
    ART.mkdir(exist_ok=True)
    df = add_features(pd.read_csv(DATA))
    X = df[FEATURES].to_numpy(dtype=np.float32)
    splits = np.load(W2 / "conformal_splits.npz")
    tr, cal, test = splits["tr"], splits["cal"], splits["test"]
    best = json.load(open(W2 / "final_xgb.json"))

    conformal = {}
    point_models = {}
    for target, tag in [("Ads_%", "Ads_pct"), ("logKd", "logKd")]:
        model_path = WORK / f"xgb_{tag}_conformal.pkl"
        model = fit_or_load(model_path, X[tr], df[target].to_numpy(np.float32)[tr], best[target]["best_params"])
        point_models[target] = model
        cal_scores = np.abs(df[target].to_numpy(float)[cal] - model.predict(X[cal]))
        pH_cal = df["Input_pH"].to_numpy(float)[cal]
        conformal[target] = {
            "<=5": finite_sample_q(cal_scores[pH_cal <= 5]),
            "6-7": finite_sample_q(cal_scores[(pH_cal >= 6) & (pH_cal <= 7)]),
            ">=8": finite_sample_q(cal_scores[pH_cal >= 8]),
        }
        shutil.copy2(model_path, ART / model_path.name)

    species_sum = df[SPECIES].sum(axis=1).to_numpy(float)
    fractions = np.divide(df[SPECIES].to_numpy(float), species_sum[:, None],
                          out=np.zeros((len(df), len(SPECIES))), where=species_sum[:, None] > 0)
    species_path = WORK / "xgb_surface_fractions.pkl"
    species_params = dict(n_estimators=350, max_depth=7, learning_rate=0.04,
                          subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
                          reg_lambda=2.0, objective="reg:squarederror",
                          multi_strategy="one_output_per_tree")
    species_model = fit_or_load(species_path, X[tr], fractions[tr].astype(np.float32), species_params)
    shutil.copy2(species_path, ART / species_path.name)

    raw = np.clip(species_model.predict(X[test]), 0.0, None)
    pred_frac = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-15)
    true_frac = fractions[test]
    dominant_true = np.argmax(true_frac, axis=1)
    dominant_pred = np.argmax(pred_frac, axis=1)
    per_species = []
    for j, name in enumerate(SPECIES):
        spread = float(np.std(true_frac[:, j]))
        per_species.append({
            "species": name,
            "fraction_MAE": float(mean_absolute_error(true_frac[:, j], pred_frac[:, j])),
            "mean_fraction": float(np.mean(true_frac[:, j])),
            "max_fraction": float(np.max(true_frac[:, j])),
            "fraction_R2": float(r2_score(true_frac[:, j], pred_frac[:, j])) if spread > 1e-6 else None,
        })

    mu = X[tr].mean(axis=0).astype(float)
    sd = (X[tr].std(axis=0) + 1e-9).astype(float)
    reference = ((X[tr].astype(float) - mu) / sd).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=6).fit(reference)
    dtr = nn.kneighbors(reference, return_distance=True)[0][:, 1:].mean(axis=1)
    ad_threshold = float(np.quantile(dtr, 0.95))
    np.savez_compressed(ART / "ad_reference.npz", mu=mu, sd=sd, reference=reference)

    # Sampled levels of each design axis, needed by the resolution-aware criterion.
    # Hfo_w is locked to Hfo_s at the Dzombak & Morel 1:40 ratio, so it is recorded as a
    # coupled feature rather than as an independent axis and is moved with its driver.
    design_axes = ["Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s"]
    design_levels = {a: [float(v) for v in np.sort(df[a].unique())] for a in design_axes}
    ratio = float(np.median(df["Hfo_w"].to_numpy() / df["Hfo_s"].to_numpy()))

    manifest = {
        "version": "0.2.0",
        "base_features": BASE,
        "log_columns": LOG_COLS,
        "features": FEATURES,
        "surface_species": SPECIES,
        "conformal_90": conformal,
        "ad_threshold_p95": ad_threshold,
        "design_levels": design_levels,
        "coupled_features": {"Hfo_w": ["Hfo_s", ratio]},
        "resolution_criterion": (
            "Resolution score = max over design axes of |f(x with axis at upper bracketing "
            "level) - f(x with axis at lower bracketing level)|, in adsorption points. "
            "Flagged when the score exceeds the pH-conditional conformal 90% half-width for "
            "that query, i.e. when grid resolution contributes more uncertainty than the "
            "calibrated statistical interval."),
        "training_scope": "86,375 converged PHREEQC simulations; calibrated on the fixed training/test split",
    }
    with open(ART / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    sys.path.insert(0, str(MODELING))
    from uvi_surrogate import SurrogatePredictor
    predictor = SurrogatePredictor(ART)

    mid = {
        "Input_pH": 7.0, "U_initial": 1.389495494373136e-06,
        "Carbonate": 1.930697728883e-04, "NaCl": 1.58489319246111e-02,
        "Ca": 1e-3, "Hfo_s": 3.1622776601683795e-05,
        "Hfo_w": 1.2649110640673e-03,
    }
    definitions = [
        ("carbonate_pH9", {**mid, "Input_pH": 9.0}, "Carbonate"),
        ("HFO_loading", {k: v for k, v in mid.items() if k != "Hfo_w"}, "Hfo_s"),
        ("pH_edge", mid, "Input_pH"),
        ("Ca_low_carbonate", {**mid, "Carbonate": 1e-5}, "Ca"),
        ("Ca_high_carbonate", {**mid, "Carbonate": 1e-2}, "Ca"),
    ]
    expected = {
        "carbonate_pH9": "nonincreasing",
        "HFO_loading": "nondecreasing",
        "pH_edge": "shape",
        "Ca_low_carbonate": "small-effect",
        "Ca_high_carbonate": "nonincreasing",
    }
    trend_rows, trend_metrics = [], {}
    for test_name, settings, vary in definitions:
        if vary == "Hfo_s":
            sl = exact_slice(df, settings, "Hfo_s")
            sl = sl[np.isclose(sl["Hfo_w"] / sl["Hfo_s"], 40.0, rtol=1e-6)]
        else:
            sl = exact_slice(df, settings, vary)
        pred = predictor.predict(sl[BASE])
        for k, (_, row) in enumerate(sl.iterrows()):
            trend_rows.append({
                "test": test_name, "varied_feature": vary, "x": float(row[vary]),
                "PHREEQC_Ads_%": float(row["Ads_%"]), "surrogate_Ads_%": float(pred.iloc[k]["Ads_%"]),
                "PHREEQC_logKd": float(row["logKd"]), "surrogate_logKd": float(pred.iloc[k]["logKd"]),
            })
        pred_range = float(pred["Ads_%"].max() - pred["Ads_%"].min())
        phreeqc_range = float(sl["Ads_%"].max() - sl["Ads_%"].min())
        pred_rho = float(spearmanr(sl[vary], pred["Ads_%"]).statistic) if pred_range > 1e-10 else None
        observed_rho = float(spearmanr(sl[vary], sl["Ads_%"]).statistic) if phreeqc_range > 1e-10 else None
        diffs = np.diff(pred["Ads_%"].to_numpy(float))
        if expected[test_name] == "nonincreasing":
            violation = float(max(0.0, diffs.max(initial=0.0)))
            behavior_pass = violation <= 0.10
        elif expected[test_name] == "nondecreasing":
            violation = float(max(0.0, (-diffs).max(initial=0.0)))
            behavior_pass = violation <= 0.10
        elif expected[test_name] == "small-effect":
            violation = pred_range
            behavior_pass = pred_range <= 0.10
        else:
            violation = None
            behavior_pass = r2_score(sl["Ads_%"], pred["Ads_%"]) >= 0.90
        trend_metrics[test_name] = {
            "n": int(len(sl)),
            "Ads_%_MAE": float(mean_absolute_error(sl["Ads_%"], pred["Ads_%"])),
            "Ads_%_curve_R2": float(r2_score(sl["Ads_%"], pred["Ads_%"])) if phreeqc_range >= 1e-3 else None,
            "Ads_%_range_surrogate": pred_range,
            "Ads_%_range_PHREEQC": phreeqc_range,
            "Ads_%_Spearman": pred_rho,
            "PHREEQC_Ads_%_Spearman": observed_rho,
            "expected_behavior": expected[test_name],
            "max_direction_violation_Ads_%": violation,
            "behavior_pass": bool(behavior_pass),
        }
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(WORK / "trend_data.csv", index=False)

    surface_settings = {**mid, "Carbonate": 1.930697728883e-04}
    surface_slice = exact_slice(df, surface_settings, "Input_pH")
    surface_pred = predictor.predict(surface_slice[BASE])
    surface_out = surface_slice[["Input_pH"] + SPECIES + ["U_ads"]].copy()
    for name in SPECIES:
        surface_out["pred_" + name] = surface_pred[name].to_numpy()
    surface_out.to_csv(WORK / "surface_species_pH.csv", index=False)

    closure_all = np.abs(df[SPECIES].sum(axis=1) - df["U_ads"])
    package_test = predictor.predict(df.iloc[test[:5000]][BASE])

    # PHREEQC reference timing.
    # The recorded wall-clock times come from the batch_automation sweep, which is a
    # DIFFERENT design (22,500 conditions) from the 86,400-condition IPhreeqc design used
    # for training. The per-case cost must therefore be normalised by the number of
    # simulations actually contained in those .out files, not by 86,400. Each sweep file
    # holds one reaction step per simulated condition, so the count is read from the files
    # rather than hard-coded.
    out_files = sorted((ROOT / "batch_automation" / "phreeqc_out").glob("sweep_*.out"))
    phreeqc_seconds = []
    phreeqc_cases = 0
    for path in out_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"End of Run after\s+([0-9.]+)\s+Seconds", text[-3000:])
        if match:
            phreeqc_seconds.append(float(match.group(1)))
            phreeqc_cases += len(re.findall(r"^Reaction step", text, flags=re.M))
    n_benchmark = min(5000, len(test))
    bench_inputs = df.iloc[test[:n_benchmark]][BASE]
    predictor.predict(bench_inputs.iloc[:10])
    timings = []
    for _ in range(3):
        t0 = time.perf_counter(); predictor.predict(bench_inputs); timings.append(time.perf_counter() - t0)
    surrogate_seconds = float(np.median(timings))
    X_bench = add_features(bench_inputs)[FEATURES].to_numpy(np.float32)
    core_timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        point_models["Ads_%"].predict(X_bench); point_models["logKd"].predict(X_bench)
        core_timings.append(time.perf_counter() - t0)
    core_seconds = float(np.median(core_timings))
    phreeqc_per_case = (float(sum(phreeqc_seconds) / phreeqc_cases)
                        if phreeqc_seconds and phreeqc_cases else float("nan"))
    surrogate_per_case = surrogate_seconds / n_benchmark

    results = {
        "dataset_rows": int(len(df)),
        "phreeqc_mass_balance_max_abs_mol": float(closure_all.max()),
        "surface_fraction_metrics": per_species,
        "dominant_surface_species_accuracy": float(accuracy_score(dominant_true, dominant_pred)),
        "mean_surface_fraction_MAE": float(np.mean(np.abs(pred_frac - true_frac))),
        "predicted_closure_max_abs_mol_5000": float(package_test["surface_closure_error"].max()),
        "consistency_pass_rate_5000": float((package_test["consistency_status"] == "pass").mean()),
        "trend_metrics": trend_metrics,
        "ad_threshold_p95": ad_threshold,
        "benchmark": {
            "phreeqc_sweeps": len(phreeqc_seconds),
            "phreeqc_reference_cases": int(phreeqc_cases),
            "phreeqc_total_seconds_recorded": float(sum(phreeqc_seconds)),
            "phreeqc_seconds_per_case": phreeqc_per_case,
            "surrogate_batch_size": n_benchmark,
            "surrogate_total_seconds_core_predictions": core_seconds,
            "core_seconds_per_case": core_seconds / n_benchmark,
            "estimated_core_speedup": float(phreeqc_per_case / (core_seconds / n_benchmark)),
            "surrogate_total_seconds_full_API": surrogate_seconds,
            "surrogate_seconds_per_case": surrogate_per_case,
            "estimated_speedup": float(phreeqc_per_case / surrogate_per_case),
        },
    }
    with open(WORK / "week4_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
