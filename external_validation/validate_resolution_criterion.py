"""Does the resolution-aware criterion predict error where the distance-based AD cannot?

Design of the test
------------------
The question cannot be answered on the released model, because the true response between
its sampled levels is unknown without new simulations. It can be answered by coarsening
the existing design: train on every second level of one axis, then predict the levels
that were removed. Those removed levels have known PHREEQC values, so the error at a
genuinely under-resolved location is measurable, and both reliability criteria can be
scored against it.

This uses only the existing 86,375 simulations. No external data and no new runs.

Two axes are tested: pH (removing 4, 6, 8, 10) and Hfo_s (removing levels 2 and 4 of 5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
MODELING = HERE.parent
sys.path.insert(0, str(MODELING))
from uvi_surrogate.resolution import ResolutionScorer  # noqa: E402

DATA = MODELING / "data" / "U_HFO_ML_Dataset_Final.csv"
OUT = HERE / "resolution_criterion_validation.csv"
CONSTANTS = HERE / "resolution_criterion_constants.json"

BASE = ["Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s", "Hfo_w"]
LOG = ["U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s", "Hfo_w"]
FEATURES = BASE + ["log10_" + c for c in LOG]
PARAMS = dict(n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.85,
              colsample_bytree=0.85, min_child_weight=5, reg_lambda=2.0,
              tree_method="hist", n_jobs=4, random_state=42, verbosity=0)
SCORED_AXES = ["Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s"]
COUPLED = {"Hfo_w": ("Hfo_s", 40.0)}
RNG = np.random.RandomState(42)


def add_logs(frame):
    out = frame.copy()
    for c in LOG:
        out["log10_" + c] = np.log10(out[c].clip(lower=1e-12))
    return out


def conformal_half_width(residuals, alpha=0.10):
    n = len(residuals)
    return float(np.quantile(residuals, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"))


def run_case(df, axis_name, keep_levels, drop_levels, sample=15000):
    col = df[axis_name].to_numpy()
    train = np.isin(col, keep_levels)
    test = np.isin(col, drop_levels)

    X = df[FEATURES].to_numpy(np.float32)
    y = df["Ads_%"].to_numpy(np.float32)

    tr_idx = np.flatnonzero(train)
    RNG.shuffle(tr_idx)
    cut = int(0.75 * len(tr_idx))
    fit_idx, cal_idx = tr_idx[:cut], tr_idx[cut:]

    model = XGBRegressor(**PARAMS).fit(X[fit_idx], y[fit_idx])

    def predict(frame):
        return model.predict(add_logs(frame)[FEATURES].to_numpy(np.float32))

    half_width = conformal_half_width(np.abs(y[cal_idx] - model.predict(X[cal_idx])))

    te_idx = np.flatnonzero(test)
    if len(te_idx) > sample:
        te_idx = RNG.choice(te_idx, sample, replace=False)
    query = df.iloc[te_idx][BASE].reset_index(drop=True)
    error = np.abs(model.predict(X[te_idx]) - y[te_idx])

    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0) + 1e-9
    ref = (X[fit_idx].astype(float) - mu) / sd
    nn = NearestNeighbors(n_neighbors=6).fit(ref)
    d_train = nn.kneighbors(ref, n_neighbors=6, return_distance=True)[0][:, 1:].mean(1)
    ad_threshold = float(np.quantile(d_train, 0.95))
    ad = nn.kneighbors((X[te_idx].astype(float) - mu) / sd,
                       n_neighbors=5, return_distance=True)[0].mean(1)

    levels = {a: np.unique(df.loc[train, a].to_numpy()) for a in SCORED_AXES}
    res_score, res_axis, _ = ResolutionScorer(levels, COUPLED).score(query, predict)

    flagged = res_score > half_width
    summary = dict(
        axis=axis_name, n=len(te_idx), MAE=error.mean(),
        conformal_half_width=half_width, ad_threshold=ad_threshold,
        ad_rho=spearmanr(ad, error).statistic,
        res_rho=spearmanr(res_score, error).statistic,
        ad_flag_rate=(ad > ad_threshold).mean(),
        res_flag_rate=flagged.mean(),
        mae_res_flagged=error[flagged].mean() if flagged.any() else np.nan,
        mae_res_clear=error[~flagged].mean() if (~flagged).any() else np.nan,
        top_axis=pd.Series(res_axis).value_counts().idxmax(),
    )
    detail = pd.DataFrame(dict(coarsened_axis=axis_name, error=error, ad_distance=ad,
                               resolution_score=res_score, responsible_axis=res_axis))
    return summary, detail


def main():
    df = add_logs(pd.read_csv(DATA))
    ph = np.unique(df["Input_pH"])
    hfo = np.unique(df["Hfo_s"])
    cases = [
        ("Input_pH", ph[[0, 2, 4, 6, 8]], ph[[1, 3, 5, 7]]),
        ("Hfo_s", hfo[[0, 2, 4]], hfo[[1, 3]]),
    ]

    rows, details = [], []
    for axis, keep, drop in cases:
        summary, detail = run_case(df, axis, keep, drop)
        rows.append(summary)
        details.append(detail)
    pd.concat(details).to_csv(OUT, index=False)

    # The two thresholds are calibrated inside this script, so anything that reports the
    # results downstream should read them from here rather than restate them.
    import json
    json.dump({r["axis"]: {"conformal_half_width": r["conformal_half_width"],
                           "ad_threshold": r["ad_threshold"]} for r in rows},
              open(CONSTANTS, "w"), indent=2)

    for r in rows:
        print("=" * 74)
        print(f"Coarsened axis: {r['axis']}   ({r['n']:,} held-out points)")
        print("=" * 74)
        print(f"  MAE on the removed levels        {r['MAE']:8.2f} adsorption points")
        print(f"  conformal 90% half-width         {r['conformal_half_width']:8.2f}")
        print("\n  rank correlation with actual error")
        print(f"    distance-based AD              {r['ad_rho']:+8.3f}")
        print(f"    resolution-aware score         {r['res_rho']:+8.3f}")
        print("\n  fraction of under-resolved points flagged")
        print(f"    distance-based AD              {r['ad_flag_rate']:8.1%}")
        print(f"    resolution-aware score         {r['res_flag_rate']:8.1%}")
        print("\n  MAE split by the resolution flag")
        print(f"    flagged                        {r['mae_res_flagged']:8.2f}")
        print(f"    not flagged                    {r['mae_res_clear']:8.2f}")
        print(f"  most frequently blamed axis      {r['top_axis']}\n")

    print(f"Per-point output written to {OUT}")


if __name__ == "__main__":
    main()
