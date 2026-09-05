"""Tests for the released U(VI)-ferrihydrite surrogate interface.

Runs under pytest, and also as a plain script if pytest is not installed:

    pytest test_surrogate.py -v
    python test_surrogate.py

The tests are deliberately about the contract of the released artefact rather than about
model accuracy, which is evaluated separately. What is checked here is that the
interface refuses bad input, that the physical invariants hold on every returned row, that
the reliability flags behave as documented, and that two identical calls
return identical numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uvi_surrogate import SurrogatePredictor  # noqa: E402

SPECIES = ["Hfo_sOUO2+", "(Hfo_sO)2UO2", "(Hfo_sO)2UO2CO3-2",
           "Hfo_wOUO2+", "(Hfo_wO)2UO2", "(Hfo_wO)2UO2CO3-2"]

# a condition sitting exactly on sampled levels of every axis
ON_GRID = {
    "Input_pH": 7.0,
    "U_initial": 1.389495494373136e-06,
    "Carbonate": 1.930697728883e-04,
    "NaCl": 1.58489319246111e-02,
    "Ca": 1e-3,
    "Hfo_s": 3.1622776601683795e-05,
    "Hfo_w": 1.2649110640673e-03,
}

_predictor = None


def predictor() -> SurrogatePredictor:
    global _predictor
    if _predictor is None:
        _predictor = SurrogatePredictor()
    return _predictor


def frame(**overrides) -> pd.DataFrame:
    row = dict(ON_GRID)
    row.update(overrides)
    return pd.DataFrame([row])


def mixed_batch(n: int = 200) -> pd.DataFrame:
    """A batch spanning on-grid, between-grid and out-of-range conditions."""
    rng = np.random.RandomState(0)
    rows = []
    for _ in range(n):
        rows.append({
            "Input_pH": rng.uniform(3.0, 11.0),
            "U_initial": 10 ** rng.uniform(-8, -3),
            "Carbonate": 10 ** rng.uniform(-5, -2),
            "NaCl": 10 ** rng.uniform(-3, 0),
            "Ca": rng.choice([0.0, 10 ** rng.uniform(-5, -2)]),
            "Hfo_s": 10 ** rng.uniform(-6, -3),
        })
        rows[-1]["Hfo_w"] = rows[-1]["Hfo_s"] * 40.0
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- manifest


def test_manifest_version_is_current():
    assert predictor().manifest["version"] == "0.2.0"


def test_manifest_carries_design_levels():
    """Without these the resolution criterion is silently disabled, not raised."""
    manifest = predictor().manifest
    levels = manifest.get("design_levels")
    assert levels, "design_levels missing; build_artifacts.py may have written a 0.1.0 manifest"
    assert set(levels) == {"Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s"}
    assert [len(v) for v in (levels[k] for k in
            ["Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s"])] == [9, 8, 8, 6, 5, 5]
    assert manifest["coupled_features"]["Hfo_w"][0] == "Hfo_s"


def test_resolution_scorer_is_active():
    assert predictor().resolution is not None


def test_required_inputs_match_manifest():
    assert list(predictor().required_inputs) == predictor().manifest["base_features"]


# ---------------------------------------------------------------------- input validation


def test_missing_column_raises():
    bad = frame().drop(columns=["Carbonate"])
    try:
        predictor().predict(bad)
    except ValueError as exc:
        assert "Carbonate" in str(exc)
    else:
        raise AssertionError("a missing input column should raise ValueError")


def test_negative_concentration_raises():
    try:
        predictor().predict(frame(Carbonate=-1e-4))
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("a negative concentration should raise ValueError")


def test_zero_calcium_is_accepted():
    """Ca = 0 is a sampled level and must not be rejected by the log transform."""
    out = predictor().predict(frame(Ca=0.0))
    assert np.isfinite(out["Ads_%"]).all()


# -------------------------------------------------------------------- physical invariants


def test_adsorption_within_bounds():
    out = predictor().predict(mixed_batch())
    assert out["Ads_%"].between(0.0, 100.0).all()
    assert out["Ads_%_lower90"].between(0.0, 100.0).all()
    assert out["Ads_%_upper90"].between(0.0, 100.0).all()


def test_interval_ordering():
    out = predictor().predict(mixed_batch())
    for target in ("Ads_%", "logKd"):
        lo, mid, hi = out[f"{target}_lower90"], out[target], out[f"{target}_upper90"]
        # clipping the adsorption interval at 0 and 100 can pull a bound onto the estimate,
        # so the requirement is ordering, not strict inequality
        assert (lo <= mid + 1e-9).all(), f"{target}: lower bound above the point estimate"
        assert (mid <= hi + 1e-9).all(), f"{target}: point estimate above the upper bound"


def test_surface_species_non_negative():
    out = predictor().predict(mixed_batch())
    assert (out[SPECIES] >= 0).all().all()


def test_surface_species_close_to_predicted_u_ads():
    out = predictor().predict(mixed_batch())
    residual = (out[SPECIES].sum(axis=1) - out["U_ads_pred"]).abs()
    tolerance = np.maximum(1e-18, out["U_ads_pred"].abs() * 1e-10)
    assert (residual <= tolerance).all()
    assert (out["consistency_status"] == "pass").all()


# ------------------------------------------------------------------- reliability flags


def test_on_grid_query_is_fully_resolved():
    out = predictor().predict(frame())
    assert out["resolution_score"].iloc[0] == 0.0
    assert out["resolution_axis"].iloc[0] == "on-grid"
    assert out["resolution_status"].iloc[0] == "resolved"


def test_midway_ph_on_the_adsorption_edge_is_flagged():
    """pH 4.5 at low carbonate sits on the steep part of the edge between sampled levels."""
    out = predictor().predict(frame(Input_pH=4.5, Carbonate=1e-5))
    assert out["resolution_status"].iloc[0] == "under-resolved"
    assert out["resolution_axis"].iloc[0] == "Input_pH"
    assert out["resolution_score"].iloc[0] > 10.0


def test_resolution_score_never_negative():
    out = predictor().predict(mixed_batch())
    assert (out["resolution_score"] >= 0).all()


def test_reliability_status_agrees_with_its_components():
    out = predictor().predict(mixed_batch())
    ad_ok = out["AD_status"] == "in-domain"
    res_ok = out["resolution_status"] == "resolved"
    expected = np.where(ad_ok & res_ok, "ok",
               np.where(~ad_ok & ~res_ok, "lower-support+under-resolved",
               np.where(ad_ok, "under-resolved", "lower-support")))
    assert (out["reliability_status"].to_numpy() == expected).all()


def test_training_conditions_are_not_falsely_flagged():
    """Sampled design points must not raise resolution alarms."""
    data = (Path(__file__).resolve().parents[1] / "data"
            / "U_HFO_ML_Dataset_Final.csv")
    if not data.exists():
        return  # dataset not present; skip rather than fail
    sample = pd.read_csv(data).sample(500, random_state=42)[list(ON_GRID)]
    out = predictor().predict(sample)
    assert (out["resolution_status"] == "resolved").all()


# ------------------------------------------------------------------------ determinism


def test_repeated_calls_are_identical():
    batch = mixed_batch(50)
    first, second = predictor().predict(batch), predictor().predict(batch)
    numeric = first.select_dtypes(include=[np.number]).columns
    assert np.array_equal(first[numeric].to_numpy(), second[numeric].to_numpy())
    for column in ("AD_status", "resolution_status", "reliability_status"):
        assert (first[column].to_numpy() == second[column].to_numpy()).all()


def test_row_order_does_not_change_results():
    batch = mixed_batch(50)
    straight = predictor().predict(batch).reset_index(drop=True)
    reversed_ = predictor().predict(batch.iloc[::-1]).iloc[::-1].reset_index(drop=True)
    numeric = straight.select_dtypes(include=[np.number]).columns
    assert np.allclose(straight[numeric].to_numpy(), reversed_[numeric].to_numpy())


def test_index_is_preserved():
    batch = mixed_batch(10).set_index(pd.Index(range(100, 110), name="sample"))
    assert list(predictor().predict(batch).index) == list(batch.index)


# ------------------------------------------------------------------- output contract


def test_expected_columns_present():
    expected = {
        "Ads_%", "Ads_%_lower90", "Ads_%_upper90",
        "logKd", "logKd_lower90", "logKd_upper90",
        "AD_distance", "AD_status",
        "resolution_score", "resolution_axis", "resolution_status",
        "reliability_status", "consistency_status",
        "U_ads_pred", "surface_closure_error", *SPECIES,
    }
    assert expected <= set(predictor().predict(frame()).columns)


def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
