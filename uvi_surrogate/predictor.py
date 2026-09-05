"""Prediction interface for the U(VI)-ferrihydrite surrogate."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .resolution import ResolutionScorer


class SurrogatePredictor:
    """Return point predictions, calibrated intervals, AD status, and surface species.

    Reliability is reported through two complementary criteria. The distance-based
    applicability domain asks whether training data exist near the query. The
    resolution score asks whether the design is fine enough to resolve the response
    around it; see `resolution.py` for why the second question is not answered by the
    first on a coarse factorial design.
    """

    def __init__(self, artifact_dir=None):
        base = Path(artifact_dir) if artifact_dir else Path(__file__).with_name("artifacts")
        with open(base / "manifest.json", encoding="utf-8") as f:
            self.manifest = json.load(f)
        with open(base / "xgb_Ads_pct_conformal.pkl", "rb") as f:
            self.ads_model = pickle.load(f)
        with open(base / "xgb_logKd_conformal.pkl", "rb") as f:
            self.kd_model = pickle.load(f)
        with open(base / "xgb_surface_fractions.pkl", "rb") as f:
            self.surface_model = pickle.load(f)
        ad = np.load(base / "ad_reference.npz")
        self.ad_mu, self.ad_sd = ad["mu"], ad["sd"]
        self.ad_nn = NearestNeighbors(n_neighbors=5).fit(ad["reference"])
        levels = self.manifest.get("design_levels")
        if levels:
            coupled = {k: (v[0], float(v[1]))
                       for k, v in self.manifest.get("coupled_features", {}).items()}
            self.resolution = ResolutionScorer(levels, coupled)
        else:
            self.resolution = None

    @property
    def required_inputs(self):
        return tuple(self.manifest["base_features"])

    def _features(self, frame):
        missing = [c for c in self.required_inputs if c not in frame]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")
        d = frame.loc[:, self.required_inputs].astype(float).copy()
        positive = self.manifest["log_columns"]
        if (d[positive] < 0).any().any():
            raise ValueError("Concentrations and HFO site densities must be non-negative.")
        for c in positive:
            d["log10_" + c] = np.log10(d[c].clip(lower=1e-12))
        return d.loc[:, self.manifest["features"]].to_numpy(dtype=np.float32)

    def predict(self, inputs):
        frame = pd.DataFrame(inputs).copy()
        X = self._features(frame)
        ads = np.clip(self.ads_model.predict(X), 0.0, 100.0)
        logkd = self.kd_model.predict(X)

        pH = frame["Input_pH"].to_numpy(dtype=float)
        bands = np.where(pH <= 5, "<=5", np.where(pH <= 7, "6-7", ">=8"))
        q_ads = np.array([self.manifest["conformal_90"]["Ads_%"][b] for b in bands])
        q_kd = np.array([self.manifest["conformal_90"]["logKd"][b] for b in bands])

        Xs = (X.astype(float) - self.ad_mu) / self.ad_sd
        ad_distance = self.ad_nn.kneighbors(Xs, return_distance=True)[0].mean(axis=1)
        ad_ok = ad_distance <= self.manifest["ad_threshold_p95"]

        raw_fraction = np.clip(self.surface_model.predict(X).astype(float), 0.0, None)
        row_sum = raw_fraction.sum(axis=1, keepdims=True)
        fractions = np.divide(raw_fraction, row_sum, out=np.full_like(raw_fraction, 1 / 6), where=row_sum > 0)
        u_ads = ads / 100.0 * frame["U_initial"].to_numpy(dtype=float)
        species = fractions * u_ads[:, None]
        closure = np.abs(species.sum(axis=1) - u_ads)
        closure_tol = np.maximum(1e-18, np.abs(u_ads) * 1e-10)

        if self.resolution is not None:
            def _point(f):
                return np.clip(self.ads_model.predict(self._features(f)), 0.0, 100.0)
            res_score, res_axis, _ = self.resolution.score(frame, _point)
            res_ok = res_score <= q_ads
        else:
            res_score = np.zeros(len(frame))
            res_axis = np.full(len(frame), "not-available", dtype=object)
            res_ok = np.ones(len(frame), dtype=bool)

        out = pd.DataFrame({
            "Ads_%": ads,
            "Ads_%_lower90": np.clip(ads - q_ads, 0.0, 100.0),
            "Ads_%_upper90": np.clip(ads + q_ads, 0.0, 100.0),
            "logKd": logkd,
            "logKd_lower90": logkd - q_kd,
            "logKd_upper90": logkd + q_kd,
            "AD_distance": ad_distance,
            "AD_status": np.where(ad_ok, "in-domain", "lower-support"),
            "resolution_score": res_score,
            "resolution_axis": res_axis,
            "resolution_status": np.where(res_ok, "resolved", "under-resolved"),
            "reliability_status": np.where(ad_ok & res_ok, "ok",
                                  np.where(~ad_ok & ~res_ok, "lower-support+under-resolved",
                                  np.where(ad_ok, "under-resolved", "lower-support"))),
            "consistency_status": np.where(closure <= closure_tol, "pass", "review"),
            "U_ads_pred": u_ads,
            "surface_closure_error": closure,
        }, index=frame.index)
        for j, name in enumerate(self.manifest["surface_species"]):
            out[name] = species[:, j]
        return out
