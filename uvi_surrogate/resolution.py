"""Resolution-aware reliability criterion for surrogates trained on a factorial design.

Motivation
----------
A distance-based applicability domain answers "have I seen training data near this
query?". On a dense factorial design the answer is almost always yes, because every
point inside the design box is within half a grid step of a sampled node. What it does
not answer is "is the response smooth enough between the nodes I have seen?".

For a tree ensemble the prediction is piecewise constant between splits, so a query
lying between two sampled levels is answered with the value of the level below it. Where
the response changes steeply across that gap, the error is large while the distance-based
flag stays green. The two questions are different and only the first one is usually asked.

Criterion
---------
For a query x and each design axis a, let lo_a and hi_a be the sampled levels bracketing
x[a]. Evaluate the model at x with x[a] replaced by lo_a and by hi_a, holding every other
input at its queried value, and take the absolute difference. That difference is the
amount of response the design cannot resolve along axis a at this location. The score is
the maximum over axes, and the argmax names the axis responsible.

Cost is 2 predictions per axis per query, which is 12 extra model evaluations here. It
requires no additional simulation.

Decision rule
-------------
The score is expressed in the units of the target, so it can be compared directly with
the calibrated conformal half-width. A prediction is flagged when grid resolution
contributes more uncertainty than the statistical interval already reports, i.e. when

    resolution_score > conformal_half_width

This avoids introducing an arbitrary constant: the threshold is whatever the conformal
calibration already decided is a meaningful amount of uncertainty for that query.
"""

from __future__ import annotations

import numpy as np


class ResolutionScorer:
    """Score how much response a factorial design fails to resolve at a query point.

    Parameters
    ----------
    levels : dict
        Mapping from base feature name to the sorted array of sampled levels.
    coupled : dict, optional
        Features that are locked to another feature by a fixed multiplier, e.g.
        ``{"Hfo_w": ("Hfo_s", 40.0)}``. These are moved together with their driver so
        the perturbation stays on the design manifold.
    """

    def __init__(self, levels: dict[str, np.ndarray], coupled: dict | None = None):
        self.levels = {k: np.sort(np.asarray(v, dtype=float)) for k, v in levels.items()}
        self.coupled = coupled or {}

    @property
    def axes(self) -> list[str]:
        return list(self.levels)

    def _bracket(self, name: str, values: np.ndarray):
        """Sampled levels immediately below and above each value."""
        lv = self.levels[name]
        idx = np.clip(np.searchsorted(lv, values, side="right") - 1, 0, len(lv) - 2)
        lo, hi = lv[idx], lv[idx + 1]
        # a query sitting exactly on a level resolves to zero spread
        on_node = np.isclose(values[:, None], lv[None, :], rtol=1e-9, atol=1e-15).any(axis=1)
        return lo, hi, on_node

    def score(self, frame, predict_fn):
        """Return (score, responsible_axis, per_axis_spread).

        ``frame`` is a DataFrame of base features; ``predict_fn`` maps such a frame to a
        1-D array of predictions.
        """
        n = len(frame)
        spreads = np.zeros((n, len(self.axes)), dtype=float)

        for j, name in enumerate(self.axes):
            values = frame[name].to_numpy(dtype=float)
            lo, hi, on_node = self._bracket(name, values)
            if on_node.all():
                continue
            low_frame, high_frame = frame.copy(), frame.copy()
            low_frame[name], high_frame[name] = lo, hi
            for dependent, (driver, factor) in self.coupled.items():
                if driver == name:
                    low_frame[dependent] = lo * factor
                    high_frame[dependent] = hi * factor
            spread = np.abs(predict_fn(high_frame) - predict_fn(low_frame))
            spread[on_node] = 0.0
            spreads[:, j] = spread

        best = spreads.argmax(axis=1)
        score = spreads[np.arange(n), best]
        axis = np.array(self.axes, dtype=object)[best]
        axis = np.where(score > 0, axis, "on-grid")
        return score, axis, spreads
