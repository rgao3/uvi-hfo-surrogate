# User guide — `SurrogatePredictor`

`SurrogatePredictor` returns, for each query condition, point estimates for U(VI) adsorption
together with calibrated reliability diagnostics. This guide documents the inputs,
outputs, options, and expected behaviour.

## Loading

```python
from uvi_surrogate import SurrogatePredictor
predictor = SurrogatePredictor()                 # loads the fitted artifacts in uvi_surrogate/artifacts/
predictor = SurrogatePredictor(artifact_dir=...) # optional: load artifacts from another directory
```

## Inputs

`predict(inputs)` accepts a `pandas.DataFrame` (or a list of dicts) with the following
seven columns. All concentrations are in mol per kg of water (≈ mol/L for dilute
solutions). Predictions are intended for interpolation within the sampled design ranges
below; queries outside these ranges are accepted but should be treated with caution and
will typically be flagged by the applicability-domain status.

| Column | Meaning | Sampled range |
|---|---|---|
| `Input_pH` | pH | 3 – 11 |
| `U_initial` | total uranium(VI) | 1×10⁻⁸ – 1×10⁻³ |
| `Carbonate` | total inorganic carbon | 1×10⁻⁵ – 1×10⁻² |
| `NaCl` | nominal NaCl electrolyte | 1×10⁻³ – 1 |
| `Ca` | total calcium (an explicit `0` level is allowed) | 0, then 1×10⁻⁵ – 1×10⁻² |
| `Hfo_s` | strong-site HFO density | 1×10⁻⁶ – 1×10⁻³ |
| `Hfo_w` | weak-site HFO density (coupled: fixed at 40 × `Hfo_s`) | 40 × `Hfo_s` |

## Outputs

`predict()` returns a `pandas.DataFrame` with one row per query and the following columns.

**Primary targets and 90% conformal intervals**

| Column | Meaning |
|---|---|
| `Ads_%` | predicted adsorption percentage (0–100) |
| `Ads_%_lower90`, `Ads_%_upper90` | 90% pH-conditional conformal prediction interval for `Ads_%` |
| `logKd` | predicted log₁₀ of the distribution coefficient |
| `logKd_lower90`, `logKd_upper90` | 90% conformal interval for `logKd` |

**Reliability diagnostics**

| Column | Meaning |
|---|---|
| `AD_distance` | distance-based applicability-domain score (mean distance to nearest training points in standardised feature space) |
| `AD_status` | `in-domain` if `AD_distance` ≤ the 95th-percentile training threshold, else `lower-support` |
| `resolution_score` | resolution-aware score: the largest surrogate response change (in adsorption points) across the design levels bracketing the query. Larger = the local simulation design may be too coarse to resolve the response |
| `resolution_axis` | the design axis contributing the largest bracketing change |
| `resolution_status` | `resolved` if `resolution_score` ≤ the pH-conditional conformal half-width, else `under-resolved` |
| `reliability_status` | combined flag: `ok`, `lower-support`, `under-resolved`, or `lower-support+under-resolved` |
| `consistency_status` | `pass` if the reconstructed surface species close to the predicted adsorbed U within tolerance, else `review` |

**Surface-species reconstruction and mass balance**

| Column | Meaning |
|---|---|
| `U_ads_pred` | predicted adsorbed U (mol) = `Ads_%`/100 × `U_initial` |
| `surface_closure_error` | \|Σ species − `U_ads_pred`\| (mass-closure residual) |
| six species columns (`Hfo_sOUO2+`, `(Hfo_sO)2UO2`, `(Hfo_sO)2UO2CO3-2`, `Hfo_wOUO2+`, `(Hfo_wO)2UO2`, `(Hfo_wO)2UO2CO3-2`) | predicted adsorbed U (mol) assigned to each surface complex |

## How to read the reliability flags

- `reliability_status == "ok"` — the query is close to training data (in-domain) **and** the
  local design is fine enough to resolve the response. Use the prediction with its interval.
- `lower-support` — few training simulations are nearby; the estimate is an extrapolation.
- `under-resolved` — the simulation design is too coarse locally; the response can change
  more between adjacent design levels than the calibrated interval implies. This is the
  situation the resolution criterion is designed to catch and the distance-based
  applicability domain typically misses.

## Options and expected behaviour

- `SurrogatePredictor(artifact_dir=path)` loads a different set of fitted artifacts.
- Input validation raises `ValueError` if a required column is missing, or if any
  concentration or HFO density is negative.
- Predictions are deterministic: two identical calls return identical numbers, and the
  result does not depend on input row order.
