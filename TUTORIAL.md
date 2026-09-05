# Tutorial

A short, self-contained walk-through of the typical use cases. It assumes Python 3.10+.

## 1. Install

From the repository root:

```
pip install -r requirements.txt
```

This installs the pinned dependencies (numpy, pandas, scipy, scikit-learn, xgboost).

## 2. Check the installation

Run the interface tests. All 20 should pass:

```
python uvi_surrogate/test_surrogate.py
```

## 3. One prediction with reliability diagnostics

```python
import pandas as pd
from uvi_surrogate import SurrogatePredictor

predictor = SurrogatePredictor()

query = pd.DataFrame([{
    "Input_pH": 7.0,
    "U_initial": 1.389e-6,
    "Carbonate": 1.931e-4,
    "NaCl": 1.585e-2,
    "Ca": 1.0e-3,
    "Hfo_s": 3.162e-5,
    "Hfo_w": 3.162e-5 * 40,   # weak sites are coupled at 40x strong sites
}])

result = predictor.predict(query)
print(result[["Ads_%", "Ads_%_lower90", "Ads_%_upper90",
              "logKd", "AD_status", "resolution_status", "reliability_status"]].T)
```

The returned row gives the adsorption estimate, its 90% conformal interval, and the
reliability flags. See `USER_GUIDE.md` for every output column.

## 4. Batch prediction from the sample dataset

A 1,000-row random sample of the simulation database is included at
`data/U_HFO_ML_Dataset_sample.csv` so the example runs without downloading the full data.

```python
import pandas as pd
from uvi_surrogate import SurrogatePredictor

df = pd.read_csv("data/U_HFO_ML_Dataset_sample.csv")
inputs = df[["Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s", "Hfo_w"]]

pred = SurrogatePredictor().predict(inputs)

# compare against the PHREEQC reference values in the sample
import numpy as np
mae = np.mean(np.abs(pred["Ads_%"].values - df["Ads_%"].values))
print(f"Ads% MAE on the sample: {mae:.3f} percentage points")
print(pred["reliability_status"].value_counts())
```

## 5. Interpreting the reliability flags

- `reliability_status == "ok"`: in-domain and locally well-resolved — use the prediction
  with its interval.
- `lower-support`: an extrapolation away from the training simulations.
- `under-resolved`: the simulation design is locally too coarse; the response may vary more
  between adjacent design levels than the interval implies. This is the case the resolution
  criterion catches and the distance-based applicability domain usually misses.

## 6. (Optional) reproduce the analysis and rebuild the artifacts

The released artifacts under `uvi_surrogate/artifacts/` reproduce all predictions directly.
To regenerate intermediate results and the fitted artifacts from the full database
(`data/U_HFO_ML_Dataset_Final.csv`, included), see the commands in `README.md`, for example:

```
python consistency_and_benchmark/build_artifacts.py     # rebuilds uvi_surrogate/artifacts/
python resolution_criterion/model_class_experiment.py   # numpy/pandas only, ~1 min
```

Random seeds are fixed (42); model fitting is cached if the `.pkl` files are present.
