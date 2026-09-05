# U(VI)–HFO surrogate

This package implements the reliability workflow for the U(VI)–HFO surrogate. One call returns point estimates for `Ads_%` and `logKd`, pH-conditional 90% conformal intervals, a distance-based applicability-domain status, six predicted surface-species concentrations, and a mass-closure status.

```python
import pandas as pd
from uvi_surrogate import SurrogatePredictor

predictor = SurrogatePredictor()
query = pd.DataFrame([{
    "Input_pH": 7.0,
    "U_initial": 1.389495494373136e-6,
    "Carbonate": 1.930697728883e-4,
    "NaCl": 1.58489319246111e-2,
    "Ca": 1e-3,
    "Hfo_s": 3.1622776601683795e-5,
    "Hfo_w": 1.2649110640673e-3,
}])
result = predictor.predict(query)
```

The fitted artifacts are under `uvi_surrogate/artifacts/`. The interface is intended for interpolation within the sampled PHREEQC grid. Queries with `AD_status == "lower-support"` should be reviewed or prioritized for additional PHREEQC simulation.

Rebuild the artifacts from the fixed training/test splits with:

```powershell
python consistency_and_benchmark/build_artifacts.py
```
