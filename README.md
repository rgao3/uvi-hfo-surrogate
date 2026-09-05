# A reliability-aware surrogate for PHREEQC U(VI)–HFO adsorption

Open-source software, data, and analysis code for a reliability-aware machine-learning
surrogate of a deterministic geochemical simulator (PHREEQC), demonstrated on uranium(VI)
(U(VI)) adsorption onto a hydrous ferric oxide (HFO) surface.

**What is new — computational:** a *resolution-aware reliability criterion* that diagnoses
whether the spacing of the simulation design is fine enough to resolve the response around
a query — a form of support that the conventional distance-based applicability domain does
not measure — delivered together with conformal prediction intervals and a mass-balance /
surface-species consistency audit through a single reusable prediction interface.

**What is new — geoscientific:** a fast, reproducible surrogate of a PHREEQC U(VI)–HFO
surface-complexation model that returns, with every prediction, calibrated uncertainty and
an explicit flag for where the geochemical simulation design is too coarse to be trusted.

## Documentation

- **`TUTORIAL.md`** — install, run a prediction, run a batch, interpret the reliability flags.
- **`USER_GUIDE.md`** — full specification of `SurrogatePredictor` inputs, outputs, options, behaviour.

## Repository layout

```
uvi-hfo-surrogate/
├── uvi_surrogate/              Released prediction package (SurrogatePredictor) + fitted artifacts + tests
├── data/                       Simulation database and PHREEQC generation scripts
│   ├── U_HFO_ML_Dataset_Final.csv                     full database (86,375 simulations)
│   ├── U_HFO_ML_Dataset_sample.csv                    1,000-row sample (tutorial/tests)
│   ├── U_HFO_ML_Dataset_Final_missing25_failed_cases.csv   25 non-converged cases
│   ├── DATASET.md              dataset description
│   └── phreeqc_generation/     scripts that generate and post-process the PHREEQC database
├── predictive_modeling/        Surrogate development: model comparison, nested CV, SHAP,
│                               conformal calibration, applicability domain, region holdouts
├── consistency_and_benchmark/  Geochemical-consistency tests, surface-species reconstruction,
│                               mass-balance audit, runtime benchmark; rebuilds the released artifacts
├── external_validation/        Resolution-criterion validation and comparison with published
│                               U(VI)–HFO sorption measurements
├── resolution_criterion/       Supporting experiments: model-class dependence and adaptive refinement
├── LICENSE / LICENSE-DATA      MIT (code) / CC BY 4.0 (data)
├── CITATION.cff                citation metadata
└── requirements.txt            pinned dependencies
```

## Installation

Python 3.10+ is recommended.

```
pip install -r requirements.txt
```

## Quick start

```python
import pandas as pd
from uvi_surrogate import SurrogatePredictor

predictor = SurrogatePredictor()
query = pd.DataFrame([{
    "Input_pH": 7.0, "U_initial": 1.389e-6, "Carbonate": 1.931e-4,
    "NaCl": 1.585e-2, "Ca": 1e-3, "Hfo_s": 3.162e-5, "Hfo_w": 3.162e-5*40,
}])
print(predictor.predict(query).T)
```

Each row returned carries `Ads_%` and `logKd` with 90% conformal intervals, the
applicability-domain status, the resolution-aware reliability flag, the six reconstructed
surface-species quantities, and a mass-closure status. See `USER_GUIDE.md`.

Run the interface tests (20 checks):

```
python uvi_surrogate/test_surrogate.py       # or: pytest uvi_surrogate/test_surrogate.py -v
```

## Reproducing the analysis

The released fitted models under `uvi_surrogate/artifacts/` reproduce all predictions
directly. To regenerate the intermediate results and artifacts from the full database:

```
# surrogate development
python predictive_modeling/run_nested_xgb2.py
python predictive_modeling/final_xgb.py
python predictive_modeling/baselines.py
python predictive_modeling/conformal.py
python predictive_modeling/applicability.py
python predictive_modeling/region_holdout.py
python predictive_modeling/shap_main.py

# consistency tests, surface-species reconstruction, mass balance, benchmark; rebuilds artifacts
python consistency_and_benchmark/build_artifacts.py

# resolution-criterion validation and design-spacing experiment
python external_validation/validate_resolution_criterion.py
python external_validation/grid_spacing_experiment.py

# supporting experiments
python resolution_criterion/model_class_experiment.py
python resolution_criterion/adaptive_refinement_experiment.py

# external comparison with published measurements
python external_validation/ad_screen_published_experiments.py
python external_validation/compare_surrogate_vs_payne.py
```

Random seeds are fixed (42). Model fitting is cached: if the fitted `.pkl` files are
present, scripts load rather than retrain. The scripts expect the full dataset at
`data/U_HFO_ML_Dataset_Final.csv` (see `data/DATASET.md`); the quick start and tests run on
the included sample.

## Data and code availability

This repository is self-contained: the code, the fitted model artifacts, and the full
simulation database (`data/U_HFO_ML_Dataset_Final.csv`, 86,375 simulations) are all
included, so the results can be reproduced directly from the repository. A versioned
snapshot of the whole repository can be archived on Zenodo for a citable DOI covering both
code and data (see `PLACEHOLDERS.md`).

The JAEA thermodynamic database (201203c0.tdb) used to generate the PHREEQC simulations is
obtained separately from JAEA and is not redistributed here.

## Citation

If you use this software or data, please cite the accompanying paper and this archive
(see `CITATION.cff`).

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`).
