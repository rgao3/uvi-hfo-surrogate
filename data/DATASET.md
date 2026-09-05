# Dataset

This folder contains the PHREEQC simulation data used throughout the repository.

| File | Description |
|---|---|
| `U_HFO_ML_Dataset_Final.csv` | Full simulation database: 86,375 converged PHREEQC simulations of U(VI)–HFO adsorption. This is the dataset all analysis scripts use. |
| `U_HFO_ML_Dataset_Final_missing25_failed_cases.csv` | The 25 non-converged design points, retained as a challenge set. |
| `U_HFO_ML_Dataset_sample.csv` | A 1,000-row random sample of the full database, for the tutorial and quick tests. |
| `phreeqc_generation/` | Scripts that generate (`HFO_U_test4.py`) and post-process (`process_phreeqc_outputs.py`) the PHREEQC database. |

Each row is one equilibrium condition. Inputs are the seven design variables
(`Input_pH`, `U_initial`, `Carbonate`, `NaCl`, `Ca`, `Hfo_s`, `Hfo_w`); the remaining
columns are the PHREEQC outputs and derived quantities (`Ads_%`, `logKd`, surface-species
concentrations, dominant species, etc.).

The JAEA thermodynamic database (201203c0.tdb) needed to *regenerate* the data is obtained
separately from JAEA and is not redistributed here.
