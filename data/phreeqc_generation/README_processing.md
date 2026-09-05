# PHREEQC Output Processing

This folder contains a small, dependency-free Python script for converting
PHREEQC U(VI)-HFO adsorption output files into a modeling-ready Excel workbook.

## Script

`process_phreeqc_outputs.py`

The script reads all `.txt` files in the `outputs` folder and writes one row per
PHREEQC simulation block. A single PHREEQC output file may contain multiple
simulation blocks, so the number of rows in the final table is usually larger
than the number of text files.

## Run

From this folder:

```powershell
python process_phreeqc_outputs.py
```

Default output:

```text
phreeqc_processed_results.xlsx
```

Optional arguments:

```powershell
python process_phreeqc_outputs.py --outputs-dir outputs --output phreeqc_processed_results.xlsx --site-density 0.005
```

## Workbook Sheets

- `results`: modeling-ready data table.
- `file_summary`: number of parsed simulations and key variable levels per source file.
- `method`: formulas and assumptions used during processing.

## Calculations

The current project convention is:

```text
pe = 20.8 - pH
U_ads = sum(U-bearing HFO surface complexes)
Ads_% = U_ads / U_total * 100
HFO_g_L = (Hfo_sOH + Hfo_wOH) / 0.005
Kd = (U_ads / HFO_g_L) / U_diss
logKd = log10(Kd)
```

The U-bearing surface complexes are:

```text
Hfo_sOUO2+
(Hfo_sO)2UO2
Hfo_wOUO2+
(Hfo_wO)2UO2
(Hfo_sO)2UO2CO3-2
(Hfo_wO)2UO2CO3-2
```

`U_diss` is extracted from the final `Solution composition` section in each
simulation block. The dominant aqueous U species is selected from the final
`U(6)` species table, and the dominant surface species is selected from the
largest U-bearing HFO surface complex.
