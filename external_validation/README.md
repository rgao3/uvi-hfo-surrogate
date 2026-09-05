# External comparison with published measurements

Comparison of the surrogate against published U(VI)–HFO sorption measurements, used for
the external-consistency assessment in the accompanying paper.

| script | purpose | output |
|---|---|---|
| `validate_resolution_criterion.py` | coarsened-design validation of the resolution criterion (internal data only) | `resolution_criterion_validation.csv`, `resolution_criterion_constants.json` |
| `grid_spacing_experiment.py` | quantifies the response variation hidden between adjacent design levels | `grid_spacing_experiment.csv` |
| `ad_screen_published_experiments.py` | screens published experimental conditions against the applicability domain | `ad_screen_results.csv` |
| `parse_payne_appendix.py` | extracts the measured points from the Payne (1999) appendix tables | `payne_1999_ferrihydrite_measured.csv` |
| `compare_surrogate_vs_payne.py` | compares surrogate predictions with the measured Payne (1999) data | `surrogate_vs_payne_results.csv` |

Experimental conditions are taken from Table 1 of Mahoney, Cadle & Jakubowski (2009),
*Environ. Sci. Technol.* **43**, 9260–9266. The Payne (1999) measurements are from
Payne, T.E. (1999), *Uranium (VI) interactions with mineral surfaces*, PhD thesis, UNSW
(open access; Appendix 1). The thesis PDF is not redistributed here; `parse_payne_appendix.py`
documents the extraction.
