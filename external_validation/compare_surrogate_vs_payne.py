"""Compare surrogate predictions with the measured Payne (1999) ferrihydrite data.

External-consistency test: the surrogate is asked to
predict conditions from published laboratory experiments that had no influence on the
design of the simulation grid, and its own reliability flags are checked against the
outcome.

Three things are deliberately kept separate in the output:

1. Whether the experimental condition can be represented in the model input space at all.
2. What the applicability-domain score says about it.
3. How far the prediction is from the measurement.

Reading (3) without (1) and (2) would be misleading, because a large error at a condition
the surrogate already flagged as unsupported is the tool working, not failing.

Interpretation caveats
----------------------
* The comparison is surrogate vs experiment, so it inherits every error in the underlying
  PHREEQC configuration. A disagreement here is evidence about the whole chain, not about
  the machine-learning step alone. Running the same conditions through PHREEQC directly
  would separate the two.
* Payne used 0.1 M NaNO3; the simulation design uses NaCl. Nitrate is essentially
  non-complexing towards uranyl whereas chloride forms UO2Cl+, so the two are matched on
  ionic strength but not on speciation.
* The experiments are open to a fixed CO2 partial pressure. The design variable is total
  dissolved carbon, so total C is computed here from Henry's law and carbonic-acid
  speciation at each pH rather than taken from a reported value.
* The surrogate takes pH as an input and the experiments report measured pH, so the pH
  axis needs no conversion.

Output: surrogate_vs_payne_results.csv and a printed summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELING = HERE.parent
sys.path.insert(0, str(MODELING))
from uvi_surrogate import SurrogatePredictor  # noqa: E402

MEASURED = HERE / "payne_1999_ferrihydrite_measured.csv"
OUT = HERE / "surrogate_vs_payne_results.csv"

# Dzombak & Morel (1990) HFO conventions, as restated by Mahoney et al. (2009)
STRONG_PER_MOL_FE = 0.005
WEAK_PER_MOL_FE = 0.200

# carbonic acid system at 25 C, infinite dilution
K_HENRY = 10 ** -1.47
K_A1 = 10 ** -6.35
K_A2 = 10 ** -10.33

# sampled ranges of the simulation design
GRID = {
    "Input_pH": (3.0, 11.0),
    "U_initial": (1e-8, 1e-3),
    "Carbonate": (1e-5, 1e-2),
    "NaCl": (1e-3, 1.0),
    "Hfo_s": (1e-6, 1e-3),
}


def total_carbon(log_pco2: float, pH: float) -> float:
    h = 10 ** -pH
    return K_HENRY * 10 ** log_pco2 * (1 + K_A1 / h + K_A1 * K_A2 / h ** 2)


def main():
    meas = pd.read_csv(MEASURED)
    meas["Input_pH"] = meas["pH"]
    meas["U_initial"] = meas["U_total_M"]
    meas["Carbonate"] = [total_carbon(p, ph)
                         for p, ph in zip(meas["log10_pCO2_atm"], meas["pH"])]
    meas["NaCl"] = meas["ionic_strength_M"]
    meas["Ca"] = 0.0
    meas["Hfo_s"] = STRONG_PER_MOL_FE * meas["Fe_total_M"]
    meas["Hfo_w"] = WEAK_PER_MOL_FE * meas["Fe_total_M"]

    outside = []
    for _, r in meas.iterrows():
        bad = [k for k, (lo, hi) in GRID.items() if not lo <= r[k] <= hi]
        outside.append(";".join(bad))
    meas["axes_outside_sampled_range"] = outside
    meas["inside_sampled_box"] = meas["axes_outside_sampled_range"] == ""

    inputs = meas[["Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s", "Hfo_w"]]
    pred = SurrogatePredictor().predict(inputs)

    res = meas.copy()
    res["pred_Ads_pct"] = pred["Ads_%"].to_numpy()
    res["pred_lower90"] = pred["Ads_%_lower90"].to_numpy()
    res["pred_upper90"] = pred["Ads_%_upper90"].to_numpy()
    res["AD_distance"] = pred["AD_distance"].to_numpy()
    res["AD_status"] = pred["AD_status"].to_numpy()
    res["residual_pct"] = res["pred_Ads_pct"] - res["U_sorbed_pct"]
    res["abs_residual_pct"] = res["residual_pct"].abs()
    res["measured_in_interval"] = ((res["U_sorbed_pct"] >= res["pred_lower90"]) &
                                   (res["U_sorbed_pct"] <= res["pred_upper90"]))
    res.to_csv(OUT, index=False)

    def block(title):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    print(f"Measured points from Payne (1999) Appendix 1: {len(res)}")

    block("STEP 1 - representable in the model input space?")
    print(f"inside sampled ranges : {res.inside_sampled_box.sum()}")
    print(f"outside               : {(~res.inside_sampled_box).sum()}")
    gaps = (res.loc[~res.inside_sampled_box, "axes_outside_sampled_range"]
               .value_counts().rename("n"))
    if len(gaps):
        print("\naxis responsible:")
        print(gaps.to_string())

    usable = res[res.inside_sampled_box]

    block("STEP 2 - applicability domain, for representable points only")
    print(usable.AD_status.value_counts().to_string())

    block("STEP 3 - agreement with measurement")
    for label, sub in [("all representable points", usable),
                       ("AD in-domain only", usable[usable.AD_status == "in-domain"]),
                       ("AD lower-support only", usable[usable.AD_status == "lower-support"])]:
        if not len(sub):
            continue
        print(f"\n{label}  (n = {len(sub)})")
        print(f"  MAE                       {sub.abs_residual_pct.mean():7.2f} points")
        print(f"  RMSE                      {np.sqrt((sub.residual_pct**2).mean()):7.2f} points")
        print(f"  median absolute residual  {sub.abs_residual_pct.median():7.2f} points")
        print(f"  mean signed residual      {sub.residual_pct.mean():+7.2f} points")
        print(f"  measured inside 90% band  {sub.measured_in_interval.mean():7.1%}")

    block("Per data set (representable points only)")
    per = (usable.groupby("dataset")
                 .agg(n=("residual_pct", "size"),
                      pH_min=("pH", "min"), pH_max=("pH", "max"),
                      MAE=("abs_residual_pct", "mean"),
                      bias=("residual_pct", "mean"),
                      in_band=("measured_in_interval", "mean"),
                      frac_lower_support=("AD_status",
                                          lambda s: (s == "lower-support").mean()))
                 .round(2))
    print(per.to_string())

    block("Largest disagreements")
    worst = usable.nlargest(10, "abs_residual_pct")[
        ["dataset", "pH", "U_sorbed_pct", "pred_Ads_pct", "residual_pct",
         "AD_status"]].round(2)
    print(worst.to_string(index=False))

    block("STEP 4 - how much of the error is pH discretisation?")
    print("The design samples pH only at integer values, and a tree ensemble is")
    print("piecewise constant between splits, so the surrogate is a staircase in pH")
    print("with one-unit treads. On the adsorption edge the true response changes by")
    print("tens of points per pH unit, so a measurement at intermediate pH is compared")
    print("against the value for the integer pH below it.\n")
    usable = usable.copy()
    usable["pH_offset"] = (usable["pH"] - usable["pH"].round()).abs()
    bins = pd.cut(usable["pH_offset"], [-0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
                  labels=["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5"])
    by_offset = (usable.groupby(bins, observed=True)
                       .agg(n=("abs_residual_pct", "size"),
                            MAE=("abs_residual_pct", "mean"),
                            median=("abs_residual_pct", "median"),
                            in_band=("measured_in_interval", "mean"))
                       .round(2))
    print("distance of measured pH from the nearest sampled integer pH:")
    print(by_offset.to_string())

    on_grid = usable[usable["pH_offset"] <= 0.1]
    off_grid = usable[usable["pH_offset"] > 0.3]
    print(f"\nnear a sampled pH (offset <= 0.1): n = {len(on_grid):3d}, "
          f"MAE = {on_grid.abs_residual_pct.mean():.2f} points")
    print(f"midway between sampled pH (> 0.3): n = {len(off_grid):3d}, "
          f"MAE = {off_grid.abs_residual_pct.mean():.2f} points")

    print("\nThe applicability-domain score does not detect this. Mean AD distance:")
    print(f"  near a sampled pH  {on_grid.AD_distance.mean():.3f}")
    print(f"  midway between     {off_grid.AD_distance.mean():.3f}")
    print(f"  threshold          {0.6194583248277084:.3f}")
    print("An intermediate pH is still close to training points in standardised")
    print("distance, so the AD flag stays green exactly where the staircase is worst.")

    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
