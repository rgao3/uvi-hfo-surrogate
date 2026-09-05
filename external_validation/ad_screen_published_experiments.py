"""Screen published U(VI)-ferrihydrite experiments against the surrogate applicability domain.

This screens whether the conditions used in the published experiments fall inside the
region the surrogate was trained on. The comparison of predicted and measured adsorption
is performed separately in compare_surrogate_vs_payne.py.

Experimental conditions are taken from Table 1 of Mahoney, Cadle & Jakubowski (2009),
Environ. Sci. Technol. 43, 9260-9266, doi:10.1021/es901586w, which compiles 233 data
points from 14 data sets across five research groups.

Iron is converted to surface sites with the Dzombak & Morel (1990) convention as stated
in that paper: 0.005 mol strong sites and 0.200 mol weak sites per mole of Fe, formula
weight 89 g/mol, specific surface area 600 m2/g. Note that this reproduces the 1:40
strong:weak ratio used in the simulation design.

Total dissolved carbon for the open-system experiments is estimated from the reported
CO2 partial pressure at each pH via Henry's law and the carbonic-acid speciation, so it
is an approximation rather than a reported value; the resulting AD scores near the
carbonate axis should be read with that in mind.

Run:  python ad_screen_published_experiments.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "uvi_surrogate" / "artifacts"
CONDITIONS = HERE / "published_experiment_conditions.csv"
OUT = HERE / "ad_screen_results.csv"

# Dzombak & Morel (1990) HFO conventions
STRONG_PER_MOL_FE = 0.005
WEAK_PER_MOL_FE = 0.200
HFO_FORMULA_WEIGHT = 89.0  # g/mol, gives 1 mmol/L Fe = 0.089 g/L, matching Mahoney Table 1

# 25 C carbonic acid system, infinite dilution
K_HENRY = 10 ** -1.47   # mol/(L*atm), CO2(g) = CO2(aq)
K_A1 = 10 ** -6.35
K_A2 = 10 ** -10.33

PH_GRID = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def total_carbon_from_pco2(log_pco2: float, pH: float) -> float:
    """Total dissolved inorganic carbon for a system open to a fixed CO2 pressure."""
    h = 10 ** -pH
    co2_aq = K_HENRY * 10 ** log_pco2
    return co2_aq * (1 + K_A1 / h + K_A1 * K_A2 / h ** 2)


def load_domain():
    manifest = json.load(open(ART / "manifest.json", encoding="utf-8"))
    ad = np.load(ART / "ad_reference.npz")
    nn = NearestNeighbors(n_neighbors=5).fit(ad["reference"])
    return manifest, ad["mu"], ad["sd"], nn


def feature_vector(row: dict, manifest) -> np.ndarray:
    values = [row[c] for c in manifest["base_features"]]
    values += [np.log10(max(row[c], 1e-12)) for c in manifest["log_columns"]]
    return np.asarray(values, dtype=float)


def main():
    manifest, mu, sd, nn = load_domain()
    threshold = manifest["ad_threshold_p95"]
    conditions = pd.read_csv(CONDITIONS)

    grid_min_hfo_s, grid_max_hfo_s = 1e-6, 1e-3
    grid_min_c, grid_max_c = 1e-5, 1e-2
    grid_min_u, grid_max_u = 1e-8, 1e-3
    grid_min_i, grid_max_i = 1e-3, 1.0

    records = []
    for _, exp in conditions.iterrows():
        fe = float(exp["Fe_total_M"])
        hfo_s = STRONG_PER_MOL_FE * fe
        hfo_w = WEAK_PER_MOL_FE * fe
        for pH in PH_GRID:
            if pd.notna(exp["pCO2_atm_log10"]):
                carbonate = total_carbon_from_pco2(float(exp["pCO2_atm_log10"]), pH)
                carbonate_basis = "open-system estimate"
            else:
                carbonate = float(exp["total_C_M"])
                carbonate_basis = "reported total C"
            row = {
                "Input_pH": pH,
                "U_initial": float(exp["U_total_M"]),
                "Carbonate": max(carbonate, 1e-12),
                "NaCl": float(exp["electrolyte_M"]),
                "Ca": 0.0,
                "Hfo_s": hfo_s,
                "Hfo_w": hfo_w,
            }
            x = (feature_vector(row, manifest) - mu) / sd
            distance = float(nn.kneighbors(x.reshape(1, -1), return_distance=True)[0].mean())

            outside = []
            if not grid_min_hfo_s <= hfo_s <= grid_max_hfo_s:
                outside.append("Hfo_s")
            if not grid_min_c <= row["Carbonate"] <= grid_max_c:
                outside.append("Carbonate")
            if not grid_min_u <= row["U_initial"] <= grid_max_u:
                outside.append("U_initial")
            if not grid_min_i <= row["NaCl"] <= grid_max_i:
                outside.append("NaCl")

            records.append({
                "series": int(exp["series"]),
                "source": exp["source"],
                "pH": pH,
                "U_initial": row["U_initial"],
                "Carbonate": row["Carbonate"],
                "carbonate_basis": carbonate_basis,
                "electrolyte_M": row["NaCl"],
                "Hfo_s": hfo_s,
                "Hfo_w": hfo_w,
                "AD_distance": distance,
                "AD_status": "in-domain" if distance <= threshold else "lower-support",
                "axes_outside_sampled_range": ";".join(outside) if outside else "",
            })

    out = pd.DataFrame(records)
    out["inside_sampled_box"] = out["axes_outside_sampled_range"] == ""
    out.to_csv(OUT, index=False)

    print(f"AD threshold (training-distance p95): {threshold:.4f}")
    print(f"Screened {len(out)} experiment x pH combinations "
          f"from {conditions['series'].nunique()} published data sets.")
    print(f"pH values screened: {PH_GRID}\n")

    print("=" * 78)
    print("STEP 1 - is the condition inside the sampled ranges of the design at all?")
    print("=" * 78)
    inbox = out.groupby(["series", "source"])["inside_sampled_box"].mean().round(2)
    print(inbox.rename("fraction of pH values inside sampled box").to_string())
    print()
    gaps = (out.loc[~out["inside_sampled_box"]]
               .groupby("axes_outside_sampled_range").size().rename("n"))
    print("Which axis puts them outside:")
    print(gaps.to_string())
    print()

    print("=" * 78)
    print("STEP 2 - AD distance, restricted to conditions inside the sampled box")
    print("=" * 78)
    usable = out[out["inside_sampled_box"]]
    if len(usable):
        print(usable["AD_status"].value_counts().to_string())
        print()
        per_series = (usable.groupby(["series", "source"])
                        .agg(n=("AD_distance", "size"),
                             mean_AD=("AD_distance", "mean"),
                             max_AD=("AD_distance", "max"),
                             frac_lower_support=("AD_status",
                                                 lambda s: (s == "lower-support").mean()))
                        .round(3))
        print(per_series.to_string())
    print()

    fully = inbox[inbox == 1.0]
    print(f"Data sets usable at every screened pH without leaving the sampled box: "
          f"{len(fully)} of {conditions['series'].nunique()}")
    for (series, source) in fully.index:
        sub = out[(out.series == series)]
        print(f"  series {series:>2d}  {source}  "
              f"(max AD {sub.AD_distance.max():.3f}, "
              f"{(sub.AD_status == 'lower-support').sum()} pH values flagged lower-support)")

    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
