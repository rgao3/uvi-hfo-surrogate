import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc


HERE = Path(__file__).resolve().parent
DEFAULT_DATABASE = HERE.parent / "input_database" / "input_database" / "201203c0.tdb"
DEFAULT_EXISTING = HERE / "U_HFO_ML_Dataset_Final.csv"
DEFAULT_OUTPUT = HERE / "U_HFO_ML_Dataset_Final_completed.csv"
DEFAULT_FAILED = HERE / "U_HFO_ML_Dataset_Final_failed_cases.csv"

SITE_DENSITY_MOL_PER_G = 0.005
PE_PLUS_PH = 20.8


HFO_DEFINITION = """
PHASES
Fix_pH
    H+ = H+
    log_k 0.0
END

SURFACE_MASTER_SPECIES
    Hfo_s  Hfo_sOH
    Hfo_w  Hfo_wOH

SURFACE_SPECIES
    Hfo_sOH = Hfo_sOH
        log_k 0.0
    Hfo_wOH = Hfo_wOH
        log_k 0.0

    Hfo_sOH + UO2+2 = Hfo_sOUO2+ + H+
        log_k 0.46
    2Hfo_sOH + UO2+2 = (Hfo_sO)2UO2 + 2H+
        log_k -2.50
    Hfo_wOH + UO2+2 = Hfo_wOUO2+ + H+
        log_k -2.60
    2Hfo_wOH + UO2+2 = (Hfo_wO)2UO2 + 2H+
        log_k -6.00

    2Hfo_sOH + UO2+2 + CO3-2 = (Hfo_sO)2UO2CO3-2 + 2H+
        log_k -12.30
    2Hfo_wOH + UO2+2 + CO3-2 = (Hfo_wO)2UO2CO3-2 + 2H+
        log_k -16.30
END
"""


SELECTED_OUTPUT_BLOCK = """
SELECTED_OUTPUT
    -reset false
    -high_precision true

    -pH true
    -pe true

    -totals
        U C Na Cl Ca

    -molalities
        UO2+2
        UO2OH+
        UO2(OH)2
        UO2CO3
        UO2(CO3)2-2
        UO2(CO3)3-4
        UO2Cl+
        CaUO2(CO3)3-2
        Ca2UO2(CO3)3
        Hfo_sOUO2+
        (Hfo_sO)2UO2
        (Hfo_sO)2UO2CO3-2
        Hfo_wOUO2+
        (Hfo_wO)2UO2
        (Hfo_wO)2UO2CO3-2

    -saturation_indices
        Na2U2O7(cr)
        UO2CO3(cr)
        beta-UO2(OH)2
END
"""


OUTPUT_COLUMNS = [
    "Input_pH", "Input_pe", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s", "Hfo_w",
    "Output_pH", "Output_pe", "U_ads", "U_diss", "HFO_gL", "Ads_%", "Kd_L_g", "logKd",
    "U_surface_total",
    "Hfo_sOUO2+", "(Hfo_sO)2UO2", "(Hfo_sO)2UO2CO3-2",
    "Hfo_wOUO2+", "(Hfo_wO)2UO2", "(Hfo_wO)2UO2CO3-2",
    "UO2+2", "UO2OH+", "UO2(OH)2", "UO2CO3", "UO2(CO3)2-2", "UO2(CO3)3-4",
    "UO2Cl+", "CaUO2(CO3)3-2", "Ca2UO2(CO3)3",
    "SI_Na2U2O7", "SI_UO2CO3", "SI_beta_UO2OH2",
    "Dominant_aq_species", "Dominant_surface_species",
]


def key_value(value):
    value = float(value)
    if value == 0:
        return "0"
    return f"{value:.12g}"


def combo_key(combo):
    pH, U, C, Na, Ca, hfo = combo
    Hfo_s, Hfo_w = hfo
    return tuple(key_value(v) for v in (pH, U, C, Na, Ca, Hfo_s, Hfo_w))


def row_key(row):
    return tuple(key_value(row[c]) for c in (
        "Input_pH", "U_initial", "Carbonate", "NaCl", "Ca", "Hfo_s", "Hfo_w"
    ))


def build_combinations():
    pH_values = np.arange(3, 11.5, 1.0)
    U_values = np.logspace(-8, -3, 8)
    C_values = np.logspace(-5, -2, 8)
    Na_values = np.logspace(-3, 0, 6)
    Ca_values = [0.0] + list(np.logspace(-5, -2, 4))
    strong_sites = np.logspace(-6, -3, 5)
    HFO_cases = [(s, s * 40) for s in strong_sites]
    return list(itertools.product(pH_values, U_values, C_values, Na_values, Ca_values, HFO_cases))


def make_engine(database):
    engine = IPhreeqc()
    engine.load_database(str(database))
    engine.run_string(HFO_DEFINITION)
    return engine


def simulation_string(pH, U, C, Na, Ca, Hfo_s, Hfo_w):
    pe = PE_PLUS_PH - pH
    return f"""
DELETE -all

SOLUTION 1
    temp 25
    pH {pH}
    pe {pe}
    redox pe
    units mol/kgw
    Na {Na}
    Cl {Na}
    Ca {Ca}
    C {C}
    U {U}

SURFACE 1
    Hfo_sOH {Hfo_s} 600.0 1.0
    Hfo_wOH {Hfo_w}
    -donnan 1e-8
    -no_edl false

EQUILIBRIUM_PHASES 1
    Fix_pH -{pH} HCl 10
    O2(g) -0.7 10

{SELECTED_OUTPUT_BLOCK}
"""


def get_value(row_dict, name, default=0.0):
    value = row_dict.get(name, default)
    return default if value is None else value


def run_combo(engine, combo):
    pH, U, C, Na, Ca, (Hfo_s, Hfo_w) = combo
    engine.run_string(simulation_string(pH, U, C, Na, Ca, Hfo_s, Hfo_w))
    output = engine.get_selected_output_array()
    if len(output) < 2:
        raise RuntimeError("PHREEQC returned no selected output row.")

    row_dict = dict(zip(output[0], output[-1]))

    ads_s_monodentate = get_value(row_dict, "m_Hfo_sOUO2+(mol/kgw)")
    ads_s_bidentate = get_value(row_dict, "m_(Hfo_sO)2UO2(mol/kgw)")
    ads_s_carb = get_value(row_dict, "m_(Hfo_sO)2UO2CO3-2(mol/kgw)")
    ads_w_monodentate = get_value(row_dict, "m_Hfo_wOUO2+(mol/kgw)")
    ads_w_bidentate = get_value(row_dict, "m_(Hfo_wO)2UO2(mol/kgw)")
    ads_w_carb = get_value(row_dict, "m_(Hfo_wO)2UO2CO3-2(mol/kgw)")

    surface_adsorbed_U = (
        ads_s_monodentate + ads_s_bidentate + ads_s_carb
        + ads_w_monodentate + ads_w_bidentate + ads_w_carb
    )
    U_ads = surface_adsorbed_U
    U_diss = get_value(row_dict, "U(mol/kgw)")
    HFO_gL = (Hfo_s + Hfo_w) / SITE_DENSITY_MOL_PER_G
    Ads_pct = 100 * U_ads / U if U > 0 else None
    Kd = (U_ads / U_diss) / HFO_gL if (U_diss and U_diss > 0 and HFO_gL > 0) else None
    logKd = math.log10(Kd) if (Kd and Kd > 0) else None

    aq_species = {
        "UO2+2": get_value(row_dict, "m_UO2+2(mol/kgw)"),
        "UO2OH+": get_value(row_dict, "m_UO2OH+(mol/kgw)"),
        "UO2(OH)2": get_value(row_dict, "m_UO2(OH)2(mol/kgw)"),
        "UO2CO3": get_value(row_dict, "m_UO2CO3(mol/kgw)"),
        "UO2(CO3)2-2": get_value(row_dict, "m_UO2(CO3)2-2(mol/kgw)"),
        "UO2(CO3)3-4": get_value(row_dict, "m_UO2(CO3)3-4(mol/kgw)"),
        "UO2Cl+": get_value(row_dict, "m_UO2Cl+(mol/kgw)"),
        "CaUO2(CO3)3-2": get_value(row_dict, "m_CaUO2(CO3)3-2(mol/kgw)"),
        "Ca2UO2(CO3)3": get_value(row_dict, "m_Ca2UO2(CO3)3(mol/kgw)"),
    }
    surface_species = {
        "Hfo_sOUO2+": ads_s_monodentate,
        "(Hfo_sO)2UO2": ads_s_bidentate,
        "(Hfo_sO)2UO2CO3-2": ads_s_carb,
        "Hfo_wOUO2+": ads_w_monodentate,
        "(Hfo_wO)2UO2": ads_w_bidentate,
        "(Hfo_wO)2UO2CO3-2": ads_w_carb,
    }

    return {
        "Input_pH": pH,
        "Input_pe": PE_PLUS_PH - pH,
        "U_initial": U,
        "Carbonate": C,
        "NaCl": Na,
        "Ca": Ca,
        "Hfo_s": Hfo_s,
        "Hfo_w": Hfo_w,
        "Output_pH": row_dict.get("pH"),
        "Output_pe": row_dict.get("pe"),
        "U_ads": U_ads,
        "U_diss": U_diss,
        "HFO_gL": HFO_gL,
        "Ads_%": Ads_pct,
        "Kd_L_g": Kd,
        "logKd": logKd,
        "U_surface_total": surface_adsorbed_U,
        "Hfo_sOUO2+": ads_s_monodentate,
        "(Hfo_sO)2UO2": ads_s_bidentate,
        "(Hfo_sO)2UO2CO3-2": ads_s_carb,
        "Hfo_wOUO2+": ads_w_monodentate,
        "(Hfo_wO)2UO2": ads_w_bidentate,
        "(Hfo_wO)2UO2CO3-2": ads_w_carb,
        **aq_species,
        "SI_Na2U2O7": row_dict.get("si_Na2U2O7(cr)"),
        "SI_UO2CO3": row_dict.get("si_UO2CO3(cr)"),
        "SI_beta_UO2OH2": row_dict.get("si_beta-UO2(OH)2"),
        "Dominant_aq_species": max(aq_species, key=aq_species.get),
        "Dominant_surface_species": max(surface_species, key=surface_species.get),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run or complete the HFO/U PHREEQC ML dataset.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-output", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--missing-only", action="store_true",
                        help="Only run combinations missing from --existing.")
    parser.add_argument("--fresh-engine-on-failure", action="store_true",
                        help="Recreate IPhreeqc after a failed run before continuing.")
    return parser.parse_args()


def main():
    args = parse_args()
    all_combos = build_combinations()
    existing_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    existing_keys = set()

    if args.missing_only and args.existing.exists():
        existing_df = pd.read_csv(args.existing)
        existing_keys = {row_key(row) for _, row in existing_df.iterrows()}

    combos_to_run = [combo for combo in all_combos if (not args.missing_only or combo_key(combo) not in existing_keys)]
    print(f"Full grid: {len(all_combos)} combinations")
    print(f"Existing rows: {len(existing_df)}")
    print(f"Combinations to run: {len(combos_to_run)}")

    engine = make_engine(args.database)
    new_rows = []
    failed_rows = []

    for index, combo in enumerate(combos_to_run, start=1):
        try:
            new_rows.append(run_combo(engine, combo))
        except Exception as exc:
            pH, U, C, Na, Ca, (Hfo_s, Hfo_w) = combo
            failed_rows.append({
                "Input_pH": pH,
                "U_initial": U,
                "Carbonate": C,
                "NaCl": Na,
                "Ca": Ca,
                "Hfo_s": Hfo_s,
                "Hfo_w": Hfo_w,
                "error": str(exc),
            })
            print(f"FAILED {index}/{len(combos_to_run)}: pH={pH} U={U} C={C} Na={Na} Ca={Ca} HFO=({Hfo_s},{Hfo_w})")
            print(exc)
            if args.fresh_engine_on_failure:
                engine = make_engine(args.database)

        if index % 100 == 0 or index == len(combos_to_run):
            print(f"Progress: {index}/{len(combos_to_run)}")

    combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    if not combined.empty:
        combined = combined.reindex(columns=OUTPUT_COLUMNS)
    combined.to_csv(args.output, index=False)
    pd.DataFrame(failed_rows).to_csv(args.failed_output, index=False)

    print(f"Saved complete/partial dataset: {args.output} ({len(combined)} rows)")
    print(f"Saved failed cases: {args.failed_output} ({len(failed_rows)} rows)")


if __name__ == "__main__":
    main()
