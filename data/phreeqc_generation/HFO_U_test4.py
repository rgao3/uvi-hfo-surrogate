import math
import itertools

import numpy as np
import pandas as pd
from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc

# ============================================================
# Load IPhreeqc + Database
# ============================================================
phreeqc = IPhreeqc()
import os
# JAEA thermodynamic database (201203c0.tdb) must be obtained separately from JAEA;
# set its path via the JAEA_DATABASE environment variable.
phreeqc.load_database(os.environ.get("JAEA_DATABASE", "201203c0.tdb"))

# ============================================================
# Custom HFO Surface Model
# ============================================================
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

phreeqc.run_string(HFO_DEFINITION)

# ============================================================
# Parameter Space  (practical ML-dataset size ~17,280 runs)
# ============================================================
pH_values = np.arange(3, 11.5, 1.0)                        # 9 values
U_values  = np.logspace(-8, -3, 8)                          # 8 values
C_values  = np.logspace(-5, -2, 8)                          # 8 values
Na_values = np.logspace(-3,  0, 6)                          # 6 values
Ca_values = [0.0] + list(np.logspace(-5, -2, 4))            # 5 values

# HFO strong:weak sites in Dzombak & Morel 1:40 ratio
strong_sites = np.logspace(-6, -3, 5)                       # 5 values
HFO_cases = [(s, s * 40) for s in strong_sites]

# ============================================================
# SELECTED_OUTPUT
# ============================================================
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
        β-UO2(OH)2
END
"""

# ============================================================
# Run All Simulations
# ============================================================
results = []
counter = 0
total_runs = (
    len(pH_values) * len(U_values) * len(C_values)
    * len(Na_values) * len(Ca_values) * len(HFO_cases)
)

for pH, U, C, Na, Ca, (Hfo_s, Hfo_w) in itertools.product(
        pH_values, U_values, C_values, Na_values, Ca_values, HFO_cases):

    counter += 1
    pe = 20.8 - pH

    simulation = f"""
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

    try:
        phreeqc.run_string(simulation)
        output = phreeqc.get_selected_output_array()

        if len(output) < 2:
            continue

        headers = output[0]
        row     = output[-1]
        row_dict = dict(zip(headers, row))

        # ── Surface species ────────────────────────────────
        ads_s_monodentate = row_dict.get("m_Hfo_sOUO2+(mol/kgw)",        0)
        ads_s_bidentate   = row_dict.get("m_(Hfo_sO)2UO2(mol/kgw)",      0)
        ads_s_carb        = row_dict.get("m_(Hfo_sO)2UO2CO3-2(mol/kgw)", 0)
        ads_w_monodentate = row_dict.get("m_Hfo_wOUO2+(mol/kgw)",        0)
        ads_w_bidentate   = row_dict.get("m_(Hfo_wO)2UO2(mol/kgw)",      0)
        ads_w_carb        = row_dict.get("m_(Hfo_wO)2UO2CO3-2(mol/kgw)", 0)

        surface_adsorbed_U = (
            ads_s_monodentate + ads_s_bidentate + ads_s_carb
            + ads_w_monodentate + ads_w_bidentate + ads_w_carb
        )

        # ── Derived quantities ─────────────────────────────
        U_ads  = surface_adsorbed_U
        U_diss = row_dict.get("U(mol/kgw)", 0)

        Ads_pct = 100 * U_ads / U if U > 0 else None

        total_sites = Hfo_s + Hfo_w
        HFO_gL = total_sites / 0.005          # Dzombak & Morel: 0.005 mol sites/g

        Kd = (U_ads / U_diss) / HFO_gL if (U_diss > 0 and HFO_gL > 0) else None

        logKd = math.log10(Kd) if (Kd and Kd > 0) else None

        # ── Dominant aqueous species ───────────────────────
        aq_species = {
            "UO2+2":          row_dict.get("m_UO2+2(mol/kgw)",          0),
            "UO2OH+":         row_dict.get("m_UO2OH+(mol/kgw)",         0),
            "UO2(OH)2":       row_dict.get("m_UO2(OH)2(mol/kgw)",       0),
            "UO2CO3":         row_dict.get("m_UO2CO3(mol/kgw)",         0),
            "UO2(CO3)2-2":    row_dict.get("m_UO2(CO3)2-2(mol/kgw)",   0),
            "UO2(CO3)3-4":    row_dict.get("m_UO2(CO3)3-4(mol/kgw)",   0),
            "UO2Cl+":         row_dict.get("m_UO2Cl+(mol/kgw)",         0),
            "CaUO2(CO3)3-2":  row_dict.get("m_CaUO2(CO3)3-2(mol/kgw)", 0),
            "Ca2UO2(CO3)3":   row_dict.get("m_Ca2UO2(CO3)3(mol/kgw)",  0),
        }
        dominant_aq_species = max(aq_species, key=aq_species.get)

        # ── Dominant surface species ───────────────────────
        surface_species = {
            "Hfo_sOUO2+":         ads_s_monodentate,
            "(Hfo_sO)2UO2":       ads_s_bidentate,
            "(Hfo_sO)2UO2CO3-2":  ads_s_carb,
            "Hfo_wOUO2+":         ads_w_monodentate,
            "(Hfo_wO)2UO2":       ads_w_bidentate,
            "(Hfo_wO)2UO2CO3-2":  ads_w_carb,
        }
        dominant_surface_species = max(surface_species, key=surface_species.get)

        # ── Append row ─────────────────────────────────────
        results.append({
            # Input conditions
            "Input_pH":  pH,
            "Input_pe":  pe,
            "U_initial": U,
            "Carbonate": C,
            "NaCl":      Na,
            "Ca":        Ca,
            "Hfo_s":     Hfo_s,
            "Hfo_w":     Hfo_w,

            # PHREEQC output conditions
            "Output_pH": row_dict.get("pH"),
            "Output_pe": row_dict.get("pe"),

            # Uranium partitioning
            "U_ads":   U_ads,
            "U_diss":  U_diss,
            "HFO_gL":  HFO_gL,
            "Ads_%":   Ads_pct,
            "Kd_L_g":  Kd,
            "logKd":   logKd,

            # Surface speciation
            "U_surface_total":       surface_adsorbed_U,
            "Hfo_sOUO2+":            ads_s_monodentate,
            "(Hfo_sO)2UO2":          ads_s_bidentate,
            "(Hfo_sO)2UO2CO3-2":     ads_s_carb,
            "Hfo_wOUO2+":            ads_w_monodentate,
            "(Hfo_wO)2UO2":          ads_w_bidentate,
            "(Hfo_wO)2UO2CO3-2":     ads_w_carb,

            # Aqueous speciation
            "UO2+2":          row_dict.get("m_UO2+2(mol/kgw)"),
            "UO2OH+":         row_dict.get("m_UO2OH+(mol/kgw)"),
            "UO2(OH)2":       row_dict.get("m_UO2(OH)2(mol/kgw)"),
            "UO2CO3":         row_dict.get("m_UO2CO3(mol/kgw)"),
            "UO2(CO3)2-2":    row_dict.get("m_UO2(CO3)2-2(mol/kgw)"),
            "UO2(CO3)3-4":    row_dict.get("m_UO2(CO3)3-4(mol/kgw)"),
            "UO2Cl+":         row_dict.get("m_UO2Cl+(mol/kgw)"),
            "CaUO2(CO3)3-2":  row_dict.get("m_CaUO2(CO3)3-2(mol/kgw)"),
            "Ca2UO2(CO3)3":   row_dict.get("m_Ca2UO2(CO3)3(mol/kgw)"),

            # Saturation indices
            "SI_Na2U2O7":     row_dict.get("si_Na2U2O7(cr)"),
            "SI_UO2CO3":      row_dict.get("si_UO2CO3(cr)"),
            "SI_beta_UO2OH2": row_dict.get("si_β-UO2(OH)2"),

            # Dominant species labels
            "Dominant_aq_species":      dominant_aq_species,
            "Dominant_surface_species": dominant_surface_species,
        })

        if counter % 100 == 0:
            print(f"Progress: {counter}/{total_runs} completed")

    except Exception as e:
        print(f"\nFAILED {counter} | pH={pH} U={U} C={C} Na={Na} Ca={Ca} Hfo=({Hfo_s},{Hfo_w})")
        print(e)

# ============================================================
# Save Results
# ============================================================
df = pd.DataFrame(results)
df.to_csv("U_HFO_ML_Dataset_Final.csv", index=False)

print(f"\nDone. {len(df)} rows saved.")
print(df.head())
