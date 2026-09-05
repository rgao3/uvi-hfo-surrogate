"""Parse Appendix 1 of Payne (1999) into a tidy CSV of measured U(VI) sorption on ferrihydrite.

Source
------
Payne, T.E., 1999. Uranium (VI) interactions with mineral surfaces: controlling factors
and surface complexation modelling. PhD thesis, UNSW Sydney.
https://doi.org/10.26190/unsworks/19387  (open access, CC BY-NC-ND 3.0 AU)

Appendix 1, pages A1-1 and A1-2, tabulates the ligand-free ferrihydrite data sets. Later
appendix pages cover phosphate, citrate, humic acid and sulfate systems, and kaolinite;
those are outside the scope of the present surface-complexation model and are not parsed.

Layout
------
Each appendix page holds three or four tables side by side, so `pdftotext -layout` output
has to be cut into vertical strips before it can be read. Within a strip the parser walks
downwards, treating any line that is a bare "pH  value" pair as data and anything else as
part of the header for the block that follows. Conditions are read out of the header text
(U=, Fe=, I=, elevated pCO2), falling back to the page-level statement of 0.1 M NaNO3 and
air equilibration. Blocks with identical conditions in the same strip are merged, which
reunites the tables split by a "(cont)" break.

Integrity check
---------------
Five of the parsed blocks correspond to series 5-9 of Table 1 in Mahoney, Cadle &
Jakubowski (2009), Environ. Sci. Technol. 43, 9260-9266. Their reported point counts are
used as an independent check on the parse.

Output: payne_1999_ferrihydrite_measured.csv
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PDF = HERE / "Payne_1999_thesis.pdf"
TXT = HERE / "_payne_layout.txt"
OUT = HERE / "payne_1999_ferrihydrite_measured.csv"

DATA_ROW = re.compile(r"^\s*(\d{1,2}\.\d{1,2})\s+(\d{1,3}(?:\.\d)?)\s*$")

# vertical strips, in characters, for each appendix page
STRIPS = {
    1: [(0, 30), (30, 60), (60, 100)],
    2: [(0, 26), (26, 52), (52, 78), (78, 115)],
}

# point counts reported by Mahoney et al. (2009), Table 1, series 5-9
MAHONEY_COUNTS = {
    (1e-6, 1e-3, 0.1, -3.5): (40, "series 5"),
    (1e-6, 2e-2, 0.1, -3.5): (20, "series 6"),
    (1e-5, 1e-3, 0.1, -3.5): (18, "series 7"),
    (1e-4, 1e-3, 0.1, -3.5): (17, "series 8"),
    (1e-6, 1e-3, 0.1, -2.0): (15, "series 9"),
}


def layout_text() -> list[str]:
    if not TXT.exists():
        subprocess.run(["pdftotext", "-layout", str(PDF), str(TXT)], check=True)
    return TXT.read_text(encoding="utf-8", errors="ignore").split("\n")


def appendix_pages(lines: list[str]) -> dict[int, list[str]]:
    """Return the raw lines of appendix pages A1-1 and A1-2."""
    anchor = next(i for i, l in enumerate(lines) if l.strip() == "U-sorption data sets")
    pages, cursor = {}, anchor
    for page in (1, 2):
        end = next(i for i, l in enumerate(lines[cursor:], cursor) if f"Page A1-{page}" in l)
        pages[page] = lines[cursor:end]
        cursor = end + 1
    return pages


def read_conditions(header: str) -> dict | None:
    """Pull experimental conditions out of a block header. None if it is not a data header."""
    flat = " ".join(header.split())
    u = re.search(r"U\s*=\s*1e-(\d+)", flat)
    if not u:
        return None
    fe = re.search(r"Fe\s*=\s*([\d.]+)\s*mmol", flat, re.I)
    if not fe:
        return None
    ionic = re.search(r"I\s*=\s*([\d.]+)", flat)
    pco2 = re.search(r"=\s*1e-(\d+)\s*atm", flat)
    return {
        "U_total_M": 10.0 ** -int(u.group(1)),
        "Fe_total_M": float(fe.group(1)) * 1e-3,
        "ionic_strength_M": float(ionic.group(1)) if ionic else 0.1,
        "log10_pCO2_atm": -float(pco2.group(1)) if pco2 else -3.5,
    }


def parse_strip(page_lines: list[str], lo: int, hi: int) -> list[tuple[dict, list]]:
    """Segment one vertical strip into (conditions, points) blocks."""
    blocks, header, points = [], [], []
    last_cond: dict | None = None

    def flush():
        nonlocal last_cond
        if points:
            text = " ".join(header)
            cond = read_conditions(text)
            if cond is None and last_cond is not None and "cont" in text.lower():
                # a "(cont)" header repeats only the label, so carry the conditions over
                cond = dict(last_cond)
            if cond:
                blocks.append((cond, list(points)))
                last_cond = cond
        points.clear()

    for line in page_lines:
        cell = line[lo:hi]
        m = DATA_ROW.match(cell)
        if m:
            points.append((float(m.group(1)), float(m.group(2))))
        elif cell.strip():
            if points:          # header text after data means a new block starts
                flush()
                header = []
            header.append(cell.strip())
    flush()

    # merge blocks with identical conditions (rejoins "(cont)" tables)
    merged: list[tuple[dict, list]] = []
    for cond, pts in blocks:
        for prev_cond, prev_pts in merged:
            if prev_cond == cond:
                prev_pts.extend(pts)
                break
        else:
            merged.append((cond, pts))
    return merged


def main():
    pages = appendix_pages(layout_text())

    collected: dict[tuple, list] = {}
    duplicates = []
    for page, strips in STRIPS.items():
        for lo, hi in strips:
            for cond, pts in parse_strip(pages[page], lo, hi):
                key = (cond["U_total_M"], cond["Fe_total_M"],
                       cond["ionic_strength_M"], cond["log10_pCO2_atm"])
                if key in collected:
                    # page A1-2 reprints the standard 0.1 M set as a reference column
                    duplicates.append((page, key, len(pts), len(collected[key])))
                    continue
                collected[key] = pts

    records = []
    for (u, fe, ionic, pco2), pts in collected.items():
        label = f"U{u:.0e}_Fe{fe*1e3:g}mM_I{ionic:g}_pCO2{pco2:g}".replace("e-0", "e-")
        for pH, pct in pts:
            records.append({
                "dataset": label,
                "pH": pH,
                "U_sorbed_pct": pct,
                "U_total_M": u,
                "Fe_total_M": fe,
                "ionic_strength_M": ionic,
                "log10_pCO2_atm": pco2,
                "electrolyte": "NaNO3",
            })

    df = pd.DataFrame(records).sort_values(["dataset", "pH"]).reset_index(drop=True)
    df.to_csv(OUT, index=False)

    print("Parsed blocks")
    print("-" * 74)
    print(f"{'U (M)':>8s} {'Fe (M)':>8s} {'I (M)':>7s} {'log pCO2':>9s} {'n':>4s}  cross-check")
    for key in sorted(collected, key=lambda k: (k[3], k[1], k[0])):
        n = len(collected[key])
        want, series = MAHONEY_COUNTS.get(key, (None, ""))
        if want is None:
            check = "not in Mahoney compilation"
        else:
            check = f"Mahoney {series}: {want} -> {'MATCH' if want == n else 'MISMATCH'}"
        print(f"{key[0]:8.0e} {key[1]:8.0e} {key[2]:7g} {key[3]:9g} {n:4d}  {check}")

    if duplicates:
        print("\nDuplicate columns skipped (same conditions reprinted on another page):")
        for page, key, n_dup, n_kept in duplicates:
            print(f"  page A1-{page}: U={key[0]:.0e} Fe={key[1]:.0e} I={key[2]:g} "
                  f"- {n_dup} points, already have {n_kept}")

    print(f"\nTotal unique measured points: {len(df)}")
    print(f"pH range {df.pH.min():.2f} to {df.pH.max():.2f}; "
          f"adsorption {df.U_sorbed_pct.min():.1f} to {df.U_sorbed_pct.max():.1f} %")
    print(f"Written: {OUT}")


if __name__ == "__main__":
    main()
