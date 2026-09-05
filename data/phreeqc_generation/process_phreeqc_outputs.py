#!/usr/bin/env python3
"""
Prepare PHREEQC U(VI)-HFO adsorption outputs for downstream modeling.

The script reads PHREEQC text output files, extracts one row per simulation
condition, and writes an Excel workbook with a modeling-ready "results" sheet.
The calculations follow the project workbook convention:

    U_ads = sum(U-bearing surface-complex molalities)
    Ads_% = U_ads / U_total * 100
    HFO_g_L = (Hfo_sOH + Hfo_wOH) / site_density
    Kd = (U_ads / HFO_g_L) / U_diss
    logKd = log10(Kd)
    pe = 20.8 - pH

Only the Python standard library is used so the script is easy to share and
re-run without environment setup.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape


DEFAULT_OUTPUTS_DIR = Path("outputs")
DEFAULT_WORKBOOK = Path("phreeqc_processed_results.xlsx")
DEFAULT_SITE_DENSITY_MOL_PER_G = 0.005
PE_PLUS_PH = 20.8

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"

SURFACE_COMPLEXES = (
    "Hfo_sOUO2+",
    "(Hfo_sO)2UO2",
    "Hfo_wOUO2+",
    "(Hfo_wO)2UO2",
    "(Hfo_sO)2UO2CO3-2",
    "(Hfo_wO)2UO2CO3-2",
)

RESULT_COLUMNS = (
    "ID",
    "SourceFile",
    "SimulationInFile",
    "pH",
    "pe",
    "Hfo_sOH",
    "Hfo_wOH",
    "U_total_M",
    "C_total_M",
    "NaCl_M",
    "Ca_M",
    "U_ads_M",
    "U_diss_M",
    "HFO_g_L",
    "Ads_percent",
    "Kd_L_g",
    "logKd",
    "Dominant_aq_species",
    "Dominant_surface_species",
)

SUMMARY_COLUMNS = (
    "SourceFile",
    "SimulationsParsed",
    "pHValues",
    "UValues",
    "CValues",
    "Hfo_sOH",
    "Hfo_wOH",
)

METHOD_ROWS = (
    ("PHREEQC block", "One output row is generated for each PHREEQC simulation block, not each text file."),
    ("U_ads_M", "Sum of the six U-bearing HFO surface complexes listed in SURFACE_COMPLEXES."),
    ("U_diss_M", "Final U value from the last Solution composition section in each simulation block."),
    ("Ads_percent", "U_ads_M / U_total_M * 100."),
    ("HFO_g_L", "(Hfo_sOH + Hfo_wOH) / site_density_mol_per_g."),
    ("Kd_L_g", "(U_ads_M / HFO_g_L) / U_diss_M."),
    ("logKd", "log10(Kd_L_g)."),
    ("pe", "Calculated as 20.8 - pH for consistency with the project design matrix."),
)


@dataclass(frozen=True)
class SimulationResult:
    ID: int
    SourceFile: str
    SimulationInFile: int
    pH: float
    pe: float
    Hfo_sOH: float
    Hfo_wOH: float
    U_total_M: float
    C_total_M: float | None
    NaCl_M: float | None
    Ca_M: float
    U_ads_M: float
    U_diss_M: float
    HFO_g_L: float
    Ads_percent: float | None
    Kd_L_g: float | None
    logKd: float | None
    Dominant_aq_species: str
    Dominant_surface_species: str


@dataclass(frozen=True)
class FileSummary:
    SourceFile: str
    SimulationsParsed: int
    pHValues: str
    UValues: str
    CValues: str
    Hfo_sOH: str
    Hfo_wOH: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract modeling-ready U(VI)-HFO adsorption data from PHREEQC output files."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help=f"Directory containing PHREEQC .txt output files. Default: {DEFAULT_OUTPUTS_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_WORKBOOK,
        help=f"Excel workbook to create. Default: {DEFAULT_WORKBOOK}",
    )
    parser.add_argument(
        "--site-density",
        type=float,
        default=DEFAULT_SITE_DENSITY_MOL_PER_G,
        help="HFO site density in mol sites per gram HFO. Default: 0.005",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Preferred text encoding for PHREEQC output files. Default: utf-8",
    )
    return parser.parse_args(argv)


def read_text_lines(path: Path, encoding: str) -> list[str]:
    try:
        return path.read_text(encoding=encoding).splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding=encoding, errors="replace").splitlines()


def unique_sorted(values: Iterable[float | None]) -> str:
    clean_values = sorted({value for value in values if value is not None})
    return ", ".join(format_float(value) for value in clean_values)


def format_float(value: float) -> str:
    return f"{value:.10g}"


def extract_input_value(lines: Iterable[str], name: str) -> float | None:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s+({NUMBER_PATTERN})")
    for line in lines:
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def extract_surface_complex(lines: Iterable[str], species: str) -> float:
    pattern = re.compile(rf"^\s*{re.escape(species)}\s+({NUMBER_PATTERN})")
    value = 0.0
    for line in lines:
        match = pattern.search(line)
        if match:
            value = float(match.group(1))
    return value


def extract_final_solution_u(lines: Sequence[str]) -> float | None:
    start = last_index_containing(lines, "Solution composition")
    if start is None:
        return None

    pattern = re.compile(rf"^\s*U\s+({NUMBER_PATTERN})")
    for line in lines[start : start + 40]:
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def extract_dominant_aqueous_species(lines: Sequence[str]) -> str:
    """Return the largest U(6) aqueous species from the final species table."""
    u6_start = None
    for index, line in enumerate(lines):
        if re.search(r"^\s*U\(6\)\s+", line):
            u6_start = index
    if u6_start is None:
        return ""

    species_pattern = re.compile(rf"^\s+(.+?)\s+({NUMBER_PATTERN})\s+")
    best_species = ""
    best_value = -1.0

    for line in lines[u6_start + 1 :]:
        if line.startswith("-") or re.search(r"^\S", line):
            break
        match = species_pattern.search(line)
        if not match:
            continue
        species = match.group(1).strip()
        value = float(match.group(2))
        if value > best_value:
            best_species = species
            best_value = value

    return best_species


def last_index_containing(lines: Sequence[str], text: str) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if text in lines[index]:
            return index
    return None


def split_simulation_blocks(lines: Sequence[str]) -> list[list[str]]:
    starts = [
        index
        for index, line in enumerate(lines)
        if "Reading input data for simulation 1." in line
    ]
    blocks: list[list[str]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        blocks.append(list(lines[start:end]))
    return blocks


def parse_simulation_block(
    block: Sequence[str],
    source_file: str,
    simulation_number: int,
    result_id: int,
    site_density_mol_per_g: float,
) -> SimulationResult | None:
    ph = extract_input_value(block, "pH")
    hfo_s = extract_input_value(block, "Hfo_sOH")
    hfo_w = extract_input_value(block, "Hfo_wOH")
    u_total = extract_input_value(block, "U")

    if ph is None or hfo_s is None or hfo_w is None or u_total is None:
        return None

    c_total = extract_input_value(block, "C")
    nacl = extract_input_value(block, "Na")
    ca = extract_input_value(block, "Ca") or 0.0
    u_diss = extract_final_solution_u(block)
    if u_diss is None:
        return None

    surface_values = {
        species: extract_surface_complex(block, species) for species in SURFACE_COMPLEXES
    }
    u_ads = sum(surface_values.values())
    hfo_g_l = (hfo_s + hfo_w) / site_density_mol_per_g
    ads_percent = u_ads / u_total * 100.0 if u_total > 0 else None
    kd = (u_ads / hfo_g_l) / u_diss if hfo_g_l > 0 and u_diss > 0 else None
    log_kd = math.log10(kd) if kd is not None and kd > 0 else None
    dominant_surface = max(surface_values, key=surface_values.get)

    return SimulationResult(
        ID=result_id,
        SourceFile=source_file,
        SimulationInFile=simulation_number,
        pH=ph,
        pe=PE_PLUS_PH - ph,
        Hfo_sOH=hfo_s,
        Hfo_wOH=hfo_w,
        U_total_M=u_total,
        C_total_M=c_total,
        NaCl_M=nacl,
        Ca_M=ca,
        U_ads_M=u_ads,
        U_diss_M=u_diss,
        HFO_g_L=hfo_g_l,
        Ads_percent=ads_percent,
        Kd_L_g=kd,
        logKd=log_kd,
        Dominant_aq_species=extract_dominant_aqueous_species(block),
        Dominant_surface_species=dominant_surface,
    )


def parse_output_directory(
    outputs_dir: Path,
    site_density_mol_per_g: float,
    encoding: str,
) -> list[SimulationResult]:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {outputs_dir}")

    results: list[SimulationResult] = []
    next_id = 1

    for txt_file in sorted(outputs_dir.glob("*.txt")):
        lines = read_text_lines(txt_file, encoding)
        blocks = split_simulation_blocks(lines)
        for simulation_number, block in enumerate(blocks, start=1):
            result = parse_simulation_block(
                block=block,
                source_file=txt_file.name,
                simulation_number=simulation_number,
                result_id=next_id,
                site_density_mol_per_g=site_density_mol_per_g,
            )
            if result is None:
                continue
            results.append(result)
            next_id += 1

    return results


def build_file_summary(results: Sequence[SimulationResult]) -> list[FileSummary]:
    grouped: dict[str, list[SimulationResult]] = {}
    for result in results:
        grouped.setdefault(result.SourceFile, []).append(result)

    summary: list[FileSummary] = []
    for source_file in sorted(grouped):
        rows = grouped[source_file]
        summary.append(
            FileSummary(
                SourceFile=source_file,
                SimulationsParsed=len(rows),
                pHValues=unique_sorted(row.pH for row in rows),
                UValues=unique_sorted(row.U_total_M for row in rows),
                CValues=unique_sorted(row.C_total_M for row in rows),
                Hfo_sOH=unique_sorted(row.Hfo_sOH for row in rows),
                Hfo_wOH=unique_sorted(row.Hfo_wOH for row in rows),
            )
        )
    return summary


def validate_results(results: Sequence[SimulationResult]) -> list[str]:
    warnings: list[str] = []
    if not results:
        warnings.append("No simulation rows were parsed.")
        return warnings

    for row in results:
        if not math.isclose(row.pH + row.pe, PE_PLUS_PH, rel_tol=0.0, abs_tol=1e-10):
            warnings.append(f"pe + pH check failed for row {row.ID}.")
        if row.U_ads_M < 0 or row.U_diss_M < 0:
            warnings.append(f"Negative uranium amount found in row {row.ID}.")
        if row.Ads_percent is not None and row.Ads_percent > 100.0 + 1e-6:
            warnings.append(f"Adsorption percent exceeds 100% in row {row.ID}.")
    return warnings


def excel_column(index: int) -> str:
    column = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        column = chr(ord("A") + remainder) + column
    return column


def cell_xml(reference: str, value: object) -> str:
    if value is None:
        return f'<c r="{reference}" t="inlineStr"><is><t></t></is></c>'
    if isinstance(value, float):
        if not math.isfinite(value):
            return f'<c r="{reference}" t="inlineStr"><is><t></t></is></c>'
        return f'<c r="{reference}"><v>{value:.17g}</v></c>'
    if isinstance(value, int):
        return f'<c r="{reference}"><v>{value}</v></c>'
    return f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def sheet_xml(columns: Sequence[str], rows: Sequence[dict[str, object]]) -> str:
    xml_rows = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>',
    ]
    header_cells = [
        cell_xml(f"{excel_column(index)}1", column)
        for index, column in enumerate(columns, start=1)
    ]
    xml_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_number, row in enumerate(rows, start=2):
        cells = [
            cell_xml(f"{excel_column(index)}{row_number}", row.get(column))
            for index, column in enumerate(columns, start=1)
        ]
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    xml_rows.append("</sheetData></worksheet>")
    return "\n".join(xml_rows)


def workbook_xml(sheet_names: Sequence[str]) -> str:
    sheets = "".join(
        f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
        for index, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets}</sheets></workbook>"
    )


def workbook_relationships_xml(sheet_names: Sequence[str]) -> str:
    relationships = "".join(
        '<Relationship '
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index, _ in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{relationships}</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    worksheet_overrides = "".join(
        '<Override '
        f'PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        f"{worksheet_overrides}</Types>"
    )


def root_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )


def write_workbook(
    output_path: Path,
    results: Sequence[SimulationResult],
    summary: Sequence[FileSummary],
    site_density_mol_per_g: float,
) -> None:
    method_rows = [
        {"Item": item, "Description": description}
        for item, description in METHOD_ROWS
    ]
    method_rows.append(
        {
            "Item": "site_density_mol_per_g",
            "Description": format_float(site_density_mol_per_g),
        }
    )

    sheets = {
        "results": sheet_xml(RESULT_COLUMNS, [asdict(row) for row in results]),
        "file_summary": sheet_xml(SUMMARY_COLUMNS, [asdict(row) for row in summary]),
        "method": sheet_xml(("Item", "Description"), method_rows),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        workbook.writestr("_rels/.rels", root_relationships_xml())
        workbook.writestr("xl/workbook.xml", workbook_xml(tuple(sheets)))
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_relationships_xml(tuple(sheets)))
        for index, sheet_body in enumerate(sheets.values(), start=1):
            workbook.writestr(f"xl/worksheets/sheet{index}.xml", sheet_body)


def print_run_report(results: Sequence[SimulationResult], warnings: Sequence[str], output: Path) -> None:
    counts = Counter(result.SourceFile for result in results)
    print(f"Parsed {len(results)} simulations from {len(counts)} PHREEQC output files.")
    print(f"Wrote workbook: {output.resolve()}")

    if counts:
        print("\nRows per source file:")
        for source_file in sorted(counts):
            print(f"  {source_file}: {counts[source_file]}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        results = parse_output_directory(
            outputs_dir=args.outputs_dir,
            site_density_mol_per_g=args.site_density,
            encoding=args.encoding,
        )
        summary = build_file_summary(results)
        warnings = validate_results(results)
        write_workbook(args.output, results, summary, args.site_density)
        print_run_report(results, warnings, args.output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
