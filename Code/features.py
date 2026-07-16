#!/usr/bin/env python3
"""
features.py
-----------
Track-A feature engineering for the HEA phase-conditioned generative project.

Turns the *cleaned* dataset into a single model-ready table holding the two
views the project needs:

    * CVAE view      -> element-fraction columns  (frac_<El>), one per palette
                        element, each row summing to 1.
    * Classifier view -> the five physics descriptors already in the data
                        (vec, atom_size_diff, elect_diff, dHmix, dSmix).

It also carries the labels (phase, phase_id, phase_conflict) and a leakage-safe
grouping key (family), and writes the element palette to JSON.

Key steps
    1.  Parse every alloy formula into element fractions. The parser handles
        equiatomic (CoCrFeNi), subscripts (Al0.5NbTaTiV), grouped/nested
        notation ((CoCrCuFeNi)96Nb4, Ni45(FeCoCr)40(AlTi)15) and square
        brackets. En-dash "balance" notation is rejected and dropped.
    2.  Build a DATA-DRIVEN palette: elements appearing in >= MIN_COUNT entries.
        Rare elements (below the threshold) are treated as off-palette.
    3.  Enforce the palette per row: drop rows whose off-palette content is
        >= OFF_PALETTE_TOL; otherwise zero the trace off-palette elements and
        renormalise over the palette.
    4.  Build the fraction matrix, recompute VEC from fractions as a QC check,
        compute the family key, and assemble the output table.

The functions parse_formula() and composition_to_vector() are importable and
are reused later to featurise CVAE-generated compositions.

Usage
    python features.py \
        --input   HEA_Phase_Dataset_v1d_clean.csv \
        --output  HEA_features.csv \
        --palette-out palette.json
"""

from __future__ import annotations
import argparse
import json
import re
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Physics reference tables
# --------------------------------------------------------------------------- #

# Valence electron concentration per element. Used ONLY to recompute VEC from
# parsed fractions as a parser sanity-check (and later to score generated
# compositions). delta / dHmix / dSmix are taken from the dataset, not
# recomputed here (they need binary-interaction tables).
VEC_TABLE = {
    "Al": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12, "Zr": 4, "Nb": 5, "Mo": 6, "Hf": 4, "Ta": 5, "W": 6,
    "Re": 7, "Si": 4, "Sn": 4, "Mg": 2, "C": 4, "N": 5, "Li": 1,
}

DESCRIPTOR_COLS = ["vec", "atom_size_diff", "elect_diff", "dHmix", "dSmix"]

# Defaults
MIN_COUNT = 20          # an element must appear in >= this many entries to earn a palette slot
OFF_PALETTE_TOL = 0.05  # drop a row if >= 5% of its atoms are off-palette
FAMILY_TOL = 0.05       # elements present at >= 5% define the leakage-safe family key


# --------------------------------------------------------------------------- #
# Composition parser
# --------------------------------------------------------------------------- #

_NUM = re.compile(r"\d*\.?\d+")
_EL = re.compile(r"[A-Z][a-z]?")


def parse_formula(s: str) -> dict[str, float]:
    """
    Parse an alloy formula into normalised element fractions (summing to 1).

    Supports: plain (CoCrFeNi), subscripts (Al0.5NbTaTiV), parenthesised /
    nested groups with a group total multiplier ((CoCrCuFeNi)96Nb4), and square
    brackets (treated as parentheses). A group's multiplier is distributed
    across the group in proportion to the group's internal amounts.

    Raises ValueError on en-dash/em-dash "balance" notation or malformed input.
    """
    s = s.replace("[", "(").replace("]", ")")
    if "\u2013" in s or "\u2014" in s:            # en / em dash => balance notation
        raise ValueError("balance/dash notation not supported")
    s = s.replace("-", "")                         # strip stray ASCII hyphens

    def read_num(i: int):
        m = _NUM.match(s, i)
        return (float(m.group()), m.end()) if m else (None, i)

    def group(i: int):
        counts: dict[str, float] = {}
        while i < len(s) and s[i] != ")":
            if s[i] == "(":
                sub, i = group(i + 1)
                if i >= len(s) or s[i] != ")":
                    raise ValueError("unbalanced parentheses")
                i += 1
                mult, i = read_num(i)
                if mult is not None:               # rescale group to the given total
                    ssum = sum(sub.values())
                    if ssum > 0:
                        for k in sub:
                            sub[k] *= mult / ssum
                for k, v in sub.items():
                    counts[k] = counts.get(k, 0.0) + v
            else:
                m = _EL.match(s, i)
                if not m:
                    raise ValueError(f"unexpected token near '{s[i:]}'")
                el = m.group(); i = m.end()
                n, i = read_num(i)
                counts[el] = counts.get(el, 0.0) + (n if n is not None else 1.0)
        return counts, i

    counts, _ = group(0)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("empty composition")
    return {k: v / total for k, v in counts.items()}


def composition_to_vector(frac: dict[str, float], palette: list[str]) -> np.ndarray:
    """Project an element-fraction dict onto the palette and renormalise to 1."""
    vec = np.array([frac.get(el, 0.0) for el in palette], dtype="float64")
    s = vec.sum()
    return vec / s if s > 0 else vec


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def build(path_in: str, path_out: str, palette_out: str,
          min_count: int = MIN_COUNT, off_tol: float = OFF_PALETTE_TOL) -> pd.DataFrame:
    report: dict[str, object] = {}
    df = pd.read_csv(path_in)
    report["rows_in"] = len(df)

    # 1) Parse every formula.
    parsed, parse_ok = [], []
    for s in df["alloy"].astype(str):
        try:
            parsed.append(parse_formula(s)); parse_ok.append(True)
        except Exception:
            parsed.append(None); parse_ok.append(False)
    df = df.assign(_frac=parsed, _ok=parse_ok)
    report["parse_failures_dropped"] = int((~df["_ok"]).sum())
    df = df[df["_ok"]].copy()

    # 2) Data-driven palette: elements appearing in >= min_count entries.
    freq: dict[str, int] = {}
    for frac in df["_frac"]:
        for el in frac:
            freq[el] = freq.get(el, 0) + 1
    palette = sorted([e for e, c in freq.items() if c >= min_count],
                     key=lambda e: (-freq[e], e))
    dropped_elems = sorted([e for e, c in freq.items() if c < min_count],
                           key=lambda e: (-freq[e], e))
    report["palette_size"] = len(palette)
    report["palette"] = palette
    report["off_palette_elements"] = {e: freq[e] for e in dropped_elems}

    missing_vec = [e for e in palette if e not in VEC_TABLE]
    if missing_vec:
        raise ValueError(f"VEC_TABLE missing palette elements: {missing_vec}")

    # 3) Enforce palette per row.
    keep_rows, frac_vectors = [], []
    dropped_off = 0
    for idx, frac in df["_frac"].items():
        off = sum(v for e, v in frac.items() if e not in palette)
        if off >= off_tol:
            dropped_off += 1
            continue
        frac_vectors.append(composition_to_vector(frac, palette))
        keep_rows.append(idx)
    report["rows_dropped_off_palette"] = dropped_off
    df = df.loc[keep_rows].copy()
    F = np.vstack(frac_vectors)

    # 4) Fraction columns, recomputed VEC (QC), family key.
    frac_cols = [f"frac_{e}" for e in palette]
    frac_df = pd.DataFrame(F, columns=frac_cols, index=df.index)

    vec_arr = np.array([VEC_TABLE[e] for e in palette])
    df["vec_recomputed"] = (F * vec_arr).sum(axis=1)
    df["n_elem_parsed"] = (F > 0).sum(axis=1)

    def family_key(row_vec: np.ndarray) -> str:
        return "-".join(sorted(el for el, x in zip(palette, row_vec) if x >= FAMILY_TOL))
    df["family"] = [family_key(v) for v in F]

    # 5) Assemble output.
    id_cols = ["alloy_id", "alloy", "num_of_elem", "n_elem_parsed", "family"]
    label_cols = ["phase", "phase_id", "phase_conflict"]
    out = pd.concat(
        [df[id_cols + DESCRIPTOR_COLS + ["vec_recomputed"] + label_cols].reset_index(drop=True),
         frac_df.reset_index(drop=True)],
        axis=1,
    )
    out.to_csv(path_out, index=False)

    with open(palette_out, "w") as fh:
        json.dump({"palette": palette, "frac_columns": frac_cols,
                   "min_count": min_count, "element_frequency": freq,
                   "off_palette_elements": report["off_palette_elements"]},
                  fh, indent=2)

    # ---- QC report -------------------------------------------------------- #
    corr = np.corrcoef(out["vec_recomputed"],
                       out["vec"].fillna(out["vec_recomputed"]))[0, 1]
    frac_sum_ok = np.allclose(F.sum(axis=1), 1.0, atol=1e-6)
    print("=" * 62)
    print("FEATURE ENGINEERING QC REPORT")
    print("=" * 62)
    for k in ["rows_in", "parse_failures_dropped", "rows_dropped_off_palette",
              "palette_size"]:
        print(f"  {k:<28}: {report[k]}")
    print(f"  palette ({len(palette)}): {', '.join(palette)}")
    print(f"  off-palette (dropped as elems): {report['off_palette_elements']}")
    print("-" * 62)
    print(f"  final rows                  : {len(out)}")
    print(f"  fraction rows all sum to 1  : {frac_sum_ok}")
    print(f"  VEC recomputed vs given corr: {corr:.4f}")
    print("-" * 62)
    print("  Phase balance (retained):")
    for name, n in out["phase"].value_counts().items():
        print(f"    {name:<10}: {n:>4}  ({100*n/len(out):4.1f}%)")
    print(f"  distinct families           : {out['family'].nunique()}")
    print(f"  columns                     : {out.shape[1]}  "
          f"({len(frac_cols)} frac + {len(DESCRIPTOR_COLS)} descriptors + labels)")
    print(f"  written                     : {path_out}")
    print(f"  palette written             : {palette_out}")
    print("=" * 62)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Feature-engineer the cleaned HEA dataset.")
    ap.add_argument("--input", required=True, help="cleaned CSV from clean.py")
    ap.add_argument("--output", required=True, help="model-ready features CSV")
    ap.add_argument("--palette-out", default="palette.json", help="palette JSON path")
    ap.add_argument("--min-count", type=int, default=MIN_COUNT,
                    help="min #entries for an element to enter the palette")
    ap.add_argument("--off-palette-tol", type=float, default=OFF_PALETTE_TOL,
                    help="drop a row if this fraction or more is off-palette")
    args = ap.parse_args()
    build(args.input, args.output, args.palette_out,
          min_count=args.min_count, off_tol=args.off_palette_tol)


if __name__ == "__main__":
    main()
