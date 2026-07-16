#!/usr/bin/env python3
"""
clean_hea_dataset.py
--------------------
Clean the raw "HEA Phase Dataset v1d" into an analysis-ready CSV suitable for
pandas, scikit-learn and PyTorch.

What it does
    1.  Skips the sheet-title row and ignores the malformed/mis-aligned header
        (columns are assigned by POSITION, which is consistent in the raw file).
    2.  Drops the processing-history columns and the DOI/reference column.
    3.  Removes fully-blank rows.
    4.  Strips whitespace from every string cell and removes internal spaces
        inside alloy formulae (e.g. "Al10 (HfNbTiZr)90" -> "Al10(HfNbTiZr)90").
    5.  Converts placeholder tokens ('', '-', 'na', ...) to NaN.
    6.  Coerces the numeric descriptor columns to float and num_of_elem to a
        nullable integer.
    7.  Normalises the phase labels to a consistent vocabulary
        {FCC, BCC, FCC+BCC, Im} and adds an integer-encoded label column.
    8.  Collapses exact duplicate rows (identical composition + descriptors +
        phase) that arise once processing columns are dropped.
    9.  Flags alloys whose composition appears with more than one phase label.
    10. Writes the cleaned CSV and prints a QC report.

Usage
    python clean_hea_dataset.py \
        --input  HEA_Phase_Dataset_v1d_raw.csv \
        --output HEA_Phase_Dataset_v1d_clean.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The raw file's first line is a sheet title; the second line is a malformed
# header. We therefore skip the title line and OVERRIDE the header by position.
TITLE_ROWS_TO_SKIP = 1

# Canonical names for all 17 raw columns, in physical order.
POSITIONAL_COLUMNS = [
    "alloy_id", "alloy", "num_of_elem", "vec", "atom_size_diff", "elect_diff",
    "dHmix", "dSmix",
    "synthesis_route", "hot_cold_working", "homog_temp", "homog_time",
    "anneal_temp", "anneal_time", "quench_proc",
    "phases", "references",
]

# Columns to discard: processing history + DOI references.
COLUMNS_TO_DROP = [
    "synthesis_route", "hot_cold_working", "homog_temp", "homog_time",
    "anneal_temp", "anneal_time", "quench_proc",
    "references",
]

# Numeric descriptor columns (float).
NUMERIC_COLS = ["vec", "atom_size_diff", "elect_diff", "dHmix", "dSmix"]

# Tokens that mean "missing".
MISSING_TOKENS = {"", "-", "--", "na", "n/a", "nan", "none", "null"}

# Deterministic integer encoding for the cleaned phase labels.
PHASE_ORDER = ["FCC", "BCC", "FCC+BCC", "Im"]
PHASE_TO_ID = {name: i for i, name in enumerate(PHASE_ORDER)}

# Columns that define a unique physical data point (used for de-duplication).
MODEL_KEY_COLS = ["alloy", "num_of_elem", "vec", "atom_size_diff",
                  "elect_diff", "dHmix", "dSmix", "phase"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_raw(path: str) -> pd.DataFrame:
    """Read the raw CSV, skip the title row, assign columns by position."""
    df = pd.read_csv(
        path,
        skiprows=TITLE_ROWS_TO_SKIP,   # drop the sheet-title line
        header=0,                      # the next line is the (malformed) header
        dtype=str,                     # read everything as text; coerce later
        keep_default_na=False,         # handle missing tokens ourselves
    )
    if df.shape[1] != len(POSITIONAL_COLUMNS):
        raise ValueError(
            f"Expected {len(POSITIONAL_COLUMNS)} columns, found {df.shape[1]}. "
            "The raw layout may have changed."
        )
    df.columns = POSITIONAL_COLUMNS    # override the broken header by position
    return df


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing whitespace from every cell."""
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip()
    return df


def to_nan(series: pd.Series) -> pd.Series:
    """Replace missing-value tokens (case-insensitive) with NaN."""
    lowered = series.astype("string").str.strip().str.lower()
    return series.where(~lowered.isin(MISSING_TOKENS), other=pd.NA)


def normalise_phase(raw) -> object:
    """
    Map any raw phase string to one of {FCC, BCC, FCC+BCC, Im}.

    Token-based so it is robust to every spelling variant:
    'FCC_SS'->FCC, 'BCC_SS'->BCC, and every dual-phase spelling
    ('FCC_PLUS_BCC', 'FCC+BCC_SS', 'BCC_FCC_', ...) -> 'FCC+BCC'.
    'Im' (and 'IM') -> 'Im'. Anything else (HCP, amorphous, blank) -> NaN.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or pd.isna(raw):
        return np.nan
    s = str(raw).strip().upper()
    if s.strip("_+- ") == "" or s.lower() in MISSING_TOKENS:
        return np.nan
    has_fcc = "FCC" in s
    has_bcc = "BCC" in s
    if has_fcc and has_bcc:
        return "FCC+BCC"
    if has_fcc:
        return "FCC"
    if has_bcc:
        return "BCC"
    if "IM" in s:
        return "Im"
    return np.nan  # HCP / amorphous / unknown -> out of scope


# --------------------------------------------------------------------------- #
# Main cleaning routine
# --------------------------------------------------------------------------- #

def clean(path_in: str, path_out: str, drop_duplicates: bool = True) -> pd.DataFrame:
    report: dict[str, object] = {}

    df = load_raw(path_in)
    report["rows_raw"] = len(df)

    # 1) Drop processing + reference columns.
    df = df.drop(columns=COLUMNS_TO_DROP)

    # 2) Strip whitespace everywhere.
    df = strip_strings(df)

    # 3) Remove fully-blank rows (no alloy AND no phase).
    blank = (df["alloy"].fillna("").str.strip() == "") & \
            (df["phases"].fillna("").str.strip() == "")
    report["rows_blank_dropped"] = int(blank.sum())
    df = df[~blank].copy()

    # 4) Clean alloy formulae: remove ALL internal whitespace.
    df["alloy"] = df["alloy"].str.replace(r"\s+", "", regex=True)

    # 5) Missing tokens -> NaN, then coerce numerics.
    for c in ["num_of_elem"] + NUMERIC_COLS:
        df[c] = to_nan(df[c])
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["num_of_elem"] = pd.to_numeric(df["num_of_elem"], errors="coerce").astype("Int64")

    # 6) Normalise phase labels + integer encoding.
    df["phase"] = df["phases"].map(normalise_phase)
    df = df.drop(columns=["phases"])
    report["rows_unmapped_phase_dropped"] = int(df["phase"].isna().sum())
    df = df[df["phase"].notna()].copy()
    df["phase_id"] = df["phase"].map(PHASE_TO_ID).astype("Int64")

    # 7) De-duplicate exact repeats on the modelling key.
    report["rows_before_dedup"] = len(df)
    n_dupes = int(df.duplicated(subset=MODEL_KEY_COLS).sum())
    report["exact_duplicate_rows"] = n_dupes
    if drop_duplicates:
        df = df.drop_duplicates(subset=MODEL_KEY_COLS, keep="first").copy()
    report["rows_after_dedup"] = len(df)

    # 8) Flag conflicting labels (same alloy, >1 distinct phase).
    phases_per_alloy = df.groupby("alloy")["phase"].transform("nunique")
    df["phase_conflict"] = (phases_per_alloy > 1)
    report["alloys_with_conflicting_phase"] = int(
        df.loc[df["phase_conflict"], "alloy"].nunique()
    )
    report["rows_with_conflicting_phase"] = int(df["phase_conflict"].sum())

    # 9) Final tidy: column order, index, dtypes.
    final_cols = ["alloy_id", "alloy", "num_of_elem",
                  "vec", "atom_size_diff", "elect_diff", "dHmix", "dSmix",
                  "phase", "phase_id", "phase_conflict"]
    df = df[final_cols].reset_index(drop=True)

    df.to_csv(path_out, index=False)

    # ---- QC report -------------------------------------------------------- #
    print("=" * 60)
    print("CLEANING QC REPORT")
    print("=" * 60)
    for k, v in report.items():
        print(f"  {k:<34}: {v}")
    print("-" * 60)
    print("  Final phase balance:")
    counts = df["phase"].value_counts()
    for name in PHASE_ORDER:
        n = int(counts.get(name, 0))
        pct = 100 * n / len(df) if len(df) else 0
        print(f"    {name:<10}: {n:>5}  ({pct:4.1f}%)")
    print("-" * 60)
    print("  Missing values per kept column:")
    for c in final_cols:
        m = int(df[c].isna().sum())
        if m:
            print(f"    {c:<16}: {m}")
    print(f"  Final shape: {df.shape}")
    print(f"  Written to : {path_out}")
    print("=" * 60)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean the raw HEA phase dataset.")
    ap.add_argument("--input", required=True, help="path to raw CSV")
    ap.add_argument("--output", required=True, help="path for cleaned CSV")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="do NOT collapse exact duplicate rows")
    args = ap.parse_args()
    clean(args.input, args.output, drop_duplicates=not args.keep_duplicates)


if __name__ == "__main__":
    main()
