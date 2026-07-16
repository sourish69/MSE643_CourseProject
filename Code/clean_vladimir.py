#!/usr/bin/env python3
"""
clean_vladimir.py
-----------------
Clean the Vladimir HEA database into the project's standard clean schema so it
can be concatenated with the original cleaned dataset and fed to features.py.

Input columns of note: 'Alloy' (formula string), 'Phase' (free-text phase),
'Type of solution'. No physics descriptors are provided, so they are COMPUTED
here with physics.py (parse formula -> palette fractions -> descriptors), giving
output identical in schema to clean.py.

Phase mapping (4-class {FCC, BCC, FCC+BCC, Im}); rows are DROPPED when out of
scope:
    * contains HCP / amorphous / oxide / anatase / nanotubular / glass  -> drop
    * ambiguous (N/A, not specified, unknown, likely, bare 'cubic')     -> drop
    * any intermetallic marker (Intermetallic, Laves, sigma, C14/C15,
      martensite, mu phase)                                             -> Im
    * both FCC and BCC present (bare B2/L12 fold into BCC/FCC)          -> FCC+BCC
    * FCC only / BCC only                                              -> FCC / BCC

Usage
    python clean_vladimir.py --input Vladimir_HEA_database.csv \
        --palette palette.json --output vladimir_clean.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import json

import features   # parse_formula, composition_to_vector
import physics     # compute_descriptors

PHASE_ORDER = ["FCC", "BCC", "FCC+BCC", "Im"]
PHASE_TO_ID = {p: i for i, p in enumerate(PHASE_ORDER)}
OFF_PALETTE_TOL = 0.05

DROP_SUBSTR = ["HCP", "AMORPH", "ANATASE", "OXIDE", "NANOTUB", "GLASS"]
AMBIG = ["N/A", "NOT SPECIFIED", "UNKNOWN", "LIKELY", "NONE"]
IM_MARK = ["INTERMETALLIC", "LAVES", "SIGMA", "C14", "C15", "MARTENSITE", "MU PHASE"]


def map_phase(raw) -> object:
    if raw is None or pd.isna(raw):
        return np.nan
    t = str(raw).strip().upper()
    if t == "" or t in AMBIG or t == "CUBIC":
        return np.nan
    if any(s in t for s in DROP_SUBSTR):
        return np.nan
    if any(s in t for s in AMBIG):
        return np.nan
    has_im = any(s in t for s in IM_MARK)
    has_fcc = ("FCC" in t) or ("L12" in t)
    has_bcc = ("BCC" in t) or ("B2" in t)
    if has_im:
        return "Im"
    if has_fcc and has_bcc:
        return "FCC+BCC"
    if has_fcc:
        return "FCC"
    if has_bcc:
        return "BCC"
    return np.nan


def clean(path_in, palette_json, path_out):
    palette = json.load(open(palette_json))["palette"]
    df = pd.read_csv(path_in, dtype=str, keep_default_na=False)
    rep = {"rows_raw": len(df)}

    df["phase"] = df["Phase"].map(map_phase)
    rep["dropped_phase_out_of_scope"] = int(df["phase"].isna().sum())
    df = df[df["phase"].notna()].copy()

    out_rows, dropped_parse, dropped_off = [], 0, 0
    for _, r in df.iterrows():
        formula = str(r["Alloy"]).strip()
        try:
            fdict = features.parse_formula(formula)
        except Exception:
            dropped_parse += 1
            continue
        off = sum(v for e, v in fdict.items() if e not in palette)
        if off >= OFF_PALETTE_TOL:
            dropped_off += 1
            continue
        vec = features.composition_to_vector(fdict, palette)
        d = physics.compute_descriptors(vec[None, :], palette).iloc[0]
        out_rows.append({
            "alloy": formula.replace(" ", ""),
            "num_of_elem": int((vec > 0).sum()),
            "vec": d["vec"], "atom_size_diff": d["delta"], "elect_diff": d["dchi"],
            "dHmix": d["dHmix"], "dSmix": d["dSmix"],
            "phase": r["phase"], "phase_id": PHASE_TO_ID[r["phase"]],
            "source": "vladimir",
        })
    rep["dropped_parse_fail"] = dropped_parse
    rep["dropped_off_palette"] = dropped_off

    out = pd.DataFrame(out_rows)
    # de-duplicate on modelling key
    key = ["alloy", "vec", "atom_size_diff", "elect_diff", "dHmix", "dSmix", "phase"]
    rep["exact_duplicates"] = int(out.duplicated(subset=key).sum())
    out = out.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    # conflict flag
    out["phase_conflict"] = out.groupby("alloy")["phase"].transform("nunique") > 1
    out.insert(0, "alloy_id", [f"vlad_{i:05d}" for i in range(len(out))])
    out = out[["alloy_id", "alloy", "num_of_elem", "vec", "atom_size_diff",
               "elect_diff", "dHmix", "dSmix", "phase", "phase_id",
               "phase_conflict", "source"]]
    out.to_csv(path_out, index=False)

    print("=" * 60)
    print("VLADIMIR CLEANING")
    print("=" * 60)
    for k, v in rep.items():
        print(f"  {k:<28}: {v}")
    print("-" * 60)
    print(f"  final rows: {len(out)}")
    print("  phase balance:")
    for p in PHASE_ORDER:
        n = int((out["phase"] == p).sum())
        print(f"    {p:<8}: {n:>5} ({100*n/len(out):4.1f}%)")
    print(f"  conflicting alloys: {out.loc[out.phase_conflict,'alloy'].nunique()}")
    print(f"  written -> {path_out}")
    print("=" * 60)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--palette", default="palette.json")
    ap.add_argument("--output", default="vladimir_clean.csv")
    a = ap.parse_args()
    clean(a.input, a.palette, a.output)


if __name__ == "__main__":
    main()
