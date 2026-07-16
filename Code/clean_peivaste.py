#!/usr/bin/env python3
"""
clean_peivaste.py
-----------------
Clean the Peivaste dataset into the project's standard clean schema.

Composition is given as one column per element (fractions summing to ~1); this
script reconstructs a formula string so the row flows through features.py
identically to the other sources. The dataset's own descriptor columns use
different conventions (its 'Enthalpy' is not the mixing enthalpy), so all five
descriptors are RECOMPUTED with physics.py for a consistent merge.

Phase codes are '+'-separated (FCC, BCC, IM, AM, HCP and combinations). Mapping
to the 4-class {FCC, BCC, FCC+BCC, Im}:
    * contains AM (amorphous) or HCP        -> drop (out of scope)
    * contains IM (intermetallic)           -> Im
    * FCC and BCC (no IM)                    -> FCC+BCC
    * FCC only / BCC only                   -> FCC / BCC

Usage
    python clean_peivaste.py --input Peivaste_dataset11252_79.csv \
        --palette palette.json --output peivaste_clean.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import json

import features
import physics

PHASE_ORDER = ["FCC", "BCC", "FCC+BCC", "Im"]
PHASE_TO_ID = {p: i for i, p in enumerate(PHASE_ORDER)}
OFF_PALETTE_TOL = 0.05


def map_phase(raw) -> object:
    if raw is None or pd.isna(raw):
        return np.nan
    toks = str(raw).strip().upper().replace(" ", "").split("+")
    if "AM" in toks or "HCP" in toks:
        return np.nan
    has_im = "IM" in toks
    has_fcc = "FCC" in toks
    has_bcc = "BCC" in toks
    if has_im:
        return "Im"
    if has_fcc and has_bcc:
        return "FCC+BCC"
    if has_fcc:
        return "FCC"
    if has_bcc:
        return "BCC"
    return np.nan


def formula_from_row(row, elem_cols) -> tuple[str, dict]:
    fd = {}
    for e in elem_cols:
        v = row[e]
        if v and v > 0:
            fd[e] = v
    s = sum(fd.values())
    if s <= 0:
        return "", {}
    fd = {e: v / s for e, v in fd.items()}
    formula = "".join(f"{e}{v*100:.5g}" for e, v in
                      sorted(fd.items(), key=lambda x: -x[1]))
    return formula, fd


def clean(path_in, palette_json, path_out):
    palette = json.load(open(palette_json))["palette"]
    df = pd.read_csv(path_in)
    rep = {"rows_raw": len(df)}

    cols = list(df.columns)
    elem_cols = cols[cols.index("Li"):cols.index("Au") + 1]
    for e in elem_cols:
        df[e] = pd.to_numeric(df[e], errors="coerce").fillna(0.0)

    df["phase"] = df["Phase"].map(map_phase)
    rep["dropped_phase_out_of_scope"] = int(df["phase"].isna().sum())
    df = df[df["phase"].notna()].copy()

    out_rows, dropped_empty, dropped_off = [], 0, 0
    for _, r in df.iterrows():
        formula, fdict = formula_from_row(r, elem_cols)
        if not fdict:
            dropped_empty += 1
            continue
        off = sum(v for e, v in fdict.items() if e not in palette)
        if off >= OFF_PALETTE_TOL:
            dropped_off += 1
            continue
        vec = features.composition_to_vector(fdict, palette)
        d = physics.compute_descriptors(vec[None, :], palette).iloc[0]
        out_rows.append({
            "alloy": formula,
            "num_of_elem": int((vec > 0).sum()),
            "vec": d["vec"], "atom_size_diff": d["delta"], "elect_diff": d["dchi"],
            "dHmix": d["dHmix"], "dSmix": d["dSmix"],
            "phase": r["phase"], "phase_id": PHASE_TO_ID[r["phase"]],
            "source": "peivaste",
        })
    rep["dropped_empty"] = dropped_empty
    rep["dropped_off_palette"] = dropped_off

    out = pd.DataFrame(out_rows)
    key = ["alloy", "vec", "atom_size_diff", "elect_diff", "dHmix", "dSmix", "phase"]
    rep["exact_duplicates"] = int(out.duplicated(subset=key).sum())
    out = out.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    out["phase_conflict"] = out.groupby("alloy")["phase"].transform("nunique") > 1
    out.insert(0, "alloy_id", [f"peiv_{i:05d}" for i in range(len(out))])
    out = out[["alloy_id", "alloy", "num_of_elem", "vec", "atom_size_diff",
               "elect_diff", "dHmix", "dSmix", "phase", "phase_id",
               "phase_conflict", "source"]]
    out.to_csv(path_out, index=False)

    print("=" * 60)
    print("PEIVASTE CLEANING")
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
    ap.add_argument("--output", default="peivaste_clean.csv")
    a = ap.parse_args()
    clean(a.input, a.palette, a.output)


if __name__ == "__main__":
    main()
