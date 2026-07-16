#!/usr/bin/env python3
"""
merge.py
--------
Merge several cleaned HEA datasets into ONE deduplicated training set.

Steps
    1. Concatenate the input clean CSVs (a `source` tag is added if absent).
    2. CANONICALISE every composition: parse the formula, project onto the
       palette, and rewrite it in one deterministic form (elements alphabetical,
       at.% to 1 dp). So CoCrFeNi, Ni25Co25Cr25Fe25 and Co25Cr25Fe25Ni25 all
       become the same string -- this is what makes cross-source dedup work.
    3. RECOMPUTE all five descriptors with physics.py, so the whole merged set
       uses one convention (removes mixed Machaka/physics descriptors).
    4. Collapse to ONE ROW PER UNIQUE COMPOSITION:
         - identical composition + identical phase  -> redundant, dropped;
         - identical composition, different phase    -> resolved by MAJORITY
           VOTE across all records (ties broken by source priority
           original > vladimir > peivaste), and flagged phase_conflict=True.
    5. Write the merged CSV (clean.py schema + provenance columns) ready for
       features.py.

Usage
    python merge.py --inputs HEA_clean.csv vladimir_clean.csv peivaste_clean.csv \
        --palette palette.json --output merged_clean.csv
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter
import numpy as np
import pandas as pd

import features
import physics

PHASE_ORDER = ["FCC", "BCC", "FCC+BCC", "Im"]
PHASE_TO_ID = {p: i for i, p in enumerate(PHASE_ORDER)}
SOURCE_PRIORITY = {"original": 0, "vladimir": 1, "peivaste": 2}
ROUND = 3   # fraction rounding for the dedup key (0.1 at.%)


def canonical(fdict, palette):
    """Return (canonical_formula, rounded_palette_vector) for a composition."""
    vec = features.composition_to_vector(fdict, palette)
    vr = np.round(vec, ROUND)
    s = vr.sum()
    if s > 0:
        vr = vr / s
    order = sorted(range(len(palette)), key=lambda i: palette[i])   # alphabetical
    formula = "".join(f"{palette[i]}{round(vr[i]*100,1):g}"
                      for i in order if vr[i] > 0.0005)
    return formula, tuple(np.round(vr, ROUND))


def load_inputs(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "source" not in df.columns:
            df["source"] = os.path.splitext(os.path.basename(p))[0].split("_")[0].lower()
        frames.append(df[["alloy", "phase", "source"]])
    return pd.concat(frames, ignore_index=True)


def resolve(group):
    """Majority vote on phase; tie-break by best (lowest) source priority."""
    phases = list(group["phase"])
    counts = Counter(phases)
    top = counts.most_common()
    best_n = top[0][1]
    tied = [p for p, n in top if n == best_n]
    if len(tied) == 1:
        winner = tied[0]
    else:  # tie -> the label carried by the most trusted source among tied rows
        sub = group[group["phase"].isin(tied)].copy()
        sub["prio"] = sub["source"].map(SOURCE_PRIORITY).fillna(9)
        winner = sub.sort_values("prio").iloc[0]["phase"]
    conflict = len(counts) > 1
    return winner, conflict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--palette", default="palette.json")
    ap.add_argument("--output", default="merged_clean.csv")
    args = ap.parse_args()

    palette = json.load(open(args.palette))["palette"]
    raw = load_inputs(args.inputs)
    rep = {"inputs": {os.path.basename(p): None for p in args.inputs},
           "concatenated": len(raw)}

    # canonicalise
    keys, forms, drop = [], [], 0
    for a in raw["alloy"].astype(str):
        try:
            fd = features.parse_formula(a)
            f, k = canonical(fd, palette)
            if not f:
                raise ValueError
            forms.append(f); keys.append(k)
        except Exception:
            forms.append(None); keys.append(None); drop += 1
    raw["canonical"] = forms
    raw["key"] = keys
    raw = raw[raw["canonical"].notna()].copy()
    rep["unparseable_dropped"] = drop

    # collapse to one row per unique composition
    out_rows = []
    n_conflict = 0
    for key, grp in raw.groupby("key"):
        winner, conflict = resolve(grp)
        n_conflict += int(conflict)
        vec = np.array(key, dtype=float)
        d = physics.compute_descriptors(vec[None, :], palette).iloc[0]
        out_rows.append({
            "alloy": grp["canonical"].iloc[0],
            "num_of_elem": int((vec > 0).sum()),
            "vec": d["vec"], "atom_size_diff": d["delta"], "elect_diff": d["dchi"],
            "dHmix": d["dHmix"], "dSmix": d["dSmix"],
            "phase": winner, "phase_id": PHASE_TO_ID[winner],
            "phase_conflict": conflict,
            "n_records": len(grp),
            "sources": ";".join(sorted(grp["source"].unique())),
        })

    out = pd.DataFrame(out_rows).sort_values("alloy").reset_index(drop=True)
    out.insert(0, "alloy_id", [f"m_{i:05d}" for i in range(len(out))])
    out.to_csv(args.output, index=False)

    print("=" * 62)
    print("MERGE + FINAL DEDUPLICATION")
    print("=" * 62)
    print(f"  concatenated records        : {rep['concatenated']}")
    print(f"  unparseable dropped         : {rep['unparseable_dropped']}")
    print(f"  unique compositions (final) : {len(out)}")
    print(f"  redundant records removed   : {rep['concatenated']-rep['unparseable_dropped']-len(out)}")
    print(f"  compositions w/ resolved conflict : {n_conflict}")
    print("-" * 62)
    print("  final phase balance:")
    for p in PHASE_ORDER:
        n = int((out["phase"] == p).sum())
        print(f"    {p:<8}: {n:>5} ({100*n/len(out):4.1f}%)")
    print("-" * 62)
    print("  composition coverage by #sources:")
    ns = out["sources"].str.count(";").add(1).value_counts().sort_index()
    for k, v in ns.items():
        print(f"    in {k} source(s): {v}")
    print(f"  written -> {args.output}")
    print("=" * 62)


if __name__ == "__main__":
    main()
