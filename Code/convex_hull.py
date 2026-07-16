#!/usr/bin/env python3
"""
convex_hull.py
--------------
The CONVEX-HULL FALLBACK SAMPLER (and baseline).

Generates HEA compositions WITHOUT any neural network, by blending real
training alloys of the requested phase. Every output is a random weighted
average (a Dirichlet-weighted mixture) of a few training compositions of that
phase, so it is:
    * valid       -- fractions are non-negative and sum to 1 by construction;
    * phase-safe  -- a blend of BCC alloys stays in the BCC region;
    * novel       -- not an exact training entry;
    * inside the convex hull of the training set -- it interpolates between
      known alloys, but (unlike the CVAE) it cannot extrapolate beyond them.

Two uses, both important:
    1. FALLBACK: if the CVAE's Day-3 diversity go/no-go flags COLLAPSE-RISK for
       a class (or all classes), generate that class with this instead. The
       project still ships a complete, honest result.
    2. BASELINE: run this alongside a working CVAE. If the CVAE beats this on
       diversity / coverage / reaching under-sampled regions, you have shown the
       CVAE adds value. Without this comparison, that claim is unsupported.

Output schema is IDENTICAL to cvae.py's generated_compositions.csv
(requested_phase, vec_recomputed, frac_*), so evaluate.py consumes it directly.

Usage
    python convex_hull.py --train HEA_features.csv --palette palette.json \
        --outdir ch_out --n-per-phase 1000
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd

try:
    import physics                      # for VEC (same source of truth)
    _VEC = physics.VEC
except Exception:                       # standalone fallback
    _VEC = {"Al": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9,
            "Ni": 10, "Cu": 11, "Zn": 12, "Zr": 4, "Nb": 5, "Mo": 6, "Hf": 4,
            "Ta": 5, "W": 6, "Si": 4, "C": 4, "Mg": 2}

PHASE_ORDER = ["FCC", "BCC", "FCC+BCC", "Im"]


def sparsify(F: np.ndarray, thresh: float) -> np.ndarray:
    F = np.where(F < thresh, 0.0, F)
    s = F.sum(1, keepdims=True)
    return np.divide(F, s, out=np.zeros_like(F), where=s > 0)


def sample_phase(parents: np.ndarray, n: int, k_min: int, k_max: int,
                 alpha: float, rng: np.random.RandomState) -> np.ndarray:
    """n compositions, each a Dirichlet-weighted blend of k random parents."""
    out = np.zeros((n, parents.shape[1]))
    n_parents = len(parents)
    for i in range(n):
        k = min(rng.randint(k_min, k_max + 1), n_parents)
        idx = rng.choice(n_parents, size=k, replace=False)
        w = rng.dirichlet(np.full(k, alpha))
        out[i] = w @ parents[idx]
    return out


def pairwise_mean(A: np.ndarray, cap: int = 600) -> float:
    if len(A) > cap:
        A = A[np.random.RandomState(0).choice(len(A), cap, replace=False)]
    if len(A) < 2:
        return float("nan")
    d = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    return float(d[np.triu_indices(len(A), 1)].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Convex-hull fallback / baseline sampler.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--palette", required=True)
    ap.add_argument("--outdir", default="ch_out")
    ap.add_argument("--n-per-phase", type=int, default=1000)
    ap.add_argument("--k-min", type=int, default=2, help="min #parents per blend")
    ap.add_argument("--k-max", type=int, default=3, help="max #parents per blend")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Dirichlet concentration for blend weights (1=uniform)")
    ap.add_argument("--sparsify", type=float, default=0.01)
    ap.add_argument("--phases", nargs="*", default=PHASE_ORDER,
                    help="which phases to generate (default: all four)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    pal = json.load(open(args.palette))
    palette, fc = pal["palette"], pal["frac_columns"]
    train = pd.read_csv(args.train)
    Ftr = train[fc].to_numpy()
    vec_vec = np.array([_VEC[e] for e in palette], dtype="float64")

    print("=" * 62)
    print(f"CONVEX-HULL SAMPLER  (n_per_phase={args.n_per_phase}, "
          f"k={args.k_min}-{args.k_max}, alpha={args.alpha})")
    print("=" * 62)

    rows = []
    for ph in args.phases:
        parents = Ftr[train["phase"].to_numpy() == ph]
        if len(parents) < 2:
            print(f"  {ph:<8}: only {len(parents)} training alloys -- skipped")
            continue
        gen = sample_phase(parents, args.n_per_phase, args.k_min, args.k_max,
                           args.alpha, rng)
        gen = sparsify(gen, args.sparsify)
        vec_gen = gen @ vec_vec
        div = pairwise_mean(gen)
        print(f"  {ph:<8}: parents={len(parents):3d}  generated={len(gen)}  "
              f"mean_VEC={vec_gen.mean():5.2f}  diversity={div:.3f}")
        for i in range(len(gen)):
            row = {"requested_phase": ph, "vec_recomputed": float(vec_gen[i])}
            row.update({f"frac_{e}": float(gen[i, j]) for j, e in enumerate(palette)})
            rows.append(row)

    out = pd.DataFrame(rows)
    path = os.path.join(args.outdir, "generated_convexhull.csv")
    out.to_csv(path, index=False)
    print("-" * 62)
    print(f"  wrote {len(out)} compositions -> {path}")
    print("  Feed this to evaluate.py exactly like the CVAE output:")
    print(f"    python evaluate.py --generated {path} \\")
    print("        --train HEA_features.csv --palette palette.json --outdir ch_eval")
    print("=" * 62)


if __name__ == "__main__":
    main()
