#!/usr/bin/env python3
"""
evaluate.py
-----------
Physics filter + evaluation for the CVAE-generated HEA compositions. Produces
the project's headline results.

Pipeline
    1. Recompute descriptors for every generated composition via physics.py
       (single source of truth, so no train/serve skew).
    2. PHYSICS FEASIBILITY FILTER (Yang-Zhang): flag solid-solution formers and
       report the feasibility rate per requested phase.
    3. MODEL-FREE VALIDATION (the headline, needs no model):
         - VEC of FCC- vs BCC-conditioned generations vs the ~6.87/8.0 boundary.
         - dHmix & delta of Im- vs solid-solution-conditioned generations
           (intermetallics should sit at more negative dHmix / higher delta).
    4. NOVELTY & DIVERSITY (guards against trivial noise-novelty): novelty rate,
       nearest-training-neighbour distance, mean pairwise diversity.
    5. CONDITIONAL CONSISTENCY (optional, needs the oracle trained on physics
       features): fraction of generations the oracle reads back as the requested
       phase.

Also writes HEA_features_physics.csv -- the TRAINING set re-featurised with
physics.py -- so the oracle can be retrained on identical descriptors (required
for a valid conditional-consistency score).

Usage
    python evaluate.py --generated cvae_out/generated_compositions.csv \
        --train HEA_features.csv --palette palette.json \
        --oracle oracle_out/oracle_phys.json --outdir eval_out
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import physics  # verified module in the same folder

PHASE_ORDER = ["FCC", "BCC", "FCC+BCC", "Im"]
ORACLE_FEATS = ["vec", "delta", "dchi", "dHmix", "dSmix"]   # physics.py names
VEC_LO, VEC_HI = 6.87, 8.0                                  # BCC / FCC boundary


# --------------------------------------------------------------------------- #
def nn_distances(gen: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Min L2 distance from each generated vector to the reference set."""
    out = np.empty(len(gen))
    step = 512
    for i in range(0, len(gen), step):
        chunk = gen[i:i + step]
        d = np.linalg.norm(chunk[:, None, :] - ref[None, :, :], axis=2)
        out[i:i + step] = d.min(1)
    return out


def pairwise_mean(A: np.ndarray, cap: int = 600) -> float:
    if len(A) > cap:
        A = A[np.random.RandomState(0).choice(len(A), cap, replace=False)]
    if len(A) < 2:
        return float("nan")
    d = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    iu = np.triu_indices(len(A), 1)
    return float(d[iu].mean())


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--palette", required=True)
    ap.add_argument("--oracle", default=None, help="xgboost json trained on physics feats")
    ap.add_argument("--outdir", default="eval_out")
    ap.add_argument("--novelty-eps", type=float, default=0.02, help="L2 novelty threshold")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pal = json.load(open(args.palette))
    palette, fc = pal["palette"], pal["frac_columns"]

    # ---- training reference + re-featurised training set for the oracle ---- #
    train = pd.read_csv(args.train)
    Ftr = train[fc].to_numpy()
    dtr = physics.compute_descriptors(Ftr, palette)
    train_phys = pd.concat(
        [train[["alloy_id", "alloy", "family", "phase", "phase_id"]].reset_index(drop=True),
         dtr[ORACLE_FEATS].rename(columns={"delta": "atom_size_diff",
                                           "dchi": "elect_diff"}).reset_index(drop=True)],
        axis=1)
    train_phys.to_csv(os.path.join(args.outdir, "HEA_features_physics.csv"), index=False)

    # ---- generated compositions ---- #
    gen = pd.read_csv(args.generated)
    Fg = gen[fc].to_numpy()
    dg = physics.compute_descriptors(Fg, palette)
    gen = pd.concat([gen[["requested_phase"]].reset_index(drop=True),
                     dg.reset_index(drop=True)], axis=1)
    for c in fc:
        gen[c] = Fg[:, fc.index(c)]

    report: dict = {"n_generated": int(len(gen))}
    print("=" * 64)
    print(f"EVALUATION  (n_generated={len(gen)}, palette={len(palette)})")
    print("=" * 64)

    # ---- 2) feasibility filter ---- #
    print("\n[Physics feasibility filter — Yang-Zhang solid-solution criteria]")
    feas = {}
    for ph in PHASE_ORDER:
        sub = gen[gen.requested_phase == ph]
        if len(sub) == 0:
            continue
        known = sub["dHmix_known"].mean()
        rate = sub["feasible"].mean()
        feas[ph] = {"n": int(len(sub)), "feasible_rate": float(rate),
                    "dHmix_coverage": float(known)}
        print(f"  {ph:<8}: feasible={rate:5.1%}   (dHmix covered {known:5.1%} of {len(sub)})")
    report["feasibility"] = feas
    print("  note: FCC/BCC/FCC+BCC should be mostly feasible; Im is expected to")
    print("        largely FAIL the solid-solution filter (it is intermetallic).")

    # ---- 3) model-free validation ---- #
    print("\n[Model-free validation]")
    means = gen.groupby("requested_phase")[["vec", "dHmix", "delta"]].mean()
    means = means.reindex([p for p in PHASE_ORDER if p in means.index])
    print("  class-conditional means of generated compositions:")
    print(means.round(2).to_string())
    fcc_v = gen.loc[gen.requested_phase == "FCC", "vec"]
    bcc_v = gen.loc[gen.requested_phase == "BCC", "vec"]
    vec_gap = float(fcc_v.mean() - bcc_v.mean()) if len(fcc_v) and len(bcc_v) else float("nan")
    report["vec_validation"] = {"FCC_mean_vec": float(fcc_v.mean()) if len(fcc_v) else None,
                                "BCC_mean_vec": float(bcc_v.mean()) if len(bcc_v) else None,
                                "gap": vec_gap}
    print(f"  VEC rule: FCC mean={fcc_v.mean():.2f}  BCC mean={bcc_v.mean():.2f}  "
          f"gap={vec_gap:.2f}  (expect FCC high, BCC low, split ~6.87-8.0)")

    # ---- 4) novelty & diversity ---- #
    print("\n[Novelty & diversity]")
    nd = {}
    for ph in PHASE_ORDER:
        idx = gen.requested_phase == ph
        if idx.sum() == 0:
            continue
        g = Fg[idx.to_numpy()]
        ref = Ftr[train.phase.to_numpy() == ph]
        nn = nn_distances(g, ref) if len(ref) else np.full(len(g), np.nan)
        novelty = float((nn > args.novelty_eps).mean())
        nd[ph] = {"novelty_rate": novelty, "mean_nn_dist": float(np.nanmean(nn)),
                  "diversity": pairwise_mean(g)}
        print(f"  {ph:<8}: novelty={novelty:5.1%}  mean_NN_to_train={np.nanmean(nn):.3f}"
              f"  diversity={pairwise_mean(g):.3f}")
    report["novelty_diversity"] = nd

    # ---- 5) conditional consistency (oracle) ---- #
    if args.oracle and os.path.exists(args.oracle):
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(args.oracle)
            Xg = gen[ORACLE_FEATS].to_numpy()
            pred = model.predict(Xg)
            req = gen.requested_phase.map({p: i for i, p in enumerate(PHASE_ORDER)}).to_numpy()
            gen["oracle_pred"] = [PHASE_ORDER[i] for i in pred]
            overall = float((pred == req).mean())
            print("\n[Conditional consistency — oracle re-reads generated phase]")
            print(f"  overall requested==predicted: {overall:5.1%}")
            per = {}
            for i, ph in enumerate(PHASE_ORDER):
                m = req == i
                if m.sum():
                    per[ph] = float((pred[m] == i).mean())
                    print(f"    {ph:<8}: {per[ph]:5.1%}")
            report["conditional_consistency"] = {"overall": overall, "per_class": per}
        except Exception as e:
            print(f"\n[Conditional consistency skipped: {e}]")
            print("  (train the oracle on eval_out/HEA_features_physics.csv, then pass --oracle)")
    else:
        print("\n[Conditional consistency skipped: no --oracle provided]")
        print("  Train oracle on eval_out/HEA_features_physics.csv for a consistent score.")

    # ---- figures ---- #
    _figures(gen, train, dtr, args.outdir)

    gen.to_csv(os.path.join(args.outdir, "generated_scored.csv"), index=False)
    with open(os.path.join(args.outdir, "evaluation_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 64)
    print(f"artifacts -> {args.outdir}/ : generated_scored.csv  evaluation_report.json")
    print("  vec_validation.png  im_validation.png  feasibility.png  novelty.png")
    print("  HEA_features_physics.csv  (retrain oracle on this for consistency)")
    print("=" * 64)


def _figures(gen, train, dtr, outdir):
    colors = {"FCC": "#2E7D32", "BCC": "#1565C0", "FCC+BCC": "#F9A825", "Im": "#B42318"}

    # Fig 1 -- VEC validation (FCC vs BCC)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for ph in ["BCC", "FCC"]:
        v = gen.loc[gen.requested_phase == ph, "vec"].dropna()
        if len(v):
            ax.hist(v, bins=30, alpha=0.6, label=f"generated {ph}", color=colors[ph], density=True)
    ax.axvspan(VEC_LO, VEC_HI, color="grey", alpha=0.15, label="transition (6.87–8.0)")
    ax.axvline(VEC_LO, color="grey", ls="--", lw=1); ax.axvline(VEC_HI, color="grey", ls="--", lw=1)
    ax.set_xlabel("VEC (computed from generated composition)"); ax.set_ylabel("density")
    ax.set_title("Model-free check: generated FCC vs BCC obey the VEC rule")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "vec_validation.png"), dpi=140); plt.close(fig)

    # Fig 2 -- Im vs solid solution on dHmix / delta
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for ph in PHASE_ORDER:
        sub = gen[gen.requested_phase == ph]
        ax.scatter(sub["dHmix"], sub["delta"], s=10, alpha=0.5,
                   color=colors[ph], label=f"gen {ph}")
    ax.axhline(physics.DELTA_MAX, color="k", ls=":", lw=1, label="δ=6.6% SS limit")
    ax.set_xlabel("ΔH_mix (kJ/mol)"); ax.set_ylabel("δ atomic-size mismatch (%)")
    ax.set_title("Model-free check: Im sits at more negative ΔH_mix / higher δ")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "im_validation.png"), dpi=140); plt.close(fig)

    # Fig 3 -- feasibility by phase
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    phs = [p for p in PHASE_ORDER if (gen.requested_phase == p).any()]
    rates = [gen.loc[gen.requested_phase == p, "feasible"].mean() for p in phs]
    ax.bar(phs, rates, color=[colors[p] for p in phs])
    ax.set_ylim(0, 1); ax.set_ylabel("solid-solution feasibility rate")
    ax.set_title("Physics-filter pass rate by requested phase")
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f"{r:.0%}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "feasibility.png"), dpi=140); plt.close(fig)

    # Fig 4 -- novelty (nn distance hist)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    Ftr = train[[c for c in train.columns if c.startswith("frac_")]].to_numpy() \
        if any(c.startswith("frac_") for c in train.columns) else None
    if Ftr is not None:
        for ph in PHASE_ORDER:
            idx = (gen.requested_phase == ph).to_numpy()
            if idx.sum() == 0:
                continue
            g = gen.loc[idx, [c for c in gen.columns if c.startswith("frac_")]].to_numpy()
            ref = Ftr[train.phase.to_numpy() == ph]
            if len(ref) == 0:
                continue
            nn = nn_distances(g, ref)
            ax.hist(nn, bins=30, alpha=0.5, color=colors[ph], label=ph, density=True)
        ax.set_xlabel("nearest-training-neighbour distance (L2)")
        ax.set_ylabel("density"); ax.set_title("Novelty: distance of generations to training set")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "novelty.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
