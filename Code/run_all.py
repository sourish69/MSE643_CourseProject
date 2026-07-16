#!/usr/bin/env python3
"""
run_all.py  (multi-source edition)
----------------------------------
One command, whole project: from three raw datasets to final figures.

Full pipeline order (loop-free):
    0.  clean.py            original raw            -> HEA_clean.csv
    1.  features.py         HEA_clean.csv           -> palette.json          [bootstrap palette]
    2.  clean_vladimir.py   Vladimir raw + palette  -> vladimir_clean.csv
    3.  clean_peivaste.py   Peivaste raw + palette  -> peivaste_clean.csv
    4.  merge.py            3 clean files           -> merged_clean.csv       [dedup + conflict vote]
    5.  features.py         merged_clean.csv        -> HEA_features.csv + palette.json
    6.  physics.py          HEA_features.csv        -> HEA_features_physics.csv (oracle-ready)
    7.  oracle.py           physics features        -> oracle_out/            (XGBoost + CV + SHAP)
    8.  cvae.py             features + palette       -> cvae_out/             (tuned; go/no-go)
    9.  evaluate.py         CVAE output              -> eval_cvae/
    10. convex_hull.py + evaluate.py                -> ch_out/ , eval_ch/
    11. random_baseline.py + evaluate.py            -> rand_out/, eval_rand/
    12. showcase.py         eval_cvae               -> generated_shortlist.csv
    13. make_plots.py       everything              -> figures/

Steps 7 and 8 need xgboost and torch. Use --skip-oracle / --skip-cvae to run
everything else. Use --single-source to skip Vladimir/Peivaste/merge and train
on the original dataset only.

Usage
    python run_all.py
    python run_all.py --skip-oracle --skip-cvae
    python run_all.py --single-source
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys


def run(cmd, desc):
    print("\n" + "#" * 70)
    print(f"# {desc}")
    print("#   $ " + " ".join(str(c) for c in cmd))
    print("#" * 70)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"\n!! stage failed: {desc} (exit {r.returncode}). Nothing downstream ran.")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description="Full HEA pipeline, raw -> figures.")
    ap.add_argument("--raw-original", default="HEA_Phase_Dataset_v1d_raw.csv")
    ap.add_argument("--raw-vladimir", default="Vladimir_HEA_database.csv")
    ap.add_argument("--raw-peivaste", default="Peivaste_dataset11252_79.csv")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--beta-max", default="0.15")
    ap.add_argument("--free-bits", default="0.3")
    ap.add_argument("--recon", default="ce")
    ap.add_argument("--epochs", default="400")
    ap.add_argument("--single-source", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--skip-cvae", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-showcase", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    py, W = args.python, args.workdir
    p = lambda *xs: os.path.join(W, *xs)

    clean0 = p("HEA_clean.csv")
    palette = p("palette.json")
    feats = p("HEA_features.csv")
    phys = p("HEA_features_physics.csv")
    oracle_model = p("oracle_out", "oracle_xgb.json")
    gen_cvae = p("cvae_out", "generated_compositions.csv")

    run([py, "clean.py", "--input", args.raw_original, "--output", clean0],
        "0  Clean original raw dataset")

    if args.single_source:
        run([py, "features.py", "--input", clean0, "--output", feats, "--palette-out", palette],
            "1  Feature engineering (single-source)")
    else:
        run([py, "features.py", "--input", clean0, "--output", p("_bootstrap_features.csv"),
             "--palette-out", palette], "1  Bootstrap palette from original")
        run([py, "clean_vladimir.py", "--input", args.raw_vladimir,
             "--palette", palette, "--output", p("vladimir_clean.csv")],
            "2  Clean Vladimir database")
        run([py, "clean_peivaste.py", "--input", args.raw_peivaste,
             "--palette", palette, "--output", p("peivaste_clean.csv")],
            "3  Clean Peivaste dataset")
        run([py, "merge.py", "--inputs", clean0, p("vladimir_clean.csv"), p("peivaste_clean.csv"),
             "--palette", palette, "--output", p("merged_clean.csv")],
            "4  Merge + final deduplication")
        run([py, "features.py", "--input", p("merged_clean.csv"), "--output", feats,
             "--palette-out", palette], "5  Feature engineering (merged)")

    run([py, "physics.py", "--emit-features", feats, "--palette", palette, "--out", phys],
        "6  Emit oracle-ready physics descriptors")

    if not args.skip_oracle:
        run([py, "oracle.py", "--input", phys, "--outdir", p("oracle_out")],
            "7  Train + validate oracle (XGBoost)")
    else:
        print("\n-- skipping oracle (7) --")

    if not args.skip_cvae:
        run([py, "cvae.py", "--input", feats, "--palette", palette, "--outdir", p("cvae_out"),
             "--beta-max", args.beta_max, "--free-bits", args.free_bits,
             "--recon", args.recon, "--epochs", args.epochs],
            "8  Train CVAE + generate + diversity go/no-go")
    else:
        print("\n-- skipping CVAE (8) --")

    def eval_cmd(gen, outdir, desc):
        cmd = [py, "evaluate.py", "--generated", gen, "--train", feats,
               "--palette", palette, "--outdir", outdir]
        if not args.skip_oracle and os.path.exists(oracle_model):
            cmd += ["--oracle", oracle_model]
        run(cmd, desc)

    if not args.skip_cvae:
        eval_cmd(gen_cvae, p("eval_cvae"), "9  Evaluate CVAE")

    if not args.skip_baselines:
        run([py, "convex_hull.py", "--train", feats, "--palette", palette, "--outdir", p("ch_out")],
            "10 Convex-hull baseline (fallback)")
        eval_cmd(p("ch_out", "generated_convexhull.csv"), p("eval_ch"), "10 Evaluate convex-hull")
        run([py, "random_baseline.py", "--train", feats, "--palette", palette, "--outdir", p("rand_out")],
            "11 Random baseline (floor)")
        eval_cmd(p("rand_out", "generated_random.csv"), p("eval_rand"), "11 Evaluate random")

    if not args.skip_showcase:
        scored = p("eval_cvae", "generated_scored.csv")
        if not os.path.exists(scored):
            scored = p("eval_ch", "generated_scored.csv")
        if os.path.exists(scored):
            run([py, "showcase.py", "--scored", scored, "--palette", palette,
                 "--top", "6", "--out", p("generated_shortlist.csv")],
                "12 Showcase generated alloys")

    if not args.skip_plots:
        run([py, "make_plots.py", "--root", W, "--outdir", p("figures")],
            "13 Generate result figures")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE.")
    if not args.single_source:
        print("  data: merged_clean.csv  (3 sources, deduped)")
    print("  compare eval_rand/ < eval_cvae/ ?<= eval_ch/  (evaluation_report.json)")
    print("  shortlist: generated_shortlist.csv     figures: figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
