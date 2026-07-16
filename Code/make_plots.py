#!/usr/bin/env python3
"""
make_plots.py
-------------
Generate a full set of publication-quality figures for the HEA project results
section, from the artifacts already produced by the pipeline.

Reads (relative to --root, all optional; missing inputs are skipped gracefully):
    HEA_features.csv, palette.json
    oracle_out/oracle_metrics.json, oracle_out/oof_predictions.csv
    cvae_out/diversity.json, cvae_out/generated_compositions.csv
    eval_cvae/{evaluation_report.json, generated_scored.csv}
    eval_ch/{evaluation_report.json, generated_scored.csv}
    eval_rand/{evaluation_report.json, generated_scored.csv}

Writes PNGs (300 dpi) into --outdir (default: figures/).

Usage
    python make_plots.py --root . --outdir figures
"""

from __future__ import annotations
import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ---------------- style ----------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.fontsize": 9, "legend.frameon": False,
})

PHASES = ["FCC", "BCC", "FCC+BCC", "Im"]
PC = {"FCC": "#2E7D32", "BCC": "#1565C0", "FCC+BCC": "#F9A825", "Im": "#C62828"}
GEN_DIRS = {"Random": "eval_rand", "CVAE": "eval_cvae", "Convex-hull": "eval_ch"}
GC = {"Random": "#9E9E9E", "CVAE": "#6A1B9A", "Convex-hull": "#EF6C00"}
VEC_LO, VEC_HI = 6.87, 8.0

OUT = "figures"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def jload(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def cload(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def desc_col(df, name):
    """Return the descriptor column under either dataset or physics naming."""
    alias = {"delta": ["delta", "atom_size_diff"], "dchi": ["dchi", "elect_diff"]}
    for c in alias.get(name, [name]):
        if c in df.columns:
            return df[c]
    return None


# =================================================================== #
# 1. Dataset / EDA
# =================================================================== #
def plot_phase_balance(feat):
    if feat is None:
        return
    counts = feat["phase"].value_counts().reindex(PHASES)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(PHASES, counts.values, color=[PC[p] for p in PHASES])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, f"{int(v)}\n({v/counts.sum()*100:.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("number of alloys"); ax.set_title("Phase class balance (n=565)")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, "fig01_phase_balance.png")


def plot_element_frequency(feat, palette):
    if feat is None or palette is None:
        return
    fc = palette["frac_columns"]
    freq = (feat[fc] > 0).sum().sort_values(ascending=True)
    els = [c.replace("frac_", "") for c in freq.index]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.barh(els, freq.values, color="#37474F")
    for i, v in enumerate(freq.values):
        ax.text(v + 2, i, str(int(v)), va="center", fontsize=8)
    ax.set_xlabel("number of alloys containing element")
    ax.set_title("Element frequency across the palette")
    save(fig, "fig02_element_frequency.png")


def plot_descriptor_violins(feat):
    if feat is None:
        return
    specs = [("vec", "VEC", [VEC_LO, VEC_HI]), ("delta", r"$\delta$ (%)", [6.6]),
             ("dHmix", r"$\Delta H_{mix}$ (kJ/mol)", None), ("dSmix", r"$\Delta S_{mix}$ (J/mol K)", None)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    for ax, (key, lab, lines) in zip(axes.ravel(), specs):
        col = desc_col(feat, key)
        if col is None:
            continue
        data = [col[feat["phase"] == p].dropna().values for p in PHASES]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for pc, p in zip(parts["bodies"], PHASES):
            pc.set_facecolor(PC[p]); pc.set_alpha(0.65)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels(PHASES, fontsize=9)
        ax.set_ylabel(lab)
        if lines:
            for y in lines:
                ax.axhline(y, ls="--", color="k", lw=0.9, alpha=0.6)
    fig.suptitle("Descriptor distributions by phase (physics separation)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig03_descriptor_violins.png")


def plot_descriptor_corr(feat):
    if feat is None:
        return
    cols = {"VEC": "vec", r"$\delta$": "delta", r"$\Delta\chi$": "dchi",
            r"$\Delta H$": "dHmix", r"$\Delta S$": "dSmix"}
    mat = pd.DataFrame({k: desc_col(feat, v) for k, v in cols.items()})
    C = mat.corr().values
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols.keys())
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols.keys())
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(C[i, j]) > 0.5 else "black", fontsize=9)
    ax.set_title("Descriptor correlation"); ax.grid(False)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    save(fig, "fig04_descriptor_correlation.png")


def plot_vec_dhmix(feat):
    if feat is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for p in PHASES:
        s = feat[feat["phase"] == p]
        ax.scatter(desc_col(s, "vec"), desc_col(s, "dHmix"), s=14, alpha=0.6,
                   color=PC[p], label=p, edgecolors="none")
    ax.axvspan(VEC_LO, VEC_HI, color="grey", alpha=0.12)
    ax.set_xlabel("VEC"); ax.set_ylabel(r"$\Delta H_{mix}$ (kJ/mol)")
    ax.set_title("Physics landscape of the dataset"); ax.legend()
    save(fig, "fig05_vec_dhmix_scatter.png")


def plot_pca(feat, palette):
    if feat is None or palette is None:
        return
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return
    fc = palette["frac_columns"]
    X = feat[fc].values
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for p in PHASES:
        m = (feat["phase"] == p).values
        ax.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.6, color=PC[p], label=p, edgecolors="none")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("Composition space (PCA of fraction vectors)"); ax.legend()
    save(fig, "fig06_pca_composition.png")


def plot_nelem(feat):
    if feat is None or "n_elem_parsed" not in feat.columns:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    vc = feat["n_elem_parsed"].value_counts().sort_index()
    ax.bar(vc.index, vc.values, color="#455A64")
    ax.set_xlabel("number of elements in alloy"); ax.set_ylabel("count")
    ax.set_title("Alloy complexity distribution")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, "fig07_nelem_hist.png")


def plot_vec_threshold(feat):
    if feat is None:
        return
    sub = feat[feat["phase"].isin(["FCC", "BCC"])]
    v = desc_col(sub, "vec").values
    y = (sub["phase"] == "FCC").astype(int).values
    best, thr = 0, None
    for t in np.linspace(v.min(), v.max(), 300):
        acc = max(((v >= t) == y).mean(), ((v < t) == y).mean())
        if acc > best:
            best, thr = acc, t
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(v[y == 1], bins=25, alpha=0.6, color=PC["FCC"], label="FCC", density=True)
    ax.hist(v[y == 0], bins=25, alpha=0.6, color=PC["BCC"], label="BCC", density=True)
    ax.axvline(thr, color="k", ls="--", lw=1.2, label=f"threshold {thr:.2f}")
    ax.set_xlabel("VEC"); ax.set_ylabel("density")
    ax.set_title(f"VEC rule separates FCC/BCC ({best*100:.1f}% accuracy)")
    ax.legend()
    save(fig, "fig08_vec_threshold.png")


# =================================================================== #
# 2. Oracle
# =================================================================== #
def plot_confusion(metrics):
    if metrics is None or "confusion_matrix" not in metrics:
        return
    cm = np.array(metrics["confusion_matrix"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(PHASES, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(PHASES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i,j]:.2f}\n({int(cm[i,j])})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Oracle confusion matrix (row-normalised)"); ax.grid(False)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    save(fig, "fig09_confusion_matrix.png")


def plot_per_class_f1(metrics):
    if metrics is None or "per_class_f1" not in metrics:
        return
    f1 = [metrics["per_class_f1"].get(p, 0) for p in PHASES]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.bar(PHASES, f1, color=[PC[p] for p in PHASES])
    macro = metrics.get("f1_macro_pooled")
    if macro:
        ax.axhline(macro, color="k", ls="--", lw=1, label=f"macro-F1 {macro:.2f}")
        ax.legend()
    for i, v in enumerate(f1):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("F1 score")
    ax.set_title("Oracle per-class F1 (family-held-out)")
    save(fig, "fig10_per_class_f1.png")


def plot_fold_f1(metrics):
    if metrics is None or "fold_f1" not in metrics:
        return
    folds = metrics["fold_f1"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(range(1, len(folds) + 1), folds, "o-", color="#6A1B9A")
    m = np.mean(folds)
    ax.axhline(m, color="grey", ls="--", lw=1, label=f"mean {m:.2f}")
    ax.set_xlabel("CV fold"); ax.set_ylabel("F1-macro"); ax.set_ylim(0, 1)
    ax.set_title("Cross-validation stability"); ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, "fig11_cv_fold_f1.png")


# =================================================================== #
# 3. Generative — three-way comparison
# =================================================================== #
def load_reports(root):
    out = {}
    for name, d in GEN_DIRS.items():
        r = jload(os.path.join(root, d, "evaluation_report.json"))
        if r:
            out[name] = r
    return out


def grouped_bar(ax, per_phase_vals, ylabel, title, ylim=None):
    x = np.arange(len(PHASES)); w = 0.26
    for i, (gen, vals) in enumerate(per_phase_vals.items()):
        ax.bar(x + (i - 1) * w, [vals.get(p, np.nan) for p in PHASES], w,
               label=gen, color=GC.get(gen, None))
    ax.set_xticks(x); ax.set_xticklabels(PHASES)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()


def plot_feasibility_cmp(reports):
    if not reports:
        return
    vals = {g: {p: reports[g]["feasibility"].get(p, {}).get("feasible_rate", np.nan)
                for p in PHASES} for g in reports}
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    grouped_bar(ax, vals, "solid-solution feasibility", "Feasibility by phase and generator", (0, 1))
    save(fig, "fig12_feasibility_comparison.png")


def plot_diversity_cmp(reports):
    if not reports:
        return
    vals = {g: {p: reports[g]["novelty_diversity"].get(p, {}).get("diversity", np.nan)
                for p in PHASES} for g in reports}
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    grouped_bar(ax, vals, "mean pairwise diversity", "Generated diversity by phase and generator")
    save(fig, "fig13_diversity_comparison.png")


def plot_consistency_cmp(reports):
    have = {g: r for g, r in reports.items() if "conditional_consistency" in r}
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(len(PHASES) + 1); w = 0.26
    labels = PHASES + ["Overall"]
    for i, (g, r) in enumerate(have.items()):
        per = r["conditional_consistency"]["per_class"]
        vals = [per.get(p, np.nan) for p in PHASES] + [r["conditional_consistency"]["overall"]]
        ax.bar(x + (i - 1) * w, vals, w, label=g, color=GC.get(g, None))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("requested == oracle-predicted")
    ax.set_title("Conditional consistency (oracle re-reads phase)"); ax.legend()
    save(fig, "fig14_consistency_comparison.png")


def plot_novelty_vs_feasibility(reports):
    if not reports:
        return
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for g in reports:
        for p in PHASES:
            nov = reports[g]["novelty_diversity"].get(p, {}).get("novelty_rate", np.nan)
            fea = reports[g]["feasibility"].get(p, {}).get("feasible_rate", np.nan)
            ax.scatter(nov, fea, s=70, color=GC.get(g), edgecolors="k", linewidths=0.4)
        ax.scatter([], [], s=70, color=GC.get(g), edgecolors="k", label=g)
    ax.set_xlabel("novelty rate"); ax.set_ylabel("feasibility rate")
    ax.set_title("Novelty means little without feasibility"); ax.legend()
    ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.02, 1.05)
    ax.text(0.97, 0.06, "random:\nnovel but\ninfeasible", ha="right", fontsize=8, color="#616161")
    save(fig, "fig15_novelty_vs_feasibility.png")


def plot_vec_validation(root):
    g = cload(os.path.join(root, "eval_cvae", "generated_scored.csv"))
    if g is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for p in ["BCC", "FCC"]:
        v = desc_col(g[g.requested_phase == p], "vec")
        if v is not None and len(v):
            ax.hist(v.dropna(), bins=30, alpha=0.6, color=PC[p], density=True, label=f"generated {p}")
    ax.axvspan(VEC_LO, VEC_HI, color="grey", alpha=0.15, label="transition")
    ax.set_xlabel("VEC of generated composition"); ax.set_ylabel("density")
    ax.set_title("CVAE: generated FCC vs BCC obey the VEC rule"); ax.legend()
    save(fig, "fig16_vec_validation_cvae.png")


def plot_im_dhmix_delta(root):
    g = cload(os.path.join(root, "eval_cvae", "generated_scored.csv"))
    if g is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for p in PHASES:
        s = g[g.requested_phase == p]
        ax.scatter(desc_col(s, "dHmix"), desc_col(s, "delta"), s=10, alpha=0.45,
                   color=PC[p], label=p, edgecolors="none")
    ax.axhline(6.6, color="k", ls=":", lw=1, label=r"$\delta$=6.6% SS limit")
    ax.set_xlabel(r"$\Delta H_{mix}$ (kJ/mol)"); ax.set_ylabel(r"$\delta$ (%)")
    ax.set_title("CVAE: Im sits at more negative $\\Delta H_{mix}$ / higher $\\delta$"); ax.legend()
    save(fig, "fig17_im_dhmix_delta.png")


def plot_gonogo(root):
    d = jload(os.path.join(root, "cvae_out", "diversity.json"))
    if d is None or "per_class" not in d:
        return
    rows = {r["label"]: r["ratio"] for r in d["per_class"]}
    ratios = [rows.get(p, np.nan) for p in PHASES]
    colors = [PC[p] for p in PHASES]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(PHASES, ratios, color=colors)
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="go/no-go threshold 0.50")
    for i, v in enumerate(ratios):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("diversity ratio (gen / train)")
    ax.set_title("CVAE diversity go/no-go by class"); ax.legend()
    ax.set_ylim(0, max(0.6, np.nanmax(ratios) * 1.15))
    save(fig, "fig18_gonogo_ratios.png")


def plot_gen_vs_train_vec(root, feat):
    g = cload(os.path.join(root, "eval_cvae", "generated_scored.csv"))
    if g is None or feat is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, p in zip(axes, ["FCC", "BCC"]):
        tv = desc_col(feat[feat.phase == p], "vec").dropna()
        gv = desc_col(g[g.requested_phase == p], "vec").dropna()
        ax.hist(tv, bins=20, alpha=0.55, color="#607D8B", density=True, label="training")
        ax.hist(gv, bins=20, alpha=0.55, color=PC[p], density=True, label=f"generated {p}")
        ax.set_title(f"{p}"); ax.set_xlabel("VEC"); ax.legend()
    axes[0].set_ylabel("density")
    fig.suptitle("Generated compositions match the training VEC region", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig19_gen_vs_train_vec.png")


# =================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Generate result figures.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    global OUT
    OUT = args.outdir
    os.makedirs(OUT, exist_ok=True)
    R = args.root

    feat = cload(os.path.join(R, "HEA_features.csv"))
    palette = jload(os.path.join(R, "palette.json"))
    metrics = jload(os.path.join(R, "oracle_out", "oracle_metrics.json"))
    reports = load_reports(R)

    print("Dataset / EDA:")
    plot_phase_balance(feat)
    plot_element_frequency(feat, palette)
    plot_descriptor_violins(feat)
    plot_descriptor_corr(feat)
    plot_vec_dhmix(feat)
    plot_pca(feat, palette)
    plot_nelem(feat)
    plot_vec_threshold(feat)

    print("Oracle:")
    plot_confusion(metrics)
    plot_per_class_f1(metrics)
    plot_fold_f1(metrics)

    print("Generative comparison:")
    plot_feasibility_cmp(reports)
    plot_diversity_cmp(reports)
    plot_consistency_cmp(reports)
    plot_novelty_vs_feasibility(reports)
    plot_vec_validation(R)
    plot_im_dhmix_delta(R)
    plot_gonogo(R)
    plot_gen_vs_train_vec(R, feat)

    print(f"\nDone. Figures in {OUT}/")


if __name__ == "__main__":
    main()
