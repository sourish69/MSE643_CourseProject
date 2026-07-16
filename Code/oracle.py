#!/usr/bin/env python3
"""
oracle.py
---------
Module-1 discriminative model: an XGBoost phase classifier (the "oracle") for
the HEA generative project. It predicts phase (FCC / BCC / FCC+BCC / Im) from
the five physics descriptors, and is validated with:

    * FAMILY-HELD-OUT cross-validation (StratifiedGroupKFold on `family`) so the
      score reflects generalisation to unseen alloy families, not interpolation
      between near-identical cousins.
    * A PHYSICS CHECK that verifies the model relies on the physically-correct
      descriptors: VEC should dominate the FCC vs BCC decision (the VEC rule),
      while dHmix and atomic-size mismatch should drive the Im (intermetallic)
      decision (the Yang-Zhang solid-solution criteria).

Class imbalance (FCC+BCC is the minority class) is handled with balanced
sample weights, and F1-macro is the headline metric.

Outputs
    oracle_metrics.json     aggregate + per-fold metrics
    oof_predictions.csv      out-of-fold predictions for every row
    confusion_matrix.png     from out-of-fold predictions
    feature_importance.png   gain + (if available) SHAP
    oracle_xgb.json          the final model, retrained on all data
                             (this is the scorer the CVAE stage will reuse)

Usage
    python oracle.py --input HEA_features.csv --outdir oracle_out
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

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

# ---- schema ---------------------------------------------------------------- #
FEATURES = ["vec", "atom_size_diff", "elect_diff", "dHmix", "dSmix"]
TARGET = "phase_id"
GROUP = "family"
PHASE_ORDER = ["FCC", "BCC", "FCC+BCC", "Im"]     # index == phase_id
N_SPLITS = 5
SEED = 42

MODEL_PARAMS = dict(
    objective="multi:softprob",
    num_class=len(PHASE_ORDER),
    n_estimators=250,
    max_depth=3,               # shallow: 5 features, ~565 rows -> guard overfit
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=1.0,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=-1,
)


def make_model() -> "xgb.XGBClassifier":
    return xgb.XGBClassifier(**MODEL_PARAMS)


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #

def run_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """Family-held-out CV. Returns out-of-fold predictions + per-fold F1-macro."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), len(PHASE_ORDER)), dtype=float)
    fold_f1 = []

    for k, (tr, te) in enumerate(sgkf.split(X, y, groups), 1):
        w = compute_sample_weight("balanced", y[tr])   # counter class imbalance
        model = make_model()
        model.fit(X.iloc[tr], y[tr], sample_weight=w)
        p = model.predict(X.iloc[te])
        oof_pred[te] = p
        oof_proba[te] = model.predict_proba(X.iloc[te])
        f1 = f1_score(y[te], p, average="macro")
        fold_f1.append(f1)
        print(f"  fold {k}: test_families={len(np.unique(groups[te])):3d} "
              f"n={len(te):3d}  F1-macro={f1:.3f}")
    return oof_pred, oof_proba, np.array(fold_f1)


# --------------------------------------------------------------------------- #
# Physics check
# --------------------------------------------------------------------------- #

def physics_check(df: pd.DataFrame, final_model, X: pd.DataFrame, outdir: str):
    print("\n" + "-" * 62)
    print("PHYSICS CHECK")
    print("-" * 62)

    # (a) Class-conditional descriptor means -- do the classes sit where theory says?
    print("  Class-conditional descriptor means:")
    means = df.groupby("phase")[["vec", "dHmix", "atom_size_diff"]].mean()
    means = means.reindex([p for p in PHASE_ORDER if p in means.index])
    print(means.round(2).to_string())

    # (b) VEC rule on the FCC vs BCC subset: best single-threshold accuracy.
    sub = df[df["phase"].isin(["FCC", "BCC"])]
    if len(sub) > 5:
        v = sub["vec"].to_numpy()
        yb = (sub["phase"] == "FCC").astype(int).to_numpy()  # 1=FCC
        best_acc, best_thr = 0.0, None
        for thr in np.linspace(v.min(), v.max(), 200):
            acc = max(((v >= thr) == yb).mean(), ((v < thr) == yb).mean())
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        print(f"\n  VEC rule (FCC vs BCC): mean VEC  FCC={sub[sub.phase=='FCC'].vec.mean():.2f}"
              f"  BCC={sub[sub.phase=='BCC'].vec.mean():.2f}")
        print(f"  best single-VEC-threshold accuracy on FCC/BCC = {best_acc:.3f} "
              f"(thr~{best_thr:.2f}; textbook boundary ~6.87-8.0)")

    # (c) Feature importance -- gain always; SHAP if available.
    booster = final_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    gain = {f: gain.get(f, 0.0) for f in FEATURES}
    order = sorted(FEATURES, key=lambda f: -gain[f])
    print("\n  XGBoost gain importance (global):")
    for f in order:
        print(f"    {f:<16}: {gain[f]:.1f}")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(order[::-1], [gain[f] for f in order[::-1]], color="#2E5984")
    ax.set_title("Oracle feature importance (XGBoost gain)")
    ax.set_xlabel("gain")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "feature_importance.png"), dpi=140)
    plt.close(fig)

    # SHAP: per-class importance to verify VEC->FCC/BCC and dHmix/delta->Im.
    try:
        import shap
        expl = shap.TreeExplainer(final_model)
        sv = expl.shap_values(X)
        sv = sv if isinstance(sv, list) else [sv[..., c] for c in range(len(PHASE_ORDER))]
        print("\n  SHAP mean|value| per class (top feature per class in brackets):")
        for c, name in enumerate(PHASE_ORDER):
            imp = np.abs(sv[c]).mean(axis=0)
            top = FEATURES[int(np.argmax(imp))]
            row = "  ".join(f"{f}={imp[i]:.2f}" for i, f in enumerate(FEATURES))
            print(f"    {name:<8} [{top:>14}]  {row}")
        shap.summary_plot(sv, X, class_names=PHASE_ORDER, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "shap_summary.png"), dpi=140, bbox_inches="tight")
        plt.close()
        print("  -> shap_summary.png written")
    except Exception as e:
        print(f"\n  (SHAP unavailable: {e}; gain importance used instead)")

    print("  Expectation: VEC tops FCC/BCC; dHmix & atom_size_diff prominent for Im.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Train + validate the XGBoost phase oracle.")
    ap.add_argument("--input", required=True, help="HEA_features.csv from features.py")
    ap.add_argument("--outdir", default="oracle_out", help="directory for artifacts")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    X = df[FEATURES].astype("float64")
    y = df[TARGET].to_numpy().astype(int)
    groups = df[GROUP].to_numpy()

    print("=" * 62)
    print(f"ORACLE — XGBoost phase classifier  (n={len(df)}, {len(FEATURES)} features, "
          f"{len(PHASE_ORDER)} classes)")
    print("=" * 62)
    print("Family-held-out CV:")
    oof_pred, oof_proba, fold_f1 = run_cv(X, y, groups)

    # aggregate metrics from out-of-fold predictions
    f1m = f1_score(y, oof_pred, average="macro")
    acc = accuracy_score(y, oof_pred)
    bacc = balanced_accuracy_score(y, oof_pred)
    per_class = f1_score(y, oof_pred, average=None, labels=list(range(len(PHASE_ORDER))))
    cm = confusion_matrix(y, oof_pred, labels=list(range(len(PHASE_ORDER))))

    print("\n" + "-" * 62)
    print("AGGREGATE (out-of-fold):")
    print(f"  F1-macro (per-fold) : {fold_f1.mean():.3f} +/- {fold_f1.std():.3f}")
    print(f"  F1-macro (pooled)   : {f1m:.3f}")
    print(f"  accuracy            : {acc:.3f}")
    print(f"  balanced accuracy   : {bacc:.3f}")
    print("  per-class F1        : " +
          ", ".join(f"{n}={v:.2f}" for n, v in zip(PHASE_ORDER, per_class)))
    print("  confusion matrix (rows=true, cols=pred), order " + str(PHASE_ORDER) + ":")
    print(cm)

    # confusion matrix figure
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(PHASE_ORDER))); ax.set_xticklabels(PHASE_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(PHASE_ORDER))); ax.set_yticklabels(PHASE_ORDER)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Oracle confusion matrix (out-of-fold)")
    for i in range(len(PHASE_ORDER)):
        for j in range(len(PHASE_ORDER)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "confusion_matrix.png"), dpi=140); plt.close(fig)

    # final model on all data (balanced) -> importance + reusable scorer
    w_all = compute_sample_weight("balanced", y)
    final_model = make_model()
    final_model.fit(X, y, sample_weight=w_all)
    physics_check(df, final_model, X, args.outdir)

    # save artifacts
    out = df[["alloy_id", "alloy", "family", "phase", "phase_id"]].copy()
    out["pred_id"] = oof_pred
    out["pred_phase"] = [PHASE_ORDER[i] for i in oof_pred]
    for c, n in enumerate(PHASE_ORDER):
        out[f"proba_{n}"] = oof_proba[:, c]
    out.to_csv(os.path.join(args.outdir, "oof_predictions.csv"), index=False)

    metrics = {
        "n": int(len(df)), "features": FEATURES, "classes": PHASE_ORDER,
        "f1_macro_mean": float(fold_f1.mean()), "f1_macro_std": float(fold_f1.std()),
        "f1_macro_pooled": float(f1m), "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "per_class_f1": {n: float(v) for n, v in zip(PHASE_ORDER, per_class)},
        "fold_f1": [float(x) for x in fold_f1],
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(args.outdir, "oracle_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    final_model.save_model(os.path.join(args.outdir, "oracle_xgb.json"))

    print("\n" + "=" * 62)
    print(f"artifacts written to: {args.outdir}/")
    print("  oracle_metrics.json  oof_predictions.csv  confusion_matrix.png")
    print("  feature_importance.png  oracle_xgb.json  [shap_summary.png if shap]")
    print("=" * 62)


if __name__ == "__main__":
    main()
