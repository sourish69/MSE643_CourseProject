#!/usr/bin/env python3
"""
physics.py
----------
Self-contained, verified physics for HEA compositions. This is the SINGLE
SOURCE OF TRUTH for descriptors across the project: the same functions are used
to (a) re-featurise the training data for the oracle, and (b) featurise
CVAE-generated compositions for scoring and filtering. Using one module on both
sides removes train/serve skew.

Descriptors computed from an element-fraction vector over the palette:
    VEC   valence electron concentration      = sum c_i * VEC_i
    delta atomic-size mismatch (%)            = 100*sqrt(sum c_i (1 - r_i/rbar)^2)
    dchi  electronegativity mismatch          = sqrt(sum c_i (chi_i - chibar)^2)
    dHmix enthalpy of mixing (kJ/mol)         = sum_{i<j} 4 H_ij c_i c_j
    dSmix ideal config. entropy (J/mol/K)     = -R sum c_i ln c_i
    Tm    rule-of-mixtures melting point (K)  = sum c_i Tm_i
    omega Yang-Zhang parameter                = Tm*dSmix / |dHmix|

Feasibility (solid-solution-former) filter -- the Yang-Zhang / Zhang criteria:
    delta <= 6.6 %   AND   omega >= 1.1   AND   -22 <= dHmix <= 7 kJ/mol

Verification against the dataset's own precomputed columns (565 alloys):
    VEC   corr 0.994     delta corr 0.87
    dSmix corr 0.93      dHmix corr 0.967 (on the 519 alloys the binary matrix
                                          fully covers)
Run `python physics.py --verify HEA_features.csv --palette palette.json` to
reproduce these numbers.

NOTE on elect_diff: the dataset used an electronegativity scale we could not
reverse-engineer, so `dchi` here (Pauling) does NOT match the dataset column.
That is fine BECAUSE the oracle is (re)trained on physics.py descriptors, so
both training and generation use this same definition. Do not mix the dataset's
elect_diff with this one.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd

R_GAS = 8.314  # J/mol/K

# --- element property tables (palette elements) ---------------------------- #
VEC = {"Al": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
       "Cu": 11, "Zn": 12, "Zr": 4, "Nb": 5, "Mo": 6, "Hf": 4, "Ta": 5, "W": 6,
       "Si": 4, "C": 4, "Mg": 2}

RADIUS = {"Al": 143.2, "Ti": 146.2, "V": 131.6, "Cr": 124.9, "Mn": 135.0,
          "Fe": 124.1, "Co": 125.1, "Ni": 124.6, "Cu": 127.8, "Zn": 133.2,
          "Zr": 160.2, "Nb": 142.9, "Mo": 136.3, "Hf": 157.8, "Ta": 143.0,
          "W": 137.0, "Si": 115.3, "C": 77.0, "Mg": 160.0}                 # pm

CHI = {"Al": 1.61, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
       "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Zr": 1.33, "Nb": 1.60,
       "Mo": 2.16, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Si": 1.90, "C": 2.55,
       "Mg": 1.31}                                                          # Pauling

TM = {"Al": 933, "Ti": 1941, "V": 2183, "Cr": 2180, "Mn": 1519, "Fe": 1811,
      "Co": 1768, "Ni": 1728, "Cu": 1358, "Zn": 693, "Zr": 2128, "Nb": 2750,
      "Mo": 2896, "Hf": 2506, "Ta": 3290, "W": 3695, "Si": 1687, "C": 3800,
      "Mg": 923}                                                            # K

# --- binary mixing enthalpies (Takeuchi-Inoue), kJ/mol, symmetric ---------- #
_PAIRS = """Al Ti -30|Al V -16|Al Cr -10|Al Mn -19|Al Fe -11|Al Co -19|Al Ni -22|Al Cu -1|Al Zr -44|Al Nb -18|Al Mo -5|Al Hf -39|Al Ta -19|Al W -2|Al Si -19|Al Zn 1|Al Mg -2|
Ti V -2|Ti Cr -7|Ti Mn -8|Ti Fe -17|Ti Co -28|Ti Ni -35|Ti Cu -9|Ti Zr 0|Ti Nb 2|Ti Mo -4|Ti Hf 0|Ti Ta 1|Ti W -6|Ti Si -66|Ti Mg 16|
V Cr -2|V Mn -1|V Fe -7|V Co -14|V Ni -18|V Cu 5|V Zr -4|V Nb -1|V Mo 0|V Hf -2|V Ta -1|V W -1|V Si -48|
Cr Mn 2|Cr Fe -1|Cr Co -4|Cr Ni -7|Cr Cu 12|Cr Zr -12|Cr Nb -7|Cr Mo 0|Cr Hf -9|Cr Ta -7|Cr W 1|Cr Si -37|
Mn Fe 0|Mn Co -5|Mn Ni -8|Mn Cu 4|Mn Zr -15|Mn Nb -4|Mn Mo 5|Mn Hf -12|Mn Ta -4|Mn W 6|Mn Si -45|
Fe Co -1|Fe Ni -2|Fe Cu 13|Fe Zr -25|Fe Nb -16|Fe Mo -2|Fe Hf -21|Fe Ta -15|Fe W 0|Fe Si -35|
Co Ni 0|Co Cu 6|Co Zr -41|Co Nb -25|Co Mo -5|Co Hf -35|Co Ta -24|Co W -1|Co Si -38|
Ni Cu 4|Ni Zr -49|Ni Nb -30|Ni Mo -7|Ni Hf -42|Ni Ta -29|Ni W -3|Ni Si -40|
Cu Zr -23|Cu Nb 3|Cu Mo 19|Cu Hf -17|Cu Ta 2|Cu W 22|Cu Si -19|Cu Zn -6|
Zr Nb 4|Zr Mo -6|Zr Hf 0|Zr Ta 3|Zr W -9|Zr Si -84|
Nb Mo -6|Nb Hf 4|Nb Ta 0|Nb W -8|Nb Si -56|
Mo Hf -4|Mo Ta -5|Mo W 0|Mo Si -35|
Hf Ta 3|Hf W -6|Hf Si -77|
Ta W -7|Ta Si -56|
W Si -31"""

H_MIX: dict[tuple[str, str], float] = {}
for _tok in _PAIRS.replace("\n", "").split("|"):
    _a, _b, _v = _tok.split()
    H_MIX[(_a, _b)] = float(_v); H_MIX[(_b, _a)] = float(_v)

# feasibility thresholds
DELTA_MAX = 6.6      # %
OMEGA_MIN = 1.1
DHMIX_LO, DHMIX_HI = -22.0, 7.0   # kJ/mol


# --------------------------------------------------------------------------- #
def compute_descriptors(F: np.ndarray, palette: list[str]) -> pd.DataFrame:
    """
    F: (N, len(palette)) matrix of element fractions (rows sum to 1).
    Returns a DataFrame with vec, delta, dchi, dHmix, dSmix, Tm, omega, feasible.
    dHmix/omega/feasible are NaN/False where the binary matrix lacks a pair.
    """
    F = np.asarray(F, dtype="float64")
    vecv = np.array([VEC[e] for e in palette])
    rv = np.array([RADIUS[e] for e in palette])
    chiv = np.array([CHI[e] for e in palette])
    tmv = np.array([TM[e] for e in palette])

    vec = F @ vecv
    rbar = F @ rv
    delta = 100.0 * np.sqrt((F * (1 - rv[None, :] / rbar[:, None]) ** 2).sum(1))
    chibar = F @ chiv
    dchi = np.sqrt((F * (chiv[None, :] - chibar[:, None]) ** 2).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        dS = -R_GAS * np.nansum(np.where(F > 0, F * np.log(np.where(F > 0, F, 1)), 0.0), axis=1)
    tm = F @ tmv

    # dHmix via binary matrix (NaN if any present pair missing)
    dH = np.full(len(F), np.nan)
    idx_present = [np.where(row > 0)[0] for row in F]
    for k, pres in enumerate(idx_present):
        tot, ok = 0.0, True
        for a in range(len(pres)):
            for b in range(a + 1, len(pres)):
                ea, eb = palette[pres[a]], palette[pres[b]]
                key = (ea, eb)
                if key not in H_MIX:
                    ok = False; break
                tot += 4 * H_MIX[key] * F[k, pres[a]] * F[k, pres[b]]
            if not ok:
                break
        dH[k] = tot if ok else np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        omega = np.where(np.abs(dH) > 1e-9, tm * dS / (np.abs(dH) * 1000.0), np.nan)

    feasible = ((delta <= DELTA_MAX) & (omega >= OMEGA_MIN) &
                (dH >= DHMIX_LO) & (dH <= DHMIX_HI))
    feasible = np.where(np.isnan(dH), False, feasible)  # unknown-enthalpy -> not feasible

    return pd.DataFrame({"vec": vec, "delta": delta, "dchi": dchi, "dHmix": dH,
                         "dSmix": dS, "Tm": tm, "omega": omega,
                         "feasible": feasible,
                         "dHmix_known": ~np.isnan(dH)})


# --------------------------------------------------------------------------- #
def verify(features_csv: str, palette_json: str) -> None:
    pal = json.load(open(palette_json))
    palette, fc = pal["palette"], pal["frac_columns"]
    df = pd.read_csv(features_csv)
    F = df[fc].to_numpy()
    d = compute_descriptors(F, palette)

    def rep(name, given, comp, mask=None):
        g = given.to_numpy() if hasattr(given, "to_numpy") else given
        m = ~np.isnan(g)
        if mask is not None:
            m &= mask
        corr = np.corrcoef(g[m], comp[m])[0, 1]
        mae = np.abs(g[m] - comp[m]).mean()
        print(f"  {name:8s} corr={corr:.4f}  MAE={mae:.3f}  (n={m.sum()})")

    print("PHYSICS VERIFICATION vs dataset columns:")
    rep("VEC", df["vec"], d["vec"].to_numpy())
    rep("delta", df["atom_size_diff"], d["delta"].to_numpy())
    rep("dSmix", df["dSmix"], d["dSmix"].to_numpy())
    rep("dHmix", df["dHmix"], d["dHmix"].to_numpy(), mask=d["dHmix_known"].to_numpy())
    print(f"  dHmix coverage: {int(d['dHmix_known'].sum())}/{len(df)} alloys "
          f"(rest contain a pair absent from the binary matrix)")
    print("  (elect_diff intentionally not matched -- see module docstring)")


def emit_features(features_csv: str, palette_json: str, out_csv: str) -> None:
    """
    Write an ORACLE-READY feature table: ids/labels + the five descriptors
    recomputed by this module, using the exact column names oracle.py expects
    (vec, atom_size_diff, elect_diff, dHmix, dSmix). Train the oracle on this so
    training and generation share one descriptor definition (no train/serve skew).
    """
    pal = json.load(open(palette_json))
    palette, fc = pal["palette"], pal["frac_columns"]
    df = pd.read_csv(features_csv)
    d = compute_descriptors(df[fc].to_numpy(), palette)
    feats = (d[["vec", "delta", "dchi", "dHmix", "dSmix"]]
             .rename(columns={"delta": "atom_size_diff", "dchi": "elect_diff"})
             .reset_index(drop=True))
    keep = df[["alloy_id", "alloy", "family", "phase", "phase_id"]].reset_index(drop=True)
    out = pd.concat([keep, feats], axis=1)
    out.to_csv(out_csv, index=False)
    print(f"wrote oracle-ready features {out.shape} -> {out_csv}")
    print(f"  columns: {list(out.columns)}")
    n_missing = int(out['dHmix'].isna().sum())
    if n_missing:
        print(f"  note: {n_missing} rows have NaN dHmix (uncovered element pair); "
              "XGBoost handles NaN natively.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", metavar="FEATURES_CSV")
    ap.add_argument("--emit-features", metavar="FEATURES_CSV",
                    help="write oracle-ready descriptor table from a features CSV")
    ap.add_argument("--out", default="HEA_features_physics.csv",
                    help="output path for --emit-features")
    ap.add_argument("--palette", default="palette.json")
    args = ap.parse_args()
    if args.verify:
        verify(args.verify, args.palette)
    elif args.emit_features:
        emit_features(args.emit_features, args.palette, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
