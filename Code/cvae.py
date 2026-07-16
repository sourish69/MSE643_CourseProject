#!/usr/bin/env python3
"""
cvae.py
-------
The generative core: a phase-conditioned Conditional Variational Autoencoder
(CVAE) that designs HEA compositions to order.

Design (matches the project plan)
    * Works over the 19 element-fraction columns (frac_*) from features.py.
    * Conditioned on phase (FCC / BCC / FCC+BCC / Im), one-hot, injected into
      the encoder AND into every decoder layer (defends against the decoder
      ignoring the condition).
    * Decoder ends in a softmax over the element palette, so every generated
      composition is valid by construction (non-negative, sums to 1).
    * Trained on the ELBO = reconstruction + beta * KL, with:
        - KL ANNEALING (beta warmed 0 -> beta_max), and
        - FREE BITS (a per-latent-dim KL floor).
      Together these are the reliable defence against posterior collapse on
      small data (more reliable than simply raising beta).
    * Class imbalance handled by a class-balanced sampler during training, so
      the minority conditions (FCC+BCC) are learned properly.

After training it generates candidates per phase, sparsifies them into clean
compositions, recomputes VEC (for the headline validation), and runs the
DIVERSITY GO/NO-GO check (mean pairwise distance + nearest-training-neighbour
distance) that decides on Day 3 whether the CVAE is healthy or whether to fall
back to the convex-hull sampler.

Usage
    python cvae.py --input HEA_features.csv --palette palette.json \
                   --outdir cvae_out --epochs 400
"""
# Importing necessary packages
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# VEC Schema for each palette element: To recompute VEC of generated compositions
VEC_TABLE = {
    "Al": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12, "Zr": 4, "Nb": 5, "Mo": 6, "Hf": 4, "Ta": 5, "W": 6,
    "Re": 7, "Si": 4, "Sn": 4, "Mg": 2, "C": 4, "N": 5, "Li": 1,
}
PHASE_ORDER = ["FCC", "BCC", "FCC+BCC", "Im"]     # index = phase_id


# Conditional Variational Autoencoder
class Encoder(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, hidden: list[int], z_dim: int):
        super().__init__()
        layers, d = [], x_dim + c_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(d, z_dim)
        self.logvar = nn.Linear(d, z_dim)

    def forward(self, x, c):
        h = self.body(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    """Condition is concatenated at the input of every layer."""
    def __init__(self, z_dim: int, c_dim: int, hidden: list[int], x_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        d = z_dim
        for h in hidden:
            self.layers.append(nn.Linear(d + c_dim, h))
            d = h
        self.out = nn.Linear(d + c_dim, x_dim)

    def forward(self, z, c):
        h = z
        for lin in self.layers:
            h = F.relu(lin(torch.cat([h, c], dim=1)))
        return self.out(torch.cat([h, c], dim=1))          # logits


class CVAE(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, z_dim: int = 8,
                 enc_hidden=(64, 32), dec_hidden=(32, 64)):
        super().__init__()
        self.x_dim, self.c_dim, self.z_dim = x_dim, c_dim, z_dim
        self.encoder = Encoder(x_dim, c_dim, list(enc_hidden), z_dim)
        self.decoder = Decoder(z_dim, c_dim, list(dec_hidden), x_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, c), mu, logvar

    @torch.no_grad()
    def generate(self, c_onehot, n, device):
        z = torch.randn(n, self.z_dim, device=device)
        logits = self.decoder(z, c_onehot)
        return F.softmax(logits, dim=1)                    # valid compositions



# Loss Calculation

def elbo_loss(logits, x, mu, logvar, beta, free_bits, recon="mse"):
    probs = F.softmax(logits, dim=1)
    if recon == "mse":
        rec = F.mse_loss(probs, x, reduction="none").sum(1).mean()
    else:  # soft cross-entropy between target and predicted distributions
        rec = -(x * F.log_softmax(logits, dim=1)).sum(1).mean()

    # KL per latent dim, averaged over batch; free-bits floor prevents collapse.
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)
    if free_bits > 0:
        kl_dim = torch.clamp(kl_dim, min=free_bits)
    kl = kl_dim.sum()
    return rec + beta * kl, rec.detach(), kl.detach()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_data(features_csv: str, palette_json: str):
    with open(palette_json) as fh:
        frac_cols = json.load(fh)["frac_columns"]
    df = pd.read_csv(features_csv)
    X = df[frac_cols].to_numpy(dtype="float32")
    y = df["phase_id"].to_numpy().astype("int64")
    palette = [c.replace("frac_", "") for c in frac_cols]
    return df, X, y, frac_cols, palette


def make_loader(X, y, n_classes, batch, seed):
    """Class-balanced sampler so minority conditions are learned properly."""
    counts = np.bincount(y, minlength=n_classes).astype("float64")
    w = (1.0 / np.maximum(counts, 1))[y]
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=len(y), replacement=True,
                                    generator=g)
    ds = TensorDataset(torch.from_numpy(X),
                       F.one_hot(torch.from_numpy(y), n_classes).float())
    return DataLoader(ds, batch_size=batch, sampler=sampler, drop_last=False)


# --------------------------------------------------------------------------- #
# Diversity go/no-go
# --------------------------------------------------------------------------- #

def _pairwise_mean(A: np.ndarray, cap: int = 800) -> float:
    if len(A) > cap:
        A = A[np.random.RandomState(0).choice(len(A), cap, replace=False)]
    d = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
    iu = np.triu_indices(len(A), 1)
    return float(d[iu].mean()) if len(iu[0]) else 0.0


def _nn_to_train(gen: np.ndarray, train: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(gen[:, None, :] - train[None, :, :], axis=2)
    return d.min(1)


def diversity_report(gen: np.ndarray, train_same_class: np.ndarray, label: str):
    gd = _pairwise_mean(gen)
    td = _pairwise_mean(train_same_class) if len(train_same_class) > 2 else float("nan")
    nn = _nn_to_train(gen, train_same_class) if len(train_same_class) else np.array([np.nan])
    ratio = gd / td if td and not np.isnan(td) else float("nan")
    verdict = "OK" if (ratio >= 0.5 and np.nanmean(nn) > 1e-3) else "COLLAPSE-RISK"
    print(f"  [{label:<8}] gen_spread={gd:.3f}  train_spread={td:.3f}  "
          f"ratio={ratio:.2f}  mean_NN_to_train={np.nanmean(nn):.3f}  -> {verdict}")
    return {"label": label, "gen_spread": gd, "train_spread": td,
            "ratio": float(ratio), "mean_nn_to_train": float(np.nanmean(nn)),
            "verdict": verdict}


def sparsify(frac: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    """Zero tiny softmax leakage and renormalise -> clean, few-element recipes."""
    f = np.where(frac < thresh, 0.0, frac)
    s = f.sum(1, keepdims=True)
    return np.divide(f, s, out=np.zeros_like(f), where=s > 0)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #

def train(model, loader, val, cfg, device):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    hist = {"recon": [], "kl": [], "beta": [], "val_recon": []}
    best_val, best_state = float("inf"), None
    Xv, Cv = val

    for epoch in range(1, cfg["epochs"] + 1):
        beta = cfg["beta_max"] * min(1.0, epoch / max(1, cfg["anneal_epochs"]))
        model.train()
        er, ek, nb = 0.0, 0.0, 0
        for xb, cb in loader:
            xb, cb = xb.to(device), cb.to(device)
            logits, mu, logvar = model(xb, cb)
            loss, rec, kl = elbo_loss(logits, xb, mu, logvar, beta,
                                      cfg["free_bits"], cfg["recon"])
            opt.zero_grad(); loss.backward(); opt.step()
            er += rec.item(); ek += kl.item(); nb += 1

        model.eval()
        with torch.no_grad():
            vlogits, vmu, vlogvar = model(Xv.to(device), Cv.to(device))
            vloss, vrec, _ = elbo_loss(vlogits, Xv.to(device), vmu, vlogvar,
                                       beta, cfg["free_bits"], cfg["recon"])
        hist["recon"].append(er / nb); hist["kl"].append(ek / nb)
        hist["beta"].append(beta); hist["val_recon"].append(vrec.item())
        if vrec.item() < best_val:
            best_val = vrec.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % max(1, cfg["epochs"] // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  beta={beta:.2f}  recon={er/nb:.4f}  "
                  f"KL={ek/nb:.3f}  val_recon={vrec.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Train the phase-conditioned CVAE.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--palette", required=True)
    ap.add_argument("--outdir", default="cvae_out")
    ap.add_argument("--z-dim", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta-max", type=float, default=0.12)
    ap.add_argument("--anneal-epochs", type=int, default=100)
    ap.add_argument("--free-bits", type=float, default=0.4)
    ap.add_argument("--recon", choices=["mse", "ce"], default="ce")
    ap.add_argument("--n-generate", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, X, y, frac_cols, palette = load_data(args.input, args.palette)
    n_classes = len(PHASE_ORDER)
    x_dim = X.shape[1]

    # family-held-out validation split (monitoring only)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    tr, va = next(gss.split(X, y, df["family"].to_numpy()))
    loader = make_loader(X[tr], y[tr], n_classes, args.batch, args.seed)
    Xv = torch.from_numpy(X[va])
    Cv = F.one_hot(torch.from_numpy(y[va]), n_classes).float()

    cfg = dict(epochs=args.epochs, lr=args.lr, beta_max=args.beta_max,
               anneal_epochs=args.anneal_epochs, free_bits=args.free_bits,
               recon=args.recon)

    model = CVAE(x_dim, n_classes, z_dim=args.z_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 64)
    print(f"CVAE  x_dim={x_dim}  z_dim={args.z_dim}  cond={n_classes}  "
          f"params={n_params}  train={len(tr)}  val={len(va)}  device={device}")
    print("=" * 64)
    hist = train(model, loader, (Xv, Cv), cfg, device)

    # ---- save model ---- #
    torch.save({"state_dict": model.state_dict(),
                "config": {"x_dim": x_dim, "c_dim": n_classes, "z_dim": args.z_dim,
                           "frac_cols": frac_cols, "palette": palette,
                           "phase_order": PHASE_ORDER}},
               os.path.join(args.outdir, "cvae.pt"))

    # ---- generate per phase + VEC + diversity go/no-go ---- #
    vec_vec = np.array([VEC_TABLE[e] for e in palette], dtype="float64")
    print("\nGENERATION + DIVERSITY GO/NO-GO:")
    rows, div = [], []
    model.eval()
    for pid, name in enumerate(PHASE_ORDER):
        c = F.one_hot(torch.full((args.n_generate,), pid), n_classes).float().to(device)
        gen = model.generate(c, args.n_generate, device).cpu().numpy()
        gen = sparsify(gen)
        vec_gen = gen @ vec_vec
        train_same = X[y == pid]
        div.append(diversity_report(gen, train_same, name))
        for i in range(len(gen)):
            row = {"requested_phase": name, "vec_recomputed": float(vec_gen[i])}
            row.update({f"frac_{e}": float(gen[i, j]) for j, e in enumerate(palette)})
            rows.append(row)
    gen_df = pd.DataFrame(rows)
    gen_df.to_csv(os.path.join(args.outdir, "generated_compositions.csv"), index=False)

    # VEC-rule quick look (headline validation preview)
    fcc_v = gen_df.loc[gen_df.requested_phase == "FCC", "vec_recomputed"].mean()
    bcc_v = gen_df.loc[gen_df.requested_phase == "BCC", "vec_recomputed"].mean()
    print(f"\n  VEC preview: generated FCC mean VEC={fcc_v:.2f}  "
          f"BCC mean VEC={bcc_v:.2f}  (expect FCC high >~7, BCC low <~6)")

    # ---- training-curve figure ---- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(hist["recon"], label="train recon")
    ax[0].plot(hist["val_recon"], label="val recon"); ax[0].set_title("reconstruction")
    ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["kl"], color="#B42318", label="KL")
    ax[1].plot(hist["beta"], color="#2E5984", label="beta")
    ax[1].set_title("KL & beta (watch KL>0 => no collapse)")
    ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "training_curves.png"), dpi=140)

    with open(os.path.join(args.outdir, "diversity.json"), "w") as fh:
        json.dump({"per_class": div,
                   "vec_preview": {"FCC": float(fcc_v), "BCC": float(bcc_v)}},
                  fh, indent=2)

    print("\n" + "=" * 64)
    print(f"artifacts -> {args.outdir}/  : cvae.pt  generated_compositions.csv  "
          f"training_curves.png  diversity.json")
    print("If any class shows COLLAPSE-RISK, fall back to the convex-hull sampler "
          "for that class (contingency plan).")
    print("=" * 64)


if __name__ == "__main__":
    main()
