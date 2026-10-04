# -*- coding: utf-8 -*-
"""Does a pretrained precipitation checkpoint beat random initialisation?

Four variants, one split, one loss, same seeds:

    unet-scratch    our 7.8M U-Net from random init. The previous reference.
    smaat-scratch   SmaAt-UNet architecture, random init.
    smaat-T1        SmaAt-UNet, pretrained weights, every layer fine-tuned.
    smaat-T2        SmaAt-UNet, pretrained weights, encoder frozen.

The pairing that carries the argument is smaat-scratch against smaat-T1. They
share an architecture, a loss, a learning rate and a data order, so any gap
between them is the effect of the pretrained weights and nothing else. Comparing
smaat-T1 against unet-scratch would confound transfer with architecture.

smaat-T2 is the configuration section 7.6 of the plan prefers for a thin
archive: freeze what the checkpoint knows about storm structure and adapt only
the decoder. For it to be a real freeze the BatchNorm layers in the frozen
encoder are held in eval mode as well. Left in train mode they keep updating
their running statistics from our data, which quietly re-adapts the encoder
and makes "frozen" untrue.

The checkpoint was trained on Dutch (KNMI) precipitation. Indian monsoon
convection is a large domain shift, which is exactly what makes the question
worth testing rather than assuming.

Every variant is retrained on the current day-based split rather than reusing
earlier numbers: the archive has grown since the loss-weight sweep, so the
splits differ and results would not be comparable.

Usage:
    python transfer_experiment.py --epochs 25 --seeds 2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baselines                                            # noqa: E402
import dataset as ds                                        # noqa: E402
import metrics                                              # noqa: E402
import pretrained as P                                      # noqa: E402
from model import UNetNowcast, balanced_loss, count_parameters   # noqa: E402

OUT_DIR = r"E:\sih\prototype\runs"
THRESHOLDS = (1, 4, 8, 16, 20)
REPORT_LEADS = (0, 2, 5)                  # +30, +90, +180 min
SCHEME = "moderate"                       # best tail scheme from the sweep

# learning rate is held equal across the three SmaAt variants so that
# scratch-versus-pretrained is the only thing that differs between them
VARIANTS = {
    "unet-scratch":  dict(kind="unet", lr=2e-3),
    "smaat-scratch": dict(kind="smaat", pretrained=False, freeze=False, lr=1e-3),
    "smaat-T1":      dict(kind="smaat", pretrained=True, freeze=False, lr=1e-3),
    "smaat-T2":      dict(kind="smaat", pretrained=True, freeze=True, lr=1e-3),
}
ENCODER = ("inc", "down1", "down2", "down3", "down4",
           "cbam1", "cbam2", "cbam3", "cbam4", "cbam5")


def build(cfg, in_ch, n_leads):
    if cfg["kind"] == "unet":
        return UNetNowcast(in_ch=in_ch, leads=n_leads, base=32)
    m = P.SmaAtUNet(n_channels=in_ch, n_classes=n_leads)
    if cfg.get("pretrained"):
        P.load_pretrained(m)
    if cfg.get("freeze"):
        P.freeze_encoder(m, True)
    return m


def set_train(model, cfg, train):
    model.train(train)
    if train and cfg.get("freeze"):
        # a frozen encoder must not keep updating BatchNorm statistics
        for name, mod in model.named_children():
            if name in ENCODER:
                mod.eval()


def run_epoch(model, loader, opt, device, cfg, train):
    set_train(model, cfg, train)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            loss = balanced_loss(model(xb), yb, scheme=SCHEME)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        total += float(loss) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, X, device, batch=8):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(torch.from_numpy(X[i:i + batch]).to(device))
                   .cpu().numpy())
    return np.concatenate(out)


def csi_by_lead(pred, obs):
    """{lead_index: [CSI per threshold]} for the reported leads."""
    # CSI only. metrics.evaluate also computes FSS per sample at every lead,
    # which is thousands of pooling passes this table never reads.
    return {li: [metrics.scores(pred[:, li], obs[:, li], t)["CSI"]
                 for t in THRESHOLDS] for li in REPORT_LEADS}


def fmt(vals):
    v = np.asarray(vals, dtype=float)
    if v.ndim == 1:
        v = v[None]
    mean = np.nanmean(v, axis=0)
    half = (np.nanmax(v, axis=0) - np.nanmin(v, axis=0)) / 2.0
    out = []
    for m, h in zip(mean, half):
        if not np.isfinite(m):
            out.append("%13s" % "-")
        elif v.shape[0] > 1:
            out.append("%13s" % ("%.3f+-%.3f" % (m, h)))
        else:
            out.append("%13s" % ("%.3f" % m))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--leads", type=int, default=6)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    for n in names:
        if n not in VARIANTS:
            print("unknown variant %r; have %s" % (n, list(VARIANTS)))
            return 1

    P.download()
    ok, missing, extra, bad = P.verify_architecture()
    if not ok:
        print("SmaAt-UNet does not match the checkpoint: %d missing, %d extra, "
              "%d bad shapes" % (len(missing), len(extra), len(bad)))
        return 1

    leads = tuple(range(1, args.leads + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s | loss scheme: %s | seeds: %d"
          % (device, SCHEME, args.seeds), flush=True)

    paired = ds.load_archive()
    X, Y, stamps = ds.build_windows(paired, n_in=3, leads=leads)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = ds.split_by_day(X, Y, stamps)
    print("train %d  val %d  test %d windows" % (len(Xtr), len(Xva), len(Xte)),
          flush=True)
    if len(Xte) == 0:
        print("empty test split")
        return 1

    tr_loader = lambda seed: DataLoader(                    # noqa: E731
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=args.batch, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=args.batch)

    # baselines do not depend on the loss or the seed
    rain_in = np.expm1(Xte[:, :-1])
    clim = Ytr.mean(axis=(0, 1))
    results = {}
    for name, pred in baselines.all_baselines(rain_in, len(leads), clim).items():
        results[name] = [csi_by_lead(pred, Yte)]

    os.makedirs(OUT_DIR, exist_ok=True)
    for vname in names:
        cfg = VARIANTS[vname]
        runs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build(cfg, Xtr.shape[1], len(leads)).to(device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            n_train = sum(p.numel() for p in trainable)
            print("\n%s  seed %d  | %.2f M trainable of %.2f M"
                  % (vname, seed, n_train / 1e6,
                     sum(p.numel() for p in model.parameters()) / 1e6),
                  flush=True)

            opt = torch.optim.AdamW(trainable, lr=cfg["lr"], weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs)
            best, path = float("inf"), os.path.join(
                OUT_DIR, "tr_%s_s%d.pt" % (vname, seed))
            t0 = time.time()
            loader = tr_loader(seed)
            for ep in range(1, args.epochs + 1):
                trl = run_epoch(model, loader, opt, device, cfg, True)
                val = run_epoch(model, va_loader, opt, device, cfg, False)
                sched.step()
                if not np.isfinite(trl):
                    print("  diverged at epoch %d" % ep, flush=True)
                    break
                if val < best:
                    best = val
                    torch.save(model.state_dict(), path)
                if ep in (1, 5, 10, 15, 20, 25):
                    print("  epoch %3d  train %9.2f  val %9.2f"
                          % (ep, trl, val), flush=True)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
            runs.append(csi_by_lead(predict(model, Xte, device), Yte))
            print("  %.1f min, best val %.2f"
                  % ((time.time() - t0) / 60, best), flush=True)
        results[vname] = runs

    # ------------------------------------------------------------ report
    order = (["persistence", "optical flow", "climatology"]
             + [v for v in VARIANTS if v in results])
    for li in REPORT_LEADS:
        print("\n" + "=" * 78)
        print("CSI at +%d min  (test set, %d windows; mean+-half-range over "
              "seeds)" % ((li + 1) * 30, len(Xte)))
        print("=" * 78)
        hdr = "%-15s" % "method" + "".join("%13s" % ("%g mm/h" % t)
                                           for t in THRESHOLDS)
        print(hdr)
        print("-" * len(hdr))
        for name in order:
            if name not in results:
                continue
            rows = [r[li] for r in results[name]]
            print("%-15s%s" % (name, fmt(rows)))

    print("\nEvent counts in the test set:")
    for t, n in metrics.event_counts(Yte[:, 0], THRESHOLDS).items():
        print("   >= %2g mm/h at +30 min : %8d pixels" % (t, n))

    with open(os.path.join(OUT_DIR, "transfer_results.json"), "w") as f:
        json.dump({k: [{str(li): [None if not np.isfinite(x) else float(x)
                                  for x in v] for li, v in run.items()}
                       for run in runs] for k, runs in results.items()},
                  f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
