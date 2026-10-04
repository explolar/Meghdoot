# -*- coding: utf-8 -*-
"""Train one model per loss-weight scheme and compare them on one test set.

The first real-data run exposed the failure the balanced loss was supposed to
prevent: CSI 0.000 at 20 mm/h against persistence at 0.089. The model hedged on
exactly the intensities a warning system exists for.

Measured on this archive, the published TrajGRU weights are far too weak here.
The >= 30 mm/h band holds 0.03% of pixels, so pure inverse frequency would ask
for 3166x against the 1x baseline where the published scheme gives 30 - under-
weighting the heaviest band by about 106x. But pure inverse frequency is not
obviously right either: a 3166x weight lets a handful of pixels dominate every
gradient, and training usually destabilises.

So this sweeps between them and reports what actually happens, rather than
picking a number and hoping. Every scheme trains on the same split and is
scored against the same baselines, because the comparison is the point.

Usage:
    python sweep_weights.py --epochs 25
    python sweep_weights.py --epochs 25 --schemes trajgru,moderate,strong
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
from model import (UNetNowcast, balanced_loss, count_parameters,  # noqa: E402
                   WEIGHT_SCHEMES)

OUT_DIR = r"E:\sih\prototype\runs"
THRESHOLDS = (1, 4, 8, 16, 20)


def run_epoch(model, loader, opt, device, scheme, train=True):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = balanced_loss(pred, yb, scheme=scheme)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # heavier weights mean larger gradients; clipping is what keeps the
            # aggressive schemes from diverging outright
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, X, device, batch=8):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,))


def csi_row(pred, obs, thresholds=THRESHOLDS):
    """CSI at +30 min for each threshold."""
    res = metrics.evaluate(pred[:, :1], obs[:, :1], thresholds=thresholds)
    return [res[0][t]["CSI"] for t in thresholds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--leads", type=int, default=6)
    ap.add_argument("--schemes", default="trajgru,moderate,strong,inv_freq")
    args = ap.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    for s in schemes:
        if s not in WEIGHT_SCHEMES:
            print("unknown scheme %r; have %s" % (s, list(WEIGHT_SCHEMES)))
            return 1

    leads = tuple(range(1, args.leads + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device, flush=True)

    paired = ds.load_archive()
    X, Y, stamps = ds.build_windows(paired, n_in=3, leads=leads)
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = ds.split_by_day(X, Y, stamps)
    print("train %s  val %s  test %s" % (Xtr.shape, Xva.shape, Xte.shape),
          flush=True)
    if len(Xte) == 0:
        print("empty test split")
        return 1

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=args.batch)

    # baselines first: they do not depend on the loss, so they are the fixed
    # reference every scheme is measured against
    rain_in = np.expm1(Xte[:, :-1])
    clim = Ytr.mean(axis=(0, 1)) if len(Ytr) else np.zeros(Yte.shape[2:])
    rows = {}
    for name, pred in baselines.all_baselines(rain_in, len(leads), clim).items():
        rows[name] = csi_row(pred, Yte)

    os.makedirs(OUT_DIR, exist_ok=True)
    for scheme in schemes:
        print("\n" + "=" * 62, flush=True)
        print("scheme: %-10s weights %s"
              % (scheme, WEIGHT_SCHEMES[scheme]), flush=True)
        print("=" * 62, flush=True)

        torch.manual_seed(0)          # same init, so the loss is the variable
        model = UNetNowcast(in_ch=Xtr.shape[1], leads=len(leads),
                            base=args.base).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                           T_max=args.epochs)
        best, best_path = float("inf"), os.path.join(
            OUT_DIR, "unet_%s.pt" % scheme)
        t0 = time.time()

        for ep in range(1, args.epochs + 1):
            trl = run_epoch(model, tr_loader, opt, device, scheme, True)
            val = run_epoch(model, va_loader, opt, device, scheme, False)
            sched.step()
            if not np.isfinite(trl):
                print("  diverged at epoch %d - weights too aggressive" % ep,
                      flush=True)
                break
            if val < best:
                best = val
                torch.save(model.state_dict(), best_path)
            if ep % 5 == 0 or ep == 1:
                print("  epoch %3d  train %12.2f  val %12.2f"
                      % (ep, trl, val), flush=True)

        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        rows["U-Net %s" % scheme] = csi_row(predict(model, Xte, device), Yte)
        print("  %.1f min, best val %.2f" % ((time.time() - t0) / 60, best),
              flush=True)

    # ---------------- the comparison ----------------
    print("\n" + "=" * 72)
    print("CSI at +30 min, by threshold (test set, %d samples)" % len(Xte))
    print("=" * 72)
    hdr = "%-22s" % "method" + "".join("%9s" % ("%g" % t) for t in THRESHOLDS)
    print(hdr)
    print("-" * len(hdr))
    order = ([k for k in rows if not k.startswith("U-Net")]
             + [k for k in rows if k.startswith("U-Net")])
    for name in order:
        row = "%-22s" % name[:22]
        for v in rows[name]:
            row += "%9s" % ("%.3f" % v if np.isfinite(v) else "-")
        print(row)

    print("\nEvent counts in the test set:")
    for t, n in metrics.event_counts(Yte, THRESHOLDS).items():
        print("   >= %2g mm/h : %8d pixels" % (t, n))

    with open(os.path.join(OUT_DIR, "sweep_results.json"), "w") as f:
        json.dump({k: [None if not np.isfinite(x) else float(x) for x in v]
                   for k, v in rows.items()}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
