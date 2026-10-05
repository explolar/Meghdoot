# -*- coding: utf-8 -*-
"""Train the deployable model: HEM history in, IMERG-like rain out.

Everything reported before this used three past IMERG frames as input. IMERG
Early arrives hours after the fact, so a model trained that way has missing
channels in production and cannot run live. Measured today: the newest HEM
granule was 59 minutes behind the clock while IMERG had published no slot at all
for the day after 3.7 hours.

This trains on what a running system actually has: the last four HEM scans. The
label is still IMERG, which is the point of the design - the live product is the
input, the microwave-anchored product is what the model learns to reproduce.

It also compares the residual base in mm/h against the earlier log-space base.
The earlier code added log1p(rain) to a head whose output was scored in mm/h, so
the claimed "persistence floor" was really log1p(persistence): CSI 0.000 at
>= 4 mm/h with the head contributing nothing, against 0.304 for true
persistence. Running both lets the effect be measured instead of asserted.

The baselines are deployable too. Persistence repeats the newest HEM frame and
optical flow advects it; both are scored against IMERG, so they carry the HEM
vs IMERG calibration gap that the model has to learn to close.

Usage:
    python train_live.py --epochs 25 --seeds 2
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
from model import UNetNowcast, balanced_loss                # noqa: E402
from transfer_experiment import (csi_by_lead, fmt, predict,  # noqa: E402
                                 REPORT_LEADS, THRESHOLDS)

OUT_DIR = r"E:\sih\prototype\runs"
SCHEME = "moderate"
N_IN = 4                                  # last four HEM scans -> 4 channels
VARIANTS = {
    "fixed (mm base)":    "mm",
    "legacy (log base)":  "log",
}


def run_epoch(model, loader, opt, device, train):
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            loss = balanced_loss(model(xb), yb, scheme=SCHEME)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--leads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    leads = tuple(range(1, args.leads + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device %s | scheme %s | history = last %d HEM scans (deployable)"
          % (device, SCHEME, N_IN), flush=True)

    paired = ds.load_archive(source="hem")
    X, Y, stamps = ds.build_windows(paired, n_in=N_IN, leads=leads,
                                    source="hem", history="hem")
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = ds.split_by_day(X, Y, stamps)
    print("windows: train %d  val %d  test %d  | input %s"
          % (len(Xtr), len(Xva), len(Xte), tuple(X.shape[1:])), flush=True)
    if len(Xte) == 0:
        print("empty test split")
        return 1

    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=args.batch)

    # deployable baselines: HEM persistence and HEM optical flow, scored
    # against IMERG. No IMERG frame is used anywhere on the input side.
    rain_in = np.expm1(Xte)
    clim = Ytr.mean(axis=(0, 1))
    results = {}
    for name, pred in baselines.all_baselines(rain_in, len(leads), clim).items():
        results[name] = [csi_by_lead(pred, Yte)]

    os.makedirs(OUT_DIR, exist_ok=True)
    best_overall = (float("inf"), None, None)
    for vname, space in VARIANTS.items():
        runs = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = UNetNowcast(in_ch=N_IN, leads=len(leads), base=32,
                                last_idx=N_IN - 1, base_space=space).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs)
            loader = DataLoader(
                TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                batch_size=args.batch, shuffle=True,
                generator=torch.Generator().manual_seed(seed))
            tag = "%s_s%d" % ("mm" if space == "mm" else "log", seed)
            path = os.path.join(OUT_DIR, "live_%s.pt" % tag)
            best, t0 = float("inf"), time.time()
            print("\n%s  seed %d" % (vname, seed), flush=True)
            for ep in range(1, args.epochs + 1):
                trl = run_epoch(model, loader, opt, device, True)
                val = run_epoch(model, va_loader, opt, device, False)
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
            print("  %.1f min, best val %.2f" % ((time.time() - t0) / 60, best),
                  flush=True)
            if space == "mm" and best < best_overall[0]:
                best_overall = (best, path, seed)
        results[vname] = runs

    # the checkpoint a live run loads: best fixed-base model by validation loss
    if best_overall[1]:
        live = os.path.join(OUT_DIR, "live_unet.pt")
        torch.save(torch.load(best_overall[1], map_location="cpu"), live)
        meta = {
            "in_ch": N_IN, "leads": len(leads), "last_idx": N_IN - 1,
            "base_space": "mm", "scheme": SCHEME, "val_loss": best_overall[0],
            "seed": best_overall[2],
            "train_span": [stamps[0].isoformat(), stamps[-1].isoformat()],
            "n_train": int(len(Xtr)),
            "lead_minutes": [30 * l for l in leads],
            "history": "last %d HEM scans, 30-minute spacing" % N_IN,
        }
        with open(os.path.join(OUT_DIR, "live_unet.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print("\nlive checkpoint -> %s (seed %d, val %.2f)"
              % (live, best_overall[2], best_overall[0]))

    order = ["persistence", "optical flow", "climatology"] + list(VARIANTS)
    for li in REPORT_LEADS:
        print("\n" + "=" * 78)
        print("CSI at +%d min  (test %d windows; HEM-only inputs; mean+-half-range)"
              % ((li + 1) * 30, len(Xte)))
        print("=" * 78)
        hdr = "%-18s" % "method" + "".join("%13s" % ("%g mm/h" % t)
                                           for t in THRESHOLDS)
        print(hdr)
        print("-" * len(hdr))
        for name in order:
            if name in results:
                print("%-18s%s" % (name, fmt([r[li] for r in results[name]])))

    print("\nEvent counts in the test set (+30 min):")
    for t, n in metrics.event_counts(Yte[:, 0], THRESHOLDS).items():
        print("   >= %2g mm/h : %8d pixels" % (t, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
