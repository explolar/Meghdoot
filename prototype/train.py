# -*- coding: utf-8 -*-
"""Train the U-Net nowcaster and score it against every baseline.

This is the whole claim of the project in one script: a model is only
interesting if it beats persistence, climatology and optical flow, so all three
run on the same test set in the same pass and the table prints them side by
side. If the model loses, the table says so.

Usage:
    python train.py --epochs 30
    python train.py --epochs 30 --synthetic      # no downloaded data needed
"""
import argparse
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
from model import UNetNowcast, balanced_loss, count_parameters   # noqa: E402

OUT_DIR = r"E:\sih\prototype\runs"
THRESHOLDS = (1, 4, 8, 16, 20)


# --------------------------------------------------------------------------
def load_real(leads, n_in=3):
    paired = ds.load_archive()
    if len(paired) < n_in + max(leads) + 4:
        raise RuntimeError(
            "only %d paired frames in the archive; need at least %d. "
            "Run harvest.py and harvest_imerg.py first, or pass --synthetic."
            % (len(paired), n_in + max(leads) + 4))
    X, Y, stamps = ds.build_windows(paired, n_in=n_in, leads=leads)
    print("windows: %d" % len(X))
    ds.describe(Y, "all targets")
    return ds.split_by_day(X, Y, stamps)


def load_synthetic(leads, n_in=3):
    """Fallback so the pipeline is testable without network access."""
    import data as synth
    tr, va, te = synth.build_dataset(n_train=700, n_val=200, n_test=260,
                                     n_in=n_in, leads=leads)

    def _prep(pair):
        X, Y = pair
        # synthetic data has no IR channel, so append the last rain frame as a
        # stand-in to keep the tensor shape identical to the real path
        X = np.concatenate([np.log1p(X), np.log1p(X[:, -1:])], axis=1)
        return X.astype(np.float32), Y.astype(np.float32)

    return _prep(tr), _prep(va), _prep(te)


# --------------------------------------------------------------------------
def run_epoch(model, loader, opt, device, train=True):
    model.train() if train else model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = balanced_loss(pred, yb)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
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


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--leads", type=int, default=6)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    leads = tuple(range(1, args.leads + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    if args.synthetic:
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_synthetic(leads)
    else:
        try:
            (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = load_real(leads)
        except RuntimeError as e:
            print("\n%s\n" % e)
            return 1

    print("train %s  val %s  test %s" % (Xtr.shape, Xva.shape, Xte.shape))
    if len(Xte) == 0:
        print("empty test split; need more days in the archive")
        return 1

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=args.batch, shuffle=True, drop_last=False)
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=args.batch)

    model = UNetNowcast(in_ch=Xtr.shape[1], leads=len(leads),
                        base=args.base).to(device)
    print("model: %.2f M parameters" % (count_parameters(model) / 1e6))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(OUT_DIR, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(OUT_DIR, "unet_best.pt")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        trl = run_epoch(model, tr_loader, opt, device, train=True)
        val = run_epoch(model, va_loader, opt, device, train=False)
        sched.step()
        flag = ""
        if val < best:
            best = val
            torch.save(model.state_dict(), best_path)
            flag = "  *"
        print("epoch %3d/%d   train %10.3f   val %10.3f%s"
              % (ep, args.epochs, trl, val, flag))

    print("trained in %.1f min; best val %.3f" % ((time.time() - t0) / 60, best))
    model.load_state_dict(torch.load(best_path, map_location=device))

    # ---------------- evaluation: model against every baseline -------------
    print("\n" + "=" * 68)
    print("TEST SET RESULTS")
    print("=" * 68)
    ds.describe(Yte, "test targets")

    # the rain channels of the input, de-normalised, for the baselines
    rain_in = np.expm1(Xte[:, :-1])
    clim = Ytr.mean(axis=(0, 1)) if len(Ytr) else np.zeros(Yte.shape[2:])

    forecasts = {"MEGHDOOT U-Net": predict(model, Xte, device)}
    forecasts.update(baselines.all_baselines(rain_in, len(leads), clim))

    for name, pred in forecasts.items():
        res = metrics.evaluate(pred, Yte, thresholds=THRESHOLDS)
        print("\n--- %s ---" % name)
        print(metrics.summary_table(res, THRESHOLDS, "CSI"))

    # headline comparison at the thresholds that matter
    print("\n" + "=" * 68)
    print("CSI at +30 min, by threshold")
    print("=" * 68)
    hdr = "%-20s" % "method" + "".join("%9s" % ("%g" % t) for t in THRESHOLDS)
    print(hdr)
    print("-" * len(hdr))
    for name, pred in forecasts.items():
        res = metrics.evaluate(pred[:, :1], Yte[:, :1], thresholds=THRESHOLDS)
        row = "%-20s" % name[:20]
        for t in THRESHOLDS:
            v = res[0][t]["CSI"]
            row += "%9s" % ("%.3f" % v if np.isfinite(v) else "-")
        print(row)

    print("\nEvent counts in the test set (a score over a handful of events "
          "is noise):")
    for t, n in metrics.event_counts(Yte, THRESHOLDS).items():
        print("   >= %2g mm/h : %8d pixels" % (t, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
