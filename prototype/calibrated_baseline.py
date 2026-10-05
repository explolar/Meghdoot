# -*- coding: utf-8 -*-
"""Is the model forecasting, or just correcting HEM's bias against IMERG?

The HEM-only model beats HEM persistence at >= 1 mm/h at every lead time, and
the margin barely shrinks as the lead grows. That pattern is suspicious. A real
nowcasting gain should fade with lead time as the storm evolves away from what
the model can extrapolate. A gain that does not fade is what you would see from
a systematic mismatch between the two products - HEM's idea of "raining" differs
from IMERG's - which a network learns to correct at every lead equally.

Plain persistence cannot correct that mismatch, so beating it proves little.
The fair comparison is persistence given the same one-parameter calibration the
model could learn: for each lead and threshold, pick the HEM cut-off tau that
maximises CSI against IMERG on the *validation* set, then forecast "event" where
HEM >= tau and score it on the held-out test set. If calibrated persistence ties
the model, the model has added calibration and nothing else.

tau is tuned on validation and scored on test. Tuning it on the test set would
be exactly the self-verification inflation the plan warns about.

Usage:
    python calibrated_baseline.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as ds                                        # noqa: E402
import metrics                                              # noqa: E402
from model import UNetNowcast                               # noqa: E402

RUNS = r"E:\sih\prototype\runs"
LEADS = (0, 2, 5)                                  # +30, +90, +180 min
THRESHOLDS = (1, 4, 8)
TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]


def csi(pred_event, obs_event):
    h = int(np.sum(pred_event & obs_event))
    m = int(np.sum(~pred_event & obs_event))
    f = int(np.sum(pred_event & ~obs_event))
    return h / (h + m + f) if (h + m + f) else float("nan")


def best_tau(hem_val, obs_val):
    """The cut-off on HEM that maximises validation CSI."""
    scores = [(csi(hem_val >= t, obs_val), t) for t in TAUS]
    scores = [(s, t) for s, t in scores if np.isfinite(s)]
    return max(scores)[1] if scores else None


@torch.no_grad()
def model_forecast(seed, X, device):
    m = UNetNowcast(in_ch=4, leads=6, base=32, last_idx=3,
                    base_space="mm").to(device)
    m.load_state_dict(torch.load(os.path.join(RUNS, "live_mm_s%d.pt" % seed),
                                 map_location=device))
    m.eval()
    out = []
    for i in range(0, len(X), 8):
        out.append(m(torch.from_numpy(X[i:i + 8]).to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paired = ds.load_archive(verbose=False)
    X, Y, st = ds.build_windows(paired, n_in=4, leads=(1, 2, 3, 4, 5, 6),
                                source="hem", history="hem")
    (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = ds.split_by_day(X, Y, st)
    hem_va = np.expm1(Xva[:, 3])
    hem_te = np.expm1(Xte[:, 3])
    print("validation %d windows (tau is tuned here) | test %d windows "
          "(scored here)\n" % (len(Xva), len(Xte)))

    preds = [model_forecast(s, Xte, device) for s in (0, 1)]

    print("%-9s %-12s %-26s %s" % ("lead", "threshold", "method", "test CSI"))
    print("-" * 70)
    for li in LEADS:
        for thr in THRESHOLDS:
            obs_va = Yva[:, li] >= thr
            obs_te = Yte[:, li] >= thr
            tau = best_tau(hem_va, obs_va)
            rows = [
                ("persistence (HEM >= %g)" % thr, csi(hem_te >= thr, obs_te)),
                ("calibrated persistence (tau=%g)" % tau,
                 csi(hem_te >= tau, obs_te)),
                ("U-Net, mean of 2 seeds",
                 float(np.mean([csi(p[:, li] >= thr, obs_te) for p in preds]))),
            ]
            for k, (name, v) in enumerate(rows):
                print("%-9s %-12s %-32s %.3f" % (
                    "+%d min" % ((li + 1) * 30) if k == 0 else "",
                    ">= %g mm/h" % thr if k == 0 else "", name, v))
            print()
    print("test events at +30 min: >=1: %d  >=4: %d  >=8: %d pixels"
          % tuple(int((Yte[:, 0] >= t).sum()) for t in THRESHOLDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
