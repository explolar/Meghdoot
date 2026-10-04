# -*- coding: utf-8 -*-
"""Assemble INSAT inputs and IMERG labels into training windows.

The pairing rule, which is the part that is easy to get wrong:

An INSAT scan stamped 08:15 is acquired over roughly 08:15-08:42. The IMERG
half-hour beginning 08:00 covers 08:00-08:29. Those windows overlap, and the
IMERG slot whose window brackets the scan start is the one used. We do not
interpolate IMERG to the scan instant, because that would invent sub-scan
structure that neither instrument observed.

Splits are by *day*, never by random window. Random windows leak: the same
storm appears in train and test at different offsets, and the reported skill is
then partly memorisation. Day-based splits are the short-run equivalent of the
year-based splits the full plan uses.
"""
import datetime as dt
import glob
import os

import numpy as np

ARCHIVE = r"E:\sih\data\archive"


# --------------------------------------------------------------------------
def load_archive(folder=ARCHIVE, verbose=True, source="hem"):
    """Read every harvested day into {datetime: (input, label)} pairs.

    `source` selects the live input channel:

      "hem"    ISRO's operational Hydro-Estimator rain rate, which is what the
               plan specifies and what the deployed system would read
      "insat"  raw L1C brightness temperature, kept so the earlier runs remain
               reproducible

    Only timestamps present in *both* products are kept: an input frame with no
    IMERG label cannot be trained on, and a label with no input cannot be
    predicted from.
    """
    insat, imerg = {}, {}

    prefix = "hem_" if source == "hem" else "insat_"
    cut = len(prefix)
    for p in sorted(glob.glob(os.path.join(folder, prefix + "*.npz"))):
        day = os.path.basename(p)[cut:cut + 8]
        with np.load(p) as z:
            for k in z.files:
                t = dt.datetime.strptime(day + k, "%Y%m%d%H%M")
                insat[t] = z[k]

    for p in sorted(glob.glob(os.path.join(folder, "imerg_*.npz"))):
        day = os.path.basename(p)[6:14]
        with np.load(p) as z:
            for k in z.files:
                t = dt.datetime.strptime(day + k, "%Y%m%d%H%M")
                imerg[t] = z[k]

    paired = {}
    for t, tb in insat.items():
        # the IMERG slot whose half-hour window contains this scan
        slot = t.replace(minute=0 if t.minute < 30 else 30,
                         second=0, microsecond=0)
        if slot in imerg:
            paired[t] = (tb, imerg[slot])

    if verbose:
        print("archive: %d INSAT frames, %d IMERG frames, %d paired"
              % (len(insat), len(imerg), len(paired)))
        if paired:
            ts = sorted(paired)
            print("         %s .. %s"
                  % (ts[0].strftime("%d %b %H:%M"),
                     ts[-1].strftime("%d %b %H:%M")))
    return paired


# --------------------------------------------------------------------------
def normalise_tb(tb):
    """Brightness temperature to roughly [0, 1], cold = high.

    Inverted on purpose: colder cloud tops mean deeper convection and more
    rain, so the useful signal should be the large number. 180-300 K spans
    everything from the coldest overshooting top to clear ground.
    """
    return np.clip((300.0 - tb) / 120.0, 0.0, 1.0).astype(np.float32)


def normalise_rain(rain):
    """log(1+x) on rain rate.

    Not min-max: the rain-rate histogram is heavily skewed, so min-max leaves
    almost all the data crushed into the bottom few percent of the range, and
    the network sees no gradient where nearly every pixel lives.
    """
    return np.log1p(np.clip(rain, 0.0, None)).astype(np.float32)


def denormalise_rain(x):
    return np.expm1(np.clip(x, 0.0, None)).astype(np.float32)


# --------------------------------------------------------------------------
def build_windows(paired, n_in=3, leads=(1, 2, 3, 4, 5, 6), step_min=30,
                  source="hem"):
    """Carve consecutive-in-time windows out of the paired archive.

    A window is only emitted when every frame it needs is present at exactly
    the right offset. Gaps in the archive therefore break windows rather than
    silently producing a sample that skips an hour.
    """
    times = sorted(paired)
    tset = set(times)
    max_lead = max(leads)
    X, Y, stamps = [], [], []

    for t in times:
        needed = [t - dt.timedelta(minutes=step_min * k)
                  for k in range(n_in - 1, 0, -1)] + [t]
        targets = [t + dt.timedelta(minutes=step_min * L) for L in leads]
        if not all(n in tset for n in needed):
            continue
        if not all(g in tset for g in targets):
            continue

        frames = [normalise_rain(paired[n][1]) for n in needed]   # past rain
        # the live channel: HEM is already a rain rate, so it takes the same
        # log transform as the IMERG frames; L1C is a temperature and needs
        # its own scaling
        extra = paired[t][0]
        frames.append(normalise_rain(extra) if source == "hem"
                      else normalise_tb(extra))
        X.append(np.stack(frames))
        Y.append(np.stack([paired[g][1] for g in targets]))       # mm/h, raw
        stamps.append(t)

    if not X:
        return (np.zeros((0, n_in + 1, 128, 128), np.float32),
                np.zeros((0, len(leads), 128, 128), np.float32), [])
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32), stamps)


def split_by_day(X, Y, stamps, val_frac=0.2, test_frac=0.2):
    """Split on day boundaries so no storm spans two sets."""
    days = sorted({s.date() for s in stamps})
    n = len(days)
    if n < 3:
        # too few days to split cleanly: fall back to a contiguous time split,
        # which still avoids interleaving but cannot guarantee storm separation
        k1 = int(len(X) * (1 - val_frac - test_frac))
        k2 = int(len(X) * (1 - test_frac))
        return ((X[:k1], Y[:k1]), (X[k1:k2], Y[k1:k2]), (X[k2:], Y[k2:]))

    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_days = set(days[-n_test:])
    val_days = set(days[-(n_test + n_val):-n_test])

    idx_tr, idx_va, idx_te = [], [], []
    for i, s in enumerate(stamps):
        d = s.date()
        (idx_te if d in test_days else idx_va if d in val_days
         else idx_tr).append(i)

    return ((X[idx_tr], Y[idx_tr]),
            (X[idx_va], Y[idx_va]),
            (X[idx_te], Y[idx_te]))


def describe(Y, name="targets"):
    """Event counts per threshold: a score over five events is noise."""
    print("%s: %s" % (name, Y.shape))
    for t in (0.1, 1, 4, 8, 16, 20):
        n = int((Y >= t).sum())
        print("   >= %4.1f mm/h : %8d px  (%6.3f%%)"
              % (t, n, 100.0 * n / Y.size if Y.size else 0.0))
