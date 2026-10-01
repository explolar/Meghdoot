# -*- coding: utf-8 -*-
"""Synthetic rainfall fields for the MEGHDOOT prototype.

The real system ingests INSAT-3DR Hydro-Estimator and GPM IMERG. Neither is
used here, and that is deliberate: this prototype exists to prove the pipeline
(baselines, loss, metrics, direct multi-lead heads) runs end to end and that
the evaluation harness reports what it claims to. Swapping this module for a
MOSDAC/GES DISC reader is the only change needed to run it on real data.

The generator is built to have the properties that make nowcasting hard, so
the numbers it produces are not trivially good:

  * rain is rare and heavy rain is much rarer, matching the skewed histogram
    that makes plain MSE pathological
  * cells advect with a mean wind plus per-cell deviation, so optical flow is a
    genuinely strong baseline rather than a straw man
  * cells grow and decay on their own timescale, so persistence degrades with
    lead time the way it does in reality
  * new cells initiate at random, which is the component no advection-based
    method can predict and which sets the skill ceiling
"""
import numpy as np

# grid and timing match the real specification: 128x128 at 4 km, 30-min steps
GRID = 128
DX_KM = 4.0
DT_MIN = 30


class RainFieldSimulator:
    """Advecting, growing, decaying convective cells on a periodic grid."""

    def __init__(self, grid=GRID, seed=0, n_cells=14,
                 mean_wind=(1.15, 0.55), init_rate=0.35):
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self.mean_wind = np.array(mean_wind, dtype=float)
        self.init_rate = init_rate
        self.cells = [self._new_cell(spin_up=True) for _ in range(n_cells)]

    def _new_cell(self, spin_up=False):
        g = self.grid
        # lognormal peak intensity: many weak cells, few very heavy ones
        peak = float(self.rng.lognormal(mean=2.45, sigma=0.85))
        peak = min(peak, 110.0)
        return {
            "pos": self.rng.uniform(0, g, size=2),
            # each cell deviates from the mean wind, so a single global
            # motion vector cannot describe the field perfectly
            "vel": self.mean_wind + self.rng.normal(0, 0.28, size=2),
            "sigma": float(self.rng.uniform(1.5, 4.2)),
            "peak": peak,
            # age/life drive a grow-then-decay envelope
            "age": float(self.rng.uniform(0, 14)) if spin_up else 0.0,
            "life": float(self.rng.uniform(7, 26)),
        }

    def _amplitude(self, cell):
        """Grow to full strength at mid-life, decay after. Zero outside life."""
        frac = cell["age"] / cell["life"]
        if frac >= 1.0:
            return 0.0
        return float(np.sin(np.pi * frac) ** 0.75)

    def step(self):
        """Advance one 30-minute step and return the rain-rate field in mm/h."""
        g = self.grid
        yy, xx = np.mgrid[0:g, 0:g]
        field = np.zeros((g, g), dtype=np.float32)

        for cell in self.cells:
            amp = self._amplitude(cell)
            if amp > 0:
                # periodic distance, so cells wrap rather than vanish at edges
                dx = np.abs(xx - cell["pos"][0])
                dy = np.abs(yy - cell["pos"][1])
                dx = np.minimum(dx, g - dx)
                dy = np.minimum(dy, g - dy)
                r2 = dx * dx + dy * dy
                field += (amp * cell["peak"] *
                          np.exp(-r2 / (2.0 * cell["sigma"] ** 2)))
            cell["pos"] = (cell["pos"] + cell["vel"]) % g
            cell["age"] += 1.0

        # retire dead cells, seed new ones: this is convective initiation, and
        # it is the part of the field no advection scheme can anticipate
        self.cells = [c for c in self.cells if c["age"] < c["life"]]
        while self.rng.random() < self.init_rate or len(self.cells) < 8:
            self.cells.append(self._new_cell())
            if len(self.cells) > 22:
                break

        # light stochastic texture, then threshold away drizzle so the field
        # has the sharp wet/dry boundary a real rain field has
        field += self.rng.gamma(1.4, 0.16, size=(g, g)).astype(np.float32)
        field[field < 0.35] = 0.0
        return np.clip(field, 0.0, 120.0)


def make_sequence(n_steps, seed=0, **kw):
    """Return (n_steps, grid, grid) float32 of consecutive rain-rate fields."""
    sim = RainFieldSimulator(seed=seed, **kw)
    return np.stack([sim.step() for _ in range(n_steps)]).astype(np.float32)


def build_dataset(n_train=900, n_val=180, n_test=260, n_in=3, leads=(1, 2, 3, 4, 5, 6),
                  seed=0, verbose=True):
    """Build train/val/test splits as (inputs, targets) arrays.

    Splits come from *different simulator seeds*, not from random windows of
    one run. Random windows of a single sequence leak: the same storm appears
    in train and test at different offsets, and the reported skill is then
    partly memorisation. Separate seeds are the synthetic equivalent of the
    year-based splits the real plan uses.
    """
    max_lead = max(leads)

    def _carve(n_frames, s):
        seq = make_sequence(n_frames, seed=s)
        X, Y = [], []
        for t in range(n_in - 1, len(seq) - max_lead):
            X.append(seq[t - n_in + 1:t + 1])
            Y.append(np.stack([seq[t + L] for L in leads]))
        return (np.asarray(X, dtype=np.float32),
                np.asarray(Y, dtype=np.float32))

    tr = _carve(n_train, seed + 1)
    va = _carve(n_val, seed + 101)
    te = _carve(n_test, seed + 202)
    if verbose:
        print("dataset  train %s -> %s" % (tr[0].shape, tr[1].shape))
        print("         val   %s -> %s" % (va[0].shape, va[1].shape))
        print("         test  %s -> %s" % (te[0].shape, te[1].shape))
    return tr, va, te


def rain_histogram(arr, bins=(0.1, 1, 4, 8, 16, 20, 32, 64)):
    """Fraction of pixels at or above each threshold, for the class table."""
    out = {}
    tot = arr.size
    for b in bins:
        out[b] = float((arr >= b).sum()) / tot
    return out
