# -*- coding: utf-8 -*-
"""Read real INSAT-3R L1C granules and turn them into a rain-rate stack.

This is the real-data path. It reads the HDF5 granules MOSDAC distributes,
converts raw counts to brightness temperature through the file's own lookup
table, crops to the eastern-India study box, and applies a published
IR-to-rain-rate relationship.

What is honest about this, and what is not:

  * The grid is genuinely 4 km. The L1C product is Mercator at 4000.4 m in X
    and 4001.0 m in Y, so no resampling is needed to reach the target
    resolution. Nothing is interpolated up.
  * The brightness temperatures are genuinely measured. TIR1 is the 10.8 um
    window channel, which is the channel the architecture takes as input.
  * The rain rate is NOT ISRO's Hydro-Estimator. It is a power-law fit to
    cloud-top temperature, which is the classical GOES Auto-Estimator form.
    The real Hydro-Estimator adds precipitable-water-dependent coefficients,
    an orographic correction, a warm-cloud correction via the level of neutral
    buoyancy, and sub-cloud evaporation, none of which are reproducible from
    the L1C file alone. The operational HE product is a separate MOSDAC
    download, and swapping it in is a one-function change.

So: real satellite, real geometry, real radiometry, approximate retrieval. The
retrieval is the part to replace first, and it is labelled everywhere it is
used so no number from this prototype is mistaken for an HE-based number.
"""
import glob
import os
import re

import h5py
import numpy as np

MOSDAC_DIR = r"C:\Users\ankit\AppData\Local\Temp\nira_mosdac"

# Study box: the Kolkata metropolitan region and its upstream fetch.
# Section 3 of the plan centres the domain near 22.6 N, 88.4 E.
LAT_C, LON_C = 22.6, 88.4
BOX = 128                      # 128 x 128 cells at 4 km = 512 km across

# Mercator parameters, read from the granule's Projection_Information
R_MAJOR = 6378137.0
LON_ORIGIN = 75.0


def _merc_xy(lat, lon):
    """Forward spherical Mercator, matching the product's own projection."""
    x = R_MAJOR * np.radians(lon - LON_ORIGIN)
    y = R_MAJOR * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def granule_time(path):
    """Pull the acquisition timestamp out of the filename."""
    m = re.search(r"_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", os.path.basename(path))
    return (m.group(1), m.group(2)) if m else (None, None)


def list_granules(folder=MOSDAC_DIR):
    """Every L1C granule on disk, in acquisition order."""
    files = sorted(glob.glob(os.path.join(folder, "3RIMG_*_L1C_*.h5")))
    return sorted(files, key=lambda p: (granule_time(p)[1] or ""))


def read_tb(path, box=BOX, lat_c=LAT_C, lon_c=LON_C, channel="TIR1"):
    """Return (tb, meta): brightness temperature in K over the study box.

    Counts are converted through the granule's own IMG_<ch>_TEMP lookup table,
    which is the calibration ISRO ships with the file.
    """
    with h5py.File(path, "r") as h:
        X, Y = h["X"][:], h["Y"][:]
        cx, cy = _merc_xy(lat_c, lon_c)
        i = int(np.argmin(np.abs(X - cx)))
        j = int(np.argmin(np.abs(Y - cy)))
        half = box // 2
        i0, j0 = max(i - half, 0), max(j - half, 0)
        i1, j1 = i0 + box, j0 + box

        counts = h["IMG_%s" % channel][0, j0:j1, i0:i1].astype(np.int32)
        lut = h["IMG_%s_TEMP" % channel][:]
        tb = lut[np.clip(counts, 0, len(lut) - 1)].astype(np.float32)

        meta = {
            "file": os.path.basename(path),
            "date": h.attrs.get("Acquisition_Date"),
            "time_gmt": h.attrs.get("Acquisition_Time_in_GMT"),
            "x_m": X[i0:i1], "y_m": Y[j0:j1],
            "dx_km": float(abs(X[1] - X[0])) / 1000.0,
            "dy_km": float(abs(Y[1] - Y[0])) / 1000.0,
        }
    # the LUT marks invalid counts as NaN; treat them as clear sky
    tb = np.where(np.isfinite(tb), tb, 300.0)
    return tb, meta


def tb_to_rain(tb):
    """Cloud-top temperature to rain rate, power-law form.

    This is the GOES Auto-Estimator relationship: colder tops imply deeper
    convection and heavier rain, with no rain above a threshold temperature.
    It is a stand-in for ISRO's Hydro-Estimator, not a reproduction of it.

        R = 1.1183e11 * exp(-3.6382e-2 * Tb^1.2)      Tb in K, R in mm/h

    Vicente, Scofield and Menzel (1998), BAMS 79, 1883.
    """
    tb = np.asarray(tb, dtype=np.float64)
    rain = 1.1183e11 * np.exp(-3.6382e-2 * np.power(tb, 1.2))
    rain = np.where(tb > 260.0, 0.0, rain)       # warm tops: no deep convection
    rain = np.clip(rain, 0.0, 120.0)
    rain[rain < 0.35] = 0.0                      # drizzle floor, as in the plan
    return rain.astype(np.float32)


def load_sequence(folder=MOSDAC_DIR, box=BOX, channel="TIR1", verbose=True):
    """Load every granule as (rain, tb, metas), ordered in time."""
    files = list_granules(folder)
    if not files:
        raise FileNotFoundError("no INSAT granules in %s" % folder)
    tbs, metas = [], []
    for p in files:
        tb, m = read_tb(p, box=box, channel=channel)
        tbs.append(tb)
        metas.append(m)
        if verbose:
            print("  %s  %s UTC   Tb %.1f-%.1f K" %
                  (m["file"][:28], m["time_gmt"], tb.min(), tb.max()))
    tb_stack = np.stack(tbs)
    rain = np.stack([tb_to_rain(t) for t in tbs])
    return rain, tb_stack, metas


def build_windows(rain, tb, n_in=3, leads=(1, 2, 3)):
    """Carve a short sequence into (inputs, targets) windows.

    inputs : (N, n_in + 1, H, W) - n_in rain frames plus the latest Tb field,
             matching the architecture's "T-2, T-1, T + IR" input spec
    targets: (N, len(leads), H, W)
    """
    max_lead = max(leads)
    X, Y = [], []
    for t in range(n_in - 1, len(rain) - max_lead):
        frames = list(rain[t - n_in + 1:t + 1])
        frames.append(tb[t])
        X.append(np.stack(frames))
        Y.append(np.stack([rain[t + L] for L in leads]))
    if not X:
        raise ValueError(
            "not enough granules: need at least %d, have %d"
            % (n_in + max_lead, len(rain)))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)
