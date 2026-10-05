# -*- coding: utf-8 -*-
"""Put HEM and IMERG on one grid, aligned by coordinates rather than by index.

Why this module exists. The two products were previously paired by array
position: a 128x128 HEM crop against a 128x128 stretch of IMERG. Both are
wrong in different ways, and together they meant the two fields did not overlap.

  * The HEM crop was taken from the wrong place (see hem.py).
  * The IMERG crop is 46 x 50 cells covering 4.5 x 4.9 degrees, stretched to
    128 x 128 by index. HEM's crop covered a different extent, so even with the
    right location pixel (j, i) in one was not pixel (j, i) in the other.
  * IMERG rows were stored south-up while HEM is north-up.

The fix is to align by geography. Each HEM pixel has a true latitude and
longitude, and the IMERG value for it is the nearest IMERG cell by those
coordinates. Nearest neighbour, not bilinear: IMERG cells are ~11 km and HEM
pixels ~4.4 km, and interpolating would invent gradients the microwave
retrieval never resolved.

The archived IMERG frames were saved after the index stretch, so the original
46 x 50 field is recovered first. That is exact: the stretch repeats each source
cell, so sampling the middle of each run of repeats returns it unchanged.
"""
import numpy as np

# the IMERG crop the archive was written from (imerg.read_box, 512 km box)
IMERG_LAT0, IMERG_LON0, IMERG_RES = 20.35, 85.95, 0.1
IMERG_SHAPE = (46, 50)                 # south-up as stored
STORED = 128

_MID = {}


def _mid_indices():
    """For each source row/col, the middle destination pixel of its repeats."""
    if "m" in _MID:
        return _MID["m"]
    ny, nx = IMERG_SHAPE
    sy = np.linspace(0, ny - 1, STORED).round().astype(int)
    sx = np.linspace(0, nx - 1, STORED).round().astype(int)
    rows = np.array([np.where(sy == i)[0][len(np.where(sy == i)[0]) // 2]
                     for i in range(ny)])
    cols = np.array([np.where(sx == j)[0][len(np.where(sx == j)[0]) // 2]
                     for j in range(nx)])
    _MID["m"] = (rows, cols)
    return rows, cols


def recover_imerg(stored):
    """Undo imerg.upsample_to: (128, 128) stretch -> original (46, 50), south-up."""
    rows, cols = _mid_indices()
    return np.asarray(stored)[np.ix_(rows, cols)].astype(np.float32)


def imerg_lat_lon():
    """Cell-centre coordinates of the recovered IMERG field (south-up rows)."""
    lat = IMERG_LAT0 + IMERG_RES * np.arange(IMERG_SHAPE[0])
    lon = IMERG_LON0 + IMERG_RES * np.arange(IMERG_SHAPE[1])
    return lat, lon


def to_grid(stored, lat2d, lon2d):
    """IMERG rain on the HEM grid: one value per HEM pixel, by coordinates.

    lat2d / lon2d are the per-pixel coordinates hem.read_box returns, shape
    (box, box), north-up. They must be 2-D: the HEM grid is tilted relative to
    lat/lon (longitude drifts ~0.7 degrees top to bottom over the box), so rows
    and columns cannot be treated as lines of constant latitude or longitude.
    The output has HEM's shape and orientation.
    """
    orig = recover_imerg(stored)
    ilat = np.clip(np.rint((np.asarray(lat2d) - IMERG_LAT0) / IMERG_RES),
                   0, IMERG_SHAPE[0] - 1).astype(int)
    ilon = np.clip(np.rint((np.asarray(lon2d) - IMERG_LON0) / IMERG_RES),
                   0, IMERG_SHAPE[1] - 1).astype(int)
    return orig[ilat, ilon]


def covered(lat2d, lon2d):
    """True if every HEM pixel lies inside the IMERG crop, so each has a genuine
    label rather than a clipped edge value."""
    lat, lon = imerg_lat_lon()
    h = IMERG_RES / 2.0
    return bool(np.min(lat2d) >= lat[0] - h and np.max(lat2d) <= lat[-1] + h
                and np.min(lon2d) >= lon[0] - h and np.max(lon2d) <= lon[-1] + h)
