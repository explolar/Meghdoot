# -*- coding: utf-8 -*-
"""Read INSAT-3R L2B Hydro-Estimator granules.

HEM is ISRO's operational Hydro-Estimator rain rate, the product the plan names
as the live input. A granule is 9.6 MB against L1C's 85 MB and carries rain rate
directly rather than raw radiance, so it is both smaller and closer to the
product the benchmark is measured against.

GEOREFERENCING - read this before touching the crop.

The file offers three things that look like a georeference. Only one is right.

    GeoX / GeoY          plain index vectors 0..N-1. Not coordinates.
    root attributes      upper/lower_latitude, left/right_longitude. These are
                         the full-DISC bounds (about +-81 degrees), not the
                         edges of the array. Interpolating linearly between
                         them looks plausible and is wrong: it put Kolkata
                         about 890 km south and 490 km west of where it is,
                         so every earlier crop showed the wrong region.
    Latitude/Longitude   per-pixel int16 grids in units of 0.01 degrees, with
                         32767 marking off-disc pixels. These are the truth.

So the study box is located by searching the 2-D grids for the pixel nearest the
target, and the coordinate vectors are read back out of those same grids. The
grid is regular at 0.04 degrees (about 4.4 km); that is checked, not assumed,
because a rotated or curved grid would make the 1-D vectors below wrong.

The grid is fixed for the product, so the pixel index is computed once per
process and cached rather than re-reading ~32 MB of coordinates per granule.
"""
import os

import h5py
import numpy as np

# study box centre, matching imerg.py
LAT_C, LON_C = 22.6, 88.4
BOX = 96                   # 96 cells x 0.04 deg = 3.84 deg ~ 420 km. A multiple
                           # of 16 so the U-Net's four poolings divide evenly.
                           # Small enough that, once the grid's tilt is allowed
                           # for, every pixel still lies inside the IMERG crop.
FILL = -999.0
OFF_DISC = 32767           # Latitude/Longitude fill for pixels off the disc

_CACHE = {}


def _locate(h, box, lat_c, lon_c):
    """Row/column of the crop and its coordinate vectors, cached per grid."""
    ny, nx = h["HEM"].shape[1], h["HEM"].shape[2]
    key = (ny, nx, box, round(lat_c, 3), round(lon_c, 3))
    if key in _CACHE:
        return _CACHE[key]

    la = h["Latitude"][:].astype(np.float32)
    lo = h["Longitude"][:].astype(np.float32)
    ok = (la != OFF_DISC) & (lo != OFF_DISC)
    la = np.where(ok, la * 0.01, np.nan)
    lo = np.where(ok, lo * 0.01, np.nan)

    dist = np.abs(la - lat_c) + np.abs(lo - lon_c)
    r, c = np.unravel_index(np.nanargmin(dist), dist.shape)
    half = box // 2
    j0, i0 = r - half, c - half
    j1, i1 = j0 + box, i0 + box
    if j0 < 0 or i0 < 0 or j1 > ny or i1 > nx:
        raise ValueError("study box falls off the edge of the grid")

    lat2d = la[j0:j1, i0:i1].astype(np.float64)
    lon2d = lo[j0:j1, i0:i1].astype(np.float64)
    if not (np.isfinite(lat2d).all() and np.isfinite(lon2d).all()):
        raise ValueError("study box includes off-disc pixels")

    # The grid is NOT regular lat/lon. Over this box longitude drifts by ~0.7
    # degrees from the top row to the bottom row, because meridians converge in
    # the satellite's projection. A 1-D latitude vector plus a 1-D longitude
    # vector would therefore be silently wrong, so the crop keeps a coordinate
    # for every pixel and callers align other products pixel by pixel.
    out = (j0, j1, i0, i1, lat2d, lon2d)
    _CACHE[key] = out
    return out


def read_box(path, box=BOX, lat_c=LAT_C, lon_c=LON_C):
    """Hydro-Estimator rain rate over the study box, in mm/h, north-up.

    Returns (rain, meta). meta["lat"] and meta["lon"] are (box, box) arrays
    holding the true coordinates of every pixel, so a caller aligns another
    product by coordinates rather than by array index. Row 0 is the northern
    edge.

    Fill values become zero: the product marks no-data with -999, and for a rain
    field over this domain "no retrieval" is honestly "no rain detected".
    """
    with h5py.File(path, "r") as h:
        j0, j1, i0, i1, lat2d, lon2d = _locate(h, box, lat_c, lon_c)
        rain = h["HEM"][0, j0:j1, i0:i1].astype(np.float32)
        meta = {
            "file": os.path.basename(path),
            "date": h.attrs.get("Acquisition_Date"),
            "time_gmt": h.attrs.get("Acquisition_Time_in_GMT"),
            "lat": lat2d,                     # (box, box), per pixel
            "lon": lon2d,
            "dlat_deg": float(abs(lat2d[box // 2 + 1, box // 2]
                                  - lat2d[box // 2, box // 2])),
            "dlon_deg": float(abs(lon2d[box // 2, box // 2 + 1]
                                  - lon2d[box // 2, box // 2])),
        }
    rain = np.where(rain <= FILL + 1.0, 0.0, rain)
    rain = np.where(np.isfinite(rain), rain, 0.0)
    return np.clip(rain, 0.0, 200.0).astype(np.float32), meta


def describe(path):
    """One-line summary of a granule, for checking a download by eye."""
    rain, meta = read_box(path)
    return ("%s  %s UTC   %.1f-%.1f mm/h   %.2f%% >= 4"
            % (meta["file"][:30], meta["time_gmt"], rain.min(), rain.max(),
               100.0 * float((rain >= 4).mean())))
