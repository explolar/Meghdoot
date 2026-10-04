# -*- coding: utf-8 -*-
"""Read INSAT-3R L2B Hydro-Estimator granules.

This replaces the L1C path for the live input channel, and it is better on
both axes that matter.

Scientifically: HEM *is* ISRO's operational Hydro-Estimator rain rate, which is
the product the plan names. The L1C path downloaded six raw radiance channels
and applied a published power-law fit to cloud-top temperature as a stand-in,
which is a reasonable approximation but not the operational retrieval. HEM has
the precipitable-water-dependent coefficients, the orographic correction and
the warm-cloud correction built in, because ISRO computed it.

Practically: a HEM granule is 9.6 MB against L1C's 85 MB, an 8.9x reduction,
and downloads in about five seconds rather than twenty. Two months of INSAT
goes from roughly thirty hours to under four.

The geolocation needs care. Latitude and Longitude are int16 with a 0.01 scale
factor, which overflows for anything past 327.67 degrees, so the raw arrays
show impossible values. The GeoX/GeoY coordinate vectors are the reliable
georeference and are what this module uses.
"""
import os

import h5py
import numpy as np

# study box, matching insat.py and imerg.py
LAT_C, LON_C = 22.6, 88.4
BOX = 128                      # 128 cells at 4 km = 512 km across

FILL = -999.0


def _attr(h, name, default=None):
    v = h.attrs.get(name, default)
    if v is None:
        return None
    return float(v[0] if hasattr(v, "__len__") and not isinstance(v, bytes) else v)


def _geo_vectors(h):
    """Return (lat_vec, lon_vec) in degrees for the granule's grid.

    Three things in this file look like a georeference and only one is:

      GeoX / GeoY          plain 0..N-1 index vectors, not coordinates
      Latitude / Longitude int16 grids where off-disc pixels are 32767, so
                           naive scaling produces impossible values
      root attributes      upper_latitude / lower_latitude and
                           left_longitude / right_longitude

    The attributes give the corner bounds of a regular grid, so the per-pixel
    coordinates follow from linear interpolation across the array shape. That
    is what this uses.
    """
    ny, nx = h["HEM"].shape[1], h["HEM"].shape[2]
    top = _attr(h, "upper_latitude")
    bot = _attr(h, "lower_latitude")
    left = _attr(h, "left_longitude")
    right = _attr(h, "right_longitude")
    if None in (top, bot, left, right):
        raise ValueError("granule is missing its corner-coordinate attributes")

    # row 0 is the northern edge, so latitude descends down the array
    lat = np.linspace(top, bot, ny)
    lon = np.linspace(left, right, nx)
    return lat, lon


def read_box(path, box=BOX, lat_c=LAT_C, lon_c=LON_C):
    """Hydro-Estimator rain rate over the study box, in mm/h.

    Returns (rain, meta). Fill values become zero: the product marks no-data
    with -999, and for a rain field the honest reading of "no retrieval" over
    this domain is "no rain detected", not "unknown".
    """
    with h5py.File(path, "r") as h:
        lat_v, lon_v = _geo_vectors(h)
        j = int(np.argmin(np.abs(lat_v - lat_c)))
        i = int(np.argmin(np.abs(lon_v - lon_c)))
        half = box // 2
        j0 = max(j - half, 0)
        i0 = max(i - half, 0)
        j1, i1 = j0 + box, i0 + box

        rain = h["HEM"][0, j0:j1, i0:i1].astype(np.float32)
        meta = {
            "file": os.path.basename(path),
            "date": h.attrs.get("Acquisition_Date"),
            "time_gmt": h.attrs.get("Acquisition_Time_in_GMT"),
            "lat": lat_v[j0:j1],
            "lon": lon_v[i0:i1],
            "dlat_deg": float(abs(lat_v[1] - lat_v[0])),
            "dlon_deg": float(abs(lon_v[1] - lon_v[0])),
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
