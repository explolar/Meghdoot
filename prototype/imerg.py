# -*- coding: utf-8 -*-
"""GPM IMERG: download, crop to the study box, and align to INSAT scan times.

IMERG is the rain *label*. INSAT brightness temperature is the *input*. That
split is the point of the design: INSAT sees cloud tops every 30 minutes over
all of India at 4 km, but it does not measure rain; IMERG measures rain from a
passive-microwave constellation but at ~11 km and with latency. Training an
INSAT-to-IMERG mapping buys the resolution and latency of one product with the
physical grounding of the other.

Which IMERG run, and why it matters:

    Early   ~4 h latency     forward propagation only
    Late    ~14 h latency    forward and backward propagation
    Final   ~3.5 months      gauge-calibrated, backward propagated

IMERG Final propagates microwave observations *backward* in time, so a Final
frame at time t can contain information from an overpass that happens after t.
A model trained on Final learns to use information that will not exist at
inference, and reports skill it cannot reproduce operationally. This module
defaults to **Early**, which is the run whose latency a 0-3 hour nowcast can
actually live with.

Credentials come from ~/.netrc (machine urs.earthdata.nasa.gov). The GES DISC
archive must also be approved once under Earthdata -> Applications ->
Authorized Apps, or every download returns 401.
"""
import datetime as dt
import os
import re

import numpy as np
import requests

try:
    import h5py
except ImportError:                                        # pragma: no cover
    h5py = None

# IMERG Early Run, V07, half-hourly
BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07"

# study box, matching insat.py
LAT_C, LON_C = 22.6, 88.4
BOX_KM = 512.0                      # 128 cells x 4 km
IMERG_RES = 0.1                     # degrees

CACHE = r"E:\sih\data\imerg"


def _session():
    """A session that authenticates once and reuses the cookie.

    GES DISC answers with a 302 to urs.earthdata.nasa.gov, which authenticates
    and redirects back. requests drops Authorization headers across hosts, so
    the credentials come from ~/.netrc, which requests re-reads per host.

    The cookie jar is what makes this fast. Without a persisted session the
    full OAuth redirect chain is renegotiated on every granule, which measured
    at 60 s for an 8 MB file - about 0.13 MB/s, and nearly all of it handshake
    rather than transfer. Reusing one session drops that to the download time.
    A connection pool sized for the retry loop avoids tearing down TLS between
    granules as well.
    """
    s = requests.Session()
    s.headers["User-Agent"] = "meghdoot-prototype/0.1"
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=4, pool_maxsize=8, max_retries=0)
    s.mount("https://", adapter)
    return s


def warm_up(session):
    """Complete the Earthdata handshake once, so later requests reuse it."""
    try:
        session.get("https://urs.earthdata.nasa.gov/profile", timeout=30)
    except Exception:                                      # noqa: BLE001
        pass
    return session


def granule_name(when):
    """IMERG Early filename for the half-hour slot containing `when`."""
    slot = when.replace(minute=0 if when.minute < 30 else 30,
                        second=0, microsecond=0)
    end = slot + dt.timedelta(minutes=29, seconds=59)
    minutes = slot.hour * 60 + slot.minute
    return ("3B-HHR-E.MS.MRG.3IMERG.%s-S%s-E%s.%04d.V07C.HDF5"
            % (slot.strftime("%Y%m%d"),
               slot.strftime("%H%M%S"),
               end.strftime("%H%M%S"),
               minutes))


def granule_url(when):
    doy = when.timetuple().tm_yday
    return "%s/%d/%03d/%s" % (BASE, when.year, doy, granule_name(when))


def download(when, cache=CACHE, session=None, timeout=180):
    """Fetch one half-hourly granule, returning the local path."""
    os.makedirs(cache, exist_ok=True)
    name = granule_name(when)
    out = os.path.join(cache, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    s = session or _session()
    url = granule_url(when)
    tmp = out + ".part"
    r = s.get(url, timeout=timeout, stream=True)
    if r.status_code == 404:
        raise FileNotFoundError("IMERG granule not published: %s" % name)
    r.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    with open(tmp, "rb") as f:
        if f.read(4) != b"\x89HDF":
            os.remove(tmp)
            raise ValueError(
                "not HDF5 - check that 'NASA GESDISC DATA ARCHIVE' is approved "
                "under Earthdata Authorized Apps")
    os.replace(tmp, out)
    return out


def read_box(path, lat_c=LAT_C, lon_c=LON_C, box_km=BOX_KM):
    """Rain rate over the study box, in mm/h, oriented (lat, lon) north-up."""
    half_deg_lat = (box_km / 2.0) / 111.0
    half_deg_lon = (box_km / 2.0) / (111.0 * np.cos(np.radians(lat_c)))
    with h5py.File(path, "r") as h:
        g = h["Grid"]
        lon = g["lon"][:]
        lat = g["lat"][:]
        # IMERG stores (time, lon, lat); transpose to (lat, lon)
        field = g["precipitation"][0].T.astype(np.float32)
        fill = g["precipitation"].attrs.get("_FillValue", -9999.9)
    i0 = int(np.searchsorted(lon, lon_c - half_deg_lon))
    i1 = int(np.searchsorted(lon, lon_c + half_deg_lon))
    j0 = int(np.searchsorted(lat, lat_c - half_deg_lat))
    j1 = int(np.searchsorted(lat, lat_c + half_deg_lat))
    sub = field[j0:j1, i0:i1]
    sub = np.where(sub == fill, 0.0, sub)
    sub = np.where(np.isfinite(sub), sub, 0.0)
    return np.clip(sub, 0.0, 200.0).astype(np.float32), (lat[j0:j1], lon[i0:i1])


def upsample_to(field, shape):
    """Nearest-neighbour from the IMERG grid onto the INSAT 4 km grid.

    Nearest neighbour, not bilinear: bilinear would manufacture smooth
    gradients between 11 km cells that the instrument never resolved, and the
    model would learn to reproduce that invented structure. Nearest keeps each
    4 km cell honest about which 11 km observation it came from.
    """
    sy = np.linspace(0, field.shape[0] - 1, shape[0]).round().astype(int)
    sx = np.linspace(0, field.shape[1] - 1, shape[1]).round().astype(int)
    return field[np.ix_(sy, sx)]


def parse_insat_time(identifier):
    """'3RIMG_01OCT2026_0815_...' -> datetime."""
    months = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6,
                  JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
    m = re.search(r"_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})_", identifier)
    if not m:
        return None
    d, mon, y, hh, mi = m.groups()
    return dt.datetime(int(y), months[mon], int(d), int(hh), int(mi))


def label_for(scan_time, shape=(128, 128), session=None, cache=CACHE):
    """The IMERG rain field matching one INSAT scan, on the INSAT grid.

    The INSAT scan at 08:15 covers roughly 08:15-08:42, so the IMERG half-hour
    beginning 08:00 is the slot whose window brackets it. We take that slot
    rather than interpolating IMERG up to the scan instant, because
    interpolation would invent sub-scan structure neither instrument observed.
    """
    path = download(scan_time, cache=cache, session=session)
    field, _ = read_box(path)
    return upsample_to(field, shape)
