# -*- coding: utf-8 -*-
"""Weather Union crowd-sourced gauges: the independent ground truth.

Weather Union is Zomato's public network of automatic weather stations. It
fills the one gap satellite data cannot: an observation of rain at the ground,
from an instrument that is not the thing being evaluated.

What it is for, and what it is emphatically not for:

  USED FOR   independent point verification of HE, IMERG and the nowcast at
             station pixels, reported separately from product-vs-product scores
  NOT FOR    a nowcast training target. Point data cannot make a gridded
             field, and pretending otherwise invents structure between
             stations that nothing observed.
  NOT FOR    a live model input. The model must run where there are no
             stations, which is most of the domain.

Two rules this module exists to enforce.

**Accumulate the gauge, never interpolate the satellite.** The matching
direction matters: bucket gauge readings into windows whose edges bound the
scan, and compare bucket to scan. Interpolating the satellite up to gauge
resolution would invent sub-scan structure the instrument never observed, and
any skill measured against it would be measuring the interpolator.

**A zero is not necessarily dry.** A tipping bucket clogged by leaves, dust or
a bird reads exactly 0.0 mm, indistinguishable from genuine dry weather in the
value alone. Operational systems - IMGW-PIB's Radar Conformity Check and NOAA
MRMS's "false zero" flag - resolve this with the neighbourhood: a station
reading zero while its neighbours and the satellite both show rain is masked
for that timestep. Masked, not deleted, and only where the reference is itself
trustworthy. Without this, every clogged gauge becomes a false "model
over-forecast" and the metric punishes the model for the sensor's failure.

On the API. There is no bulk listing endpoint and no history: both documented
endpoints return *current* conditions for one station. The station catalogue is
therefore scraped once from the public stations page, and the archive has to be
accumulated by polling - which is why the plan says to start archiving
immediately and treat a multi-season archive as something built, not downloaded.
"""
import datetime as dt
import json
import os
import re
import time

import numpy as np
import requests

API = "https://www.weatherunion.com/gw/weather/external/v0"
STATIONS_PAGE = "https://www.weatherunion.com/weather-stations"
CACHE = r"E:\sih\data\weatherunion"
CATALOGUE = os.path.join(CACHE, "stations.json")

# Kolkata study box, matching hem.py and imerg.py
LAT_C, LON_C = 22.6, 88.4
HALF_DEG = 512.0 / 2.0 / 111.0

# The plan's false-zero trigger: below this is suspect, not trusted dry.
# Not exactly zero, because a partially clogged bucket still tips sometimes.
FALSE_ZERO_MM = 0.2


def _key():
    """API key from the environment. Never read from a committed file."""
    k = os.environ.get("WEATHER_UNION_API_KEY")
    if not k:
        raise RuntimeError(
            "set WEATHER_UNION_API_KEY. Register free at weatherunion.com; "
            "the key is per-account and must not be committed.")
    return k


def _get(path, params, timeout=30, attempts=3):
    headers = {"X-Zomato-Api-Key": _key()}
    last = None
    for i in range(attempts):
        try:
            r = requests.get("%s/%s" % (API, path), headers=headers,
                             params=params, timeout=timeout)
            if r.status_code == 401:
                raise RuntimeError("401 from Weather Union: key rejected")
            r.raise_for_status()
            return r.json()
        except Exception as e:                              # noqa: BLE001
            last = e
            if i < attempts - 1:
                time.sleep(1.5 * (i + 1))
    raise last


# --------------------------------------------------------------------------
def fetch_catalogue(force=False):
    """Station catalogue, scraped once from the public stations page.

    The API exposes no way to list stations, but the public page embeds the
    whole catalogue as escaped JSON inside its server-rendered payload. Parsed
    once and cached, because it changes rarely and the page is ~240 KB.
    """
    if not force and os.path.exists(CATALOGUE):
        with open(CATALOGUE, encoding="utf-8") as f:
            return json.load(f)

    html = requests.get(STATIONS_PAGE, timeout=40).text
    plain = html.replace('\\"', '"')
    pat = re.compile(
        r'"locality_id":"(ZWL\d+)",'
        r'"locality_name":"([^"]*)",'
        r'"latitude":([0-9.]+),'
        r'"longitude":([0-9.]+)')
    rows = [{"locality_id": a, "locality_name": b,
             "latitude": float(c), "longitude": float(d)}
            for a, b, c, d in pat.findall(plain)]
    if not rows:
        raise RuntimeError("could not parse the station catalogue; the page "
                           "layout has probably changed")
    os.makedirs(CACHE, exist_ok=True)
    with open(CATALOGUE, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return rows


def stations_in_box(lat_c=LAT_C, lon_c=LON_C, half_deg=HALF_DEG):
    """Stations inside the study box.

    The plan requires the station chain to be reported as four numbers:
    available in the box -> passing QC -> successfully collocated -> used in
    the evaluation. This answers the first.
    """
    lon_half = half_deg / float(np.cos(np.radians(lat_c)))
    return [s for s in fetch_catalogue()
            if abs(s["latitude"] - lat_c) <= half_deg
            and abs(s["longitude"] - lon_c) <= lon_half]


def observation(locality_id):
    """Current reading for one station, or {} when it reports nothing."""
    j = _get("get_locality_weather_data", {"locality_id": locality_id})
    return j.get("locality_weather_data") or {}


# --------------------------------------------------------------------------
def snapshot(stations=None, path=None):
    """Append every in-box station's current reading to a daily log.

    Stations that return nulls are recorded as nulls rather than skipped: an
    offline station is a fact about the network, and the QC retention rate the
    plan asks for cannot be computed if gaps are silently dropped.
    """
    stations = stations or stations_in_box()
    os.makedirs(CACHE, exist_ok=True)
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    path = path or os.path.join(CACHE, "wu_%s.jsonl" % now.strftime("%Y%m%d"))

    rows, live = [], 0
    for st in stations:
        try:
            obs = observation(st["locality_id"])
        except Exception:                                   # noqa: BLE001
            obs = {}
        if obs.get("rain_accumulation") is not None:
            live += 1
        rows.append({
            "t": now.isoformat(timespec="seconds"),
            "locality_id": st["locality_id"],
            "name": st["locality_name"],
            "lat": st["latitude"],
            "lon": st["longitude"],
            "rain_mm": obs.get("rain_accumulation"),
            "rain_intensity": obs.get("rain_intensity"),
            "temp_c": obs.get("temperature"),
            "humidity": obs.get("humidity"),
            "wind_speed": obs.get("wind_speed"),
        })

    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows), live, path


def load_archive(folder=CACHE):
    """Every recorded observation, ordered by time."""
    rows = []
    if not os.path.isdir(folder):
        return rows
    for name in sorted(os.listdir(folder)):
        if not (name.startswith("wu_") and name.endswith(".jsonl")):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    rows.sort(key=lambda r: r.get("t", ""))
    return rows


# --------------------------------------------------------------------------
def to_grid_index(lat, lon, grid_lat, grid_lon):
    """Nearest grid cell for a station, or None if it falls outside."""
    gl = np.asarray(grid_lat, dtype=float)
    go = np.asarray(grid_lon, dtype=float)
    j = int(np.argmin(np.abs(gl - float(lat))))
    i = int(np.argmin(np.abs(go - float(lon))))
    # argmin always returns something, so verify the match is actually close
    if abs(gl[j] - lat) > abs(gl[1] - gl[0]) or abs(go[i] - lon) > abs(go[1] - go[0]):
        return None
    return j, i


def false_zero_mask(gauge_mm, neighbour_mm, satellite_mm,
                    threshold=FALSE_ZERO_MM, sat_min=1.0, neigh_min=1.0):
    """True where a reported zero should be masked rather than trusted.

    A station is masked for a timestep when all three hold:

      * it reads below the trigger - not exactly zero, because a partially
        clogged bucket still tips occasionally
      * nearby stations read real rain
      * the satellite pixel over it also reads rain

    One source disagreeing is ordinary sub-pixel variability. All three
    disagreeing at once is the signature of a blocked instrument.

    This masks, it does not delete. A clogged gauge is usually clogged for a
    run of timesteps, so the mask rate per station is itself the signal worth
    acting on.
    """
    g = np.asarray(gauge_mm, dtype=float)
    n = np.asarray(neighbour_mm, dtype=float)
    s = np.asarray(satellite_mm, dtype=float)
    return (g < threshold) & (n >= neigh_min) & (s >= sat_min)


def bucket_to_scan(rows, scan_time, window_min=30):
    """Accumulate gauge readings into the window bounding a satellite scan.

    The bucket is [scan_time, scan_time + window), left-closed and right-open.
    Stated explicitly because bucket conventions silently shift scores and are
    impossible to audit afterwards.

    rain_accumulation is a running total, so the value for a bucket is the
    span within it, not a sum of the readings.
    """
    lo = scan_time
    hi = scan_time + dt.timedelta(minutes=window_min)
    per_station = {}
    for r in rows:
        try:
            t = dt.datetime.fromisoformat(r["t"])
        except (KeyError, ValueError):
            continue
        if not (lo <= t < hi):
            continue
        v = r.get("rain_mm")
        if v is None:
            continue
        per_station.setdefault(r.get("locality_id"), []).append(float(v))
    out = {}
    for sid, vals in per_station.items():
        out[sid] = (max(vals) - min(vals)) if len(vals) > 1 else vals[0]
    return out


if __name__ == "__main__":
    try:
        sts = stations_in_box()
        print("%d stations inside the study box" % len(sts))
        n, live, p = snapshot(sts)
        print("recorded %d (%d reporting rain data) -> %s" % (n, live, p))
    except RuntimeError as e:
        print(e)
