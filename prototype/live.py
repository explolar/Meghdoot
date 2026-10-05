# -*- coding: utf-8 -*-
"""Run the nowcast on the newest INSAT data, now.

Fetches the most recent four HEM scans from MOSDAC, runs the trained model, and
writes a forecast for +30 ... +180 minutes after the newest scan.

Two honesty points are built into the output rather than left for the reader to
work out.

**The forecast is older than it looks.** HEM reaches us roughly an hour after
its scan stamp. A forecast for "+30 min" from a scan stamped 02:45 is valid at
03:15, but if the clock already reads 03:44 the first lead is in the past. The
valid times printed below are absolute, and the age of the data is stated, so a
reader sees how much of the 0-3 h window is actually still ahead of them.

**Only HEM is used on the input side.** IMERG Early arrives hours late and would
be a missing channel in production, so the model is trained and run on four HEM
frames and nothing else. That is the deployable configuration.

Usage:
    python live.py
    python live.py --out E:\\sih\\prototype\\runs\\live_forecast.png
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as ds                                        # noqa: E402
import harvest_hem as H                                     # noqa: E402
import hem                                                  # noqa: E402
from model import UNetNowcast                               # noqa: E402

M = H.M
RUNS = r"E:\sih\prototype\runs"
SCRATCH = r"E:\sih\data\_live"
IST = dt.timedelta(hours=5, minutes=30)
KOLKATA = (22.5726, 88.3639)
THRESHOLDS = (1, 4, 8, 16, 20)


def newest_scans(n=4, step_min=30, lookback_days=1):
    """The newest n HEM granules spaced exactly step_min apart.

    Training windows required frames at exact 30-minute offsets, so live input
    has to match: take the newest scan that has a complete run behind it, which
    may be a little older than the very newest granule if one is missing.
    Returns (scans oldest-first, search_age_minutes_of_newest).
    """
    cfg = yaml.safe_load(open(H.CONFIG, encoding="utf-8"))
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    res = H._search_paged(now - dt.timedelta(days=lookback_days),
                          now + dt.timedelta(days=1), cfg["mosdac"]["bbox"])
    by_time = {}
    for e in res.get("entries") or []:
        t = H.parse_stamp(e.get("identifier", ""))
        if t:
            by_time[t] = e
    if not by_time:
        raise RuntimeError("MOSDAC returned no HEM granules")

    for t in sorted(by_time, reverse=True):
        want = [t - dt.timedelta(minutes=step_min * k)
                for k in range(n - 1, -1, -1)]
        if all(w in by_time for w in want):
            return [(w, by_time[w]) for w in want], now
    raise RuntimeError("no run of %d HEM scans at %d-minute spacing"
                       % (n, step_min))


def fetch_frames(scans):
    """Download the scans in parallel and return (n, 128, 128) mm/h plus the
    georeference of the study box."""
    cfg = yaml.safe_load(open(H.CONFIG, encoding="utf-8"))
    mc = cfg["mosdac"]
    token, _ = M.get_token(mc["username"], mc["password"])
    os.makedirs(SCRATCH, exist_ok=True)

    def one(item):
        t, e = item
        ident = e.get("identifier")
        try:
            return M.download(e.get("id"), ident, token, SCRATCH)
        except Exception:                                   # noqa: BLE001
            tok, _ = M.get_token(mc["username"], mc["password"], force=True)
            return M.download(e.get("id"), ident, tok, SCRATCH)

    with cf.ThreadPoolExecutor(max_workers=len(scans)) as pool:
        paths = list(pool.map(one, scans))

    frames, meta = [], None
    for p in paths:
        rain, meta = hem.read_box(p)
        frames.append(rain)
    return np.stack(frames), meta


def load_model(device):
    with open(os.path.join(RUNS, "live_unet.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model = UNetNowcast(in_ch=meta["in_ch"], leads=meta["leads"], base=32,
                        last_idx=meta["last_idx"],
                        base_space=meta["base_space"]).to(device)
    model.load_state_dict(torch.load(os.path.join(RUNS, "live_unet.pt"),
                                     map_location=device))
    model.eval()
    return model, meta


def plot(frames, forecast, geo, scans, now, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lat, lon = geo["lat"], geo["lon"]            # (box, box), per pixel
    bounds = [0.1, 1, 2, 4, 8, 16, 32, 64]
    # seven intervals plus the open-ended top bin is eight colours
    cmap = ListedColormap(["#c6dbef", "#9ecae1", "#6baed6", "#3182bd",
                           "#fd8d3c", "#e6550d", "#a63603", "#67000d"])
    norm = BoundaryNorm(bounds, cmap.N, extend="max")

    t0 = scans[-1][0]
    panels = [("HEM observed\n%s UTC" % t0.strftime("%H:%M"), frames[-1])]
    for i in range(forecast.shape[0]):
        valid = t0 + dt.timedelta(minutes=30 * (i + 1))
        panels.append(("+%d min\nvalid %s UTC  (%s IST)"
                       % (30 * (i + 1), valid.strftime("%H:%M"),
                          (valid + IST).strftime("%H:%M")), forecast[i]))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.5),
                             sharey=True)
    for ax, (title, field) in zip(axes, panels):
        # pcolormesh on the true per-pixel coordinates: the grid is tilted
        # relative to lat/lon, so an axis-aligned image would misplace the
        # edges of the box
        im = ax.pcolormesh(lon, lat, np.where(field >= 0.1, field, np.nan),
                           cmap=cmap, norm=norm, shading="nearest")
        ax.set_facecolor("#f4f4f2")
        ax.plot(KOLKATA[1], KOLKATA[0], marker="*", color="black",
                markersize=8, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("lon", fontsize=7)
    axes[0].set_ylabel("lat", fontsize=7)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                      pad=0.18, aspect=50, ticks=bounds)
    cb.set_label("rain rate (mm/h)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    age = (now - t0).total_seconds() / 60.0
    fig.suptitle("MEGHDOOT live nowcast  |  newest scan %s UTC, %d min old  |  "
                 "star = Kolkata" % (t0.strftime("%d %b %H:%M"), age),
                 fontsize=9)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RUNS, "live_forecast.png"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(device)

    scans, now = newest_scans(n=meta["in_ch"])
    t0 = scans[-1][0]
    age = (now - t0).total_seconds() / 60.0
    print("clock now    : %s UTC  (%s IST)"
          % (now.strftime("%d %b %H:%M"), (now + IST).strftime("%H:%M")))
    print("input scans  : %s UTC" % ", ".join(
        t.strftime("%H:%M") for t, _ in scans))
    print("newest scan is %.0f min old" % age)

    frames, geo = fetch_frames(scans)
    x = np.stack([ds.normalise_rain(f) for f in frames])[None].astype(np.float32)
    with torch.no_grad():
        forecast = model(torch.from_numpy(x).to(device)).cpu().numpy()[0]

    km = geo["dlat_deg"] * 111.0
    print("\nforecast (area = share of the %.0f x %.0f km box, %.1f km pixels)"
          % (frames.shape[1] * km, frames.shape[2] * km, km))
    print("%-8s %-22s %8s" % ("lead", "valid", "max") +
          "".join("%9s" % (">=%g" % t) for t in THRESHOLDS))
    for i in range(forecast.shape[0]):
        valid = t0 + dt.timedelta(minutes=30 * (i + 1))
        past = valid < now
        row = "%-8s %-22s %7.1f " % (
            "+%d min" % (30 * (i + 1)),
            "%s UTC%s" % (valid.strftime("%H:%M"), "  (past)" if past else ""),
            float(forecast[i].max()))
        row += "".join("%8.2f%%" % (100.0 * float((forecast[i] >= t).mean()))
                       for t in THRESHOLDS)
        print(row)

    ahead = [i for i in range(forecast.shape[0])
             if t0 + dt.timedelta(minutes=30 * (i + 1)) >= now]
    print("\nleads still ahead of the clock: %d of %d"
          % (len(ahead), forecast.shape[0]))

    # per-pixel coordinates are 2-D, so find the nearest pixel in both at once
    j, i = np.unravel_index(
        int(np.argmin(np.abs(geo["lat"] - KOLKATA[0])
                      + np.abs(geo["lon"] - KOLKATA[1]))), geo["lat"].shape)
    print("\nat Kolkata (%.2fN %.2fE): observed %.1f mm/h; forecast %s mm/h"
          % (KOLKATA[0], KOLKATA[1], frames[-1][j, i],
             " ".join("%.1f" % forecast[k, j, i]
                      for k in range(forecast.shape[0]))))

    np.savez_compressed(os.path.join(RUNS, "live_forecast.npz"),
                        forecast=forecast, observed=frames,
                        scan_times=np.array([t.isoformat() for t, _ in scans]),
                        lat=geo["lat"], lon=geo["lon"])
    plot(frames, forecast, geo, scans, now, args.out)
    print("figure -> %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
