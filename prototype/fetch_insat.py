# -*- coding: utf-8 -*-
"""Download a continuous run of INSAT-3R L1C granules from MOSDAC.

Reuses the MOSDAC client already written and debugged for NIRA rather than
reimplementing it: that client handles the token refresh, the gateway's
intermittent 5xx responses, and the silently-corrupted-download case that a
byte-count check alone misses.

A nowcasting dataset needs *consecutive* frames. A random sample of granules is
useless here, because the model learns motion from the difference between
successive scans, so this script sorts by acquisition time and takes a
contiguous block.

Usage:
    python fetch_insat.py --hours 8          # 16 granules at 30-min cadence
    python fetch_insat.py --hours 24 --out E:\\sih\\data\\insat
"""
import argparse
import datetime as dt
import os
import re
import sys

import yaml

# See harvest.py: credentials live outside the repository.
NIRA = os.environ.get("MEGHDOOT_MOSDAC_DIR", r"D:\NIRA")
sys.path.insert(0, NIRA)
import mosdac_io as M                                       # noqa: E402

DEFAULT_OUT = r"E:\sih\data\insat"
CONFIG = os.path.join(NIRA, "config.yaml")


def parse_stamp(identifier):
    """'3RIMG_01OCT2026_0815_L1C_SGP_V01R00.h5' -> datetime, or None."""
    m = re.search(r"_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})_", identifier)
    if not m:
        return None
    months = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6,
                  JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
    d, mon, y, hh, mm = m.groups()
    try:
        return dt.datetime(int(y), months[mon], int(d), int(hh), int(mm))
    except (KeyError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0,
                    help="length of the contiguous run to download")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--lookback-days", type=int, default=2,
                    help="how far back to search for available granules")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    mc = cfg["mosdac"]

    end = dt.datetime.now(dt.UTC)
    start = end - dt.timedelta(days=args.lookback_days)
    res = M.search(mc["dataset_id"],
                   start=start.strftime("%Y-%m-%d"),
                   end=end.strftime("%Y-%m-%d"),
                   bbox=mc["bbox"], count="100", timeout=45)
    entries = res.get("entries") or []
    print("search: %s granules available" % len(entries))

    # sort oldest-first by acquisition time, then take the most recent run
    dated = [(parse_stamp(e.get("identifier", "")), e) for e in entries]
    dated = sorted([(t, e) for t, e in dated if t], key=lambda p: p[0])
    want = int(round(args.hours * 2))               # 30-minute cadence
    run = dated[-want:] if len(dated) > want else dated
    if not run:
        print("nothing to download")
        return 1

    print("downloading %d granules: %s -> %s UTC"
          % (len(run), run[0][0].strftime("%d%b %H:%M"),
             run[-1][0].strftime("%d%b %H:%M")))

    token, _ = M.get_token(mc["username"], mc["password"])
    if not token:
        print("could not obtain a MOSDAC token")
        return 1

    os.makedirs(args.out, exist_ok=True)
    ok = skipped = failed = 0
    for i, (t, e) in enumerate(run, 1):
        ident = e.get("identifier")
        dest = os.path.join(args.out, ident)
        if os.path.exists(dest):
            print("  [%2d/%2d] have   %s" % (i, len(run), ident[:34]))
            skipped += 1
            continue
        try:
            # MOSDAC access tokens expire after a few minutes, and a ~90 MB
            # granule can outlive one. On any failure, force a fresh token and
            # retry once: without this every download after the first few
            # returns 401 and the whole run fails.
            try:
                M.download(e.get("id") or e.get("recordId"), ident, token,
                           args.out)
            except Exception:                         # noqa: BLE001
                token, _ = M.get_token(mc["username"], mc["password"],
                                       force=True)
                M.download(e.get("id") or e.get("recordId"), ident, token,
                           args.out)
            print("  [%2d/%2d] got    %s" % (i, len(run), ident[:34]))
            ok += 1
        except Exception as exc:                      # noqa: BLE001
            print("  [%2d/%2d] FAIL   %s  (%s)"
                  % (i, len(run), ident[:34], str(exc)[:70]))
            failed += 1

    print("done: %d downloaded, %d already present, %d failed"
          % (ok, skipped, failed))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
