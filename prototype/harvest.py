# -*- coding: utf-8 -*-
"""Build a training archive from MOSDAC without filling the disk.

The arithmetic that motivates this file:

    one L1C granule on disk            88 MB
    the part we actually train on      0.066 MB   (one channel, 128x128 box)
    ratio                              0.056%

So a month of granules is ~124 GB if kept, and ~94 MB if cropped on arrival.
Download bandwidth is the real constraint, not storage, and there is no reason
to pay the storage cost as well.

This harvester therefore runs a download -> crop -> delete loop, keeping only
the extracted study-box arrays. It appends to a single .npz per day so an
interrupted run resumes instead of restarting, which matters when a month of
data takes days of wall-clock time to pull.

Usage:
    python harvest.py --days 7
    python harvest.py --start 2026-09-01 --end 2026-09-30
    python harvest.py --days 30 --keep-granules      # debugging only
"""
import argparse
import datetime as dt
import os
import re
import sys
import time

import numpy as np
import yaml

# The MOSDAC client and its credentials live outside this repository, so no
# secret is ever committed. Point MEGHDOOT_MOSDAC_DIR at the directory holding
# mosdac_io.py and config.yaml.
NIRA = os.environ.get("MEGHDOOT_MOSDAC_DIR", r"D:\NIRA")
sys.path.insert(0, NIRA)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mosdac_io as M                                      # noqa: E402
from insat import read_tb, tb_to_rain, BOX                 # noqa: E402

CONFIG = os.path.join(NIRA, "config.yaml")
ARCHIVE = r"E:\sih\data\archive"
SCRATCH = r"E:\sih\data\_scratch"

MONTHS = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6,
              JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)


def parse_stamp(identifier):
    m = re.search(r"_(\d{2})([A-Z]{3})(\d{4})_(\d{2})(\d{2})_", identifier)
    if not m:
        return None
    d, mon, y, hh, mm = m.groups()
    try:
        return dt.datetime(int(y), MONTHS[mon], int(d), int(hh), int(mm))
    except (KeyError, ValueError):
        return None


def day_path(day):
    return os.path.join(ARCHIVE, "insat_%s.npz" % day.strftime("%Y%m%d"))


def load_day(day):
    """Existing crops for a day, as {timestamp_string: tb_array}."""
    p = day_path(day)
    if not os.path.exists(p):
        return {}
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def save_day(day, store):
    os.makedirs(ARCHIVE, exist_ok=True)
    tmp = day_path(day) + ".tmp.npz"
    np.savez_compressed(tmp, **store)
    os.replace(tmp, day_path(day))


def search_window(dataset_id, bbox, start, end, attempts=3):
    """One search call, tolerating the gateway's intermittent failures."""
    for i in range(attempts):
        try:
            return M.search(dataset_id,
                            start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            bbox=bbox, count="100", timeout=45)
        except Exception as e:                              # noqa: BLE001
            if i == attempts - 1:
                print("    search failed: %s" % str(e)[:90])
                return {}
            time.sleep(2.0 * (i + 1))
    return {}


def harvest(start_day, end_day, cfg, keep_granules=False, channel="TIR1"):
    mc = cfg["mosdac"]
    os.makedirs(SCRATCH, exist_ok=True)
    token, _ = M.get_token(mc["username"], mc["password"])
    if not token:
        print("could not obtain a MOSDAC token")
        return 1

    total_new = total_have = total_fail = 0
    day = start_day
    while day <= end_day:
        nxt = day + dt.timedelta(days=1)
        store = load_day(day)
        before = len(store)

        res = search_window(mc["dataset_id"], mc["bbox"], day, nxt)
        entries = res.get("entries") or []
        dated = [(parse_stamp(e.get("identifier", "")), e) for e in entries]
        dated = sorted([(t, e) for t, e in dated if t and t.date() == day.date()],
                       key=lambda p: p[0])

        print("%s  %3d granules listed, %d already stored"
              % (day.strftime("%Y-%m-%d"), len(dated), before))

        for t, e in dated:
            key = t.strftime("%H%M")
            if key in store:
                total_have += 1
                continue
            ident = e.get("identifier")
            local = os.path.join(SCRATCH, ident)
            try:
                # token expires on long runs; refresh and retry once
                try:
                    M.download(e.get("id"), ident, token, SCRATCH)
                except Exception:                           # noqa: BLE001
                    token, _ = M.get_token(mc["username"], mc["password"],
                                           force=True)
                    M.download(e.get("id"), ident, token, SCRATCH)

                tb, _meta = read_tb(local, box=BOX, channel=channel)
                store[key] = tb.astype(np.float32)
                total_new += 1
                print("    %s  Tb %.0f-%.0f K" % (key, tb.min(), tb.max()))
            except Exception as exc:                        # noqa: BLE001
                total_fail += 1
                print("    %s  FAILED  %s" % (key, str(exc)[:70]))
            finally:
                if not keep_granules and os.path.exists(local):
                    for _ in range(3):
                        try:
                            os.remove(local)
                            break
                        except OSError:
                            time.sleep(0.5)   # antivirus may still hold it

        if len(store) > before:
            save_day(day, store)
            print("    saved %d frames -> %s"
                  % (len(store), os.path.basename(day_path(day))))
        day = nxt

    print("\nharvest complete: %d new, %d already held, %d failed"
          % (total_new, total_have, total_fail))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2,
                    help="harvest the last N days")
    ap.add_argument("--start", help="YYYY-MM-DD (overrides --days)")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--keep-granules", action="store_true",
                    help="do not delete the .h5 after cropping")
    ap.add_argument("--channel", default="TIR1")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    today = dt.datetime.now(dt.UTC).replace(tzinfo=None,
                                            hour=0, minute=0,
                                            second=0, microsecond=0)
    if args.start:
        start = dt.datetime.strptime(args.start, "%Y-%m-%d")
        end = (dt.datetime.strptime(args.end, "%Y-%m-%d")
               if args.end else today)
    else:
        end = today
        start = end - dt.timedelta(days=args.days - 1)

    print("harvesting %s .. %s  (channel %s)"
          % (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
             args.channel))
    return harvest(start, end, cfg, keep_granules=args.keep_granules,
                   channel=args.channel)


if __name__ == "__main__":
    raise SystemExit(main())
