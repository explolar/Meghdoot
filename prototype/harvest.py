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
import concurrent.futures as cf
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

# Concurrent MOSDAC transfers. A single transfer does not saturate the link;
# three measured roughly three times the sequential rate. Beyond that the
# gateway starts dropping large transfers, so the gain reverses.
WORKERS = int(os.environ.get("MEGHDOOT_WORKERS", "3"))

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


def search_window(dataset_id, bbox, start, end, attempts=6):
    """One search call, tolerating the gateway's intermittent failures.

    MOSDAC's API returns 500 sporadically under no particular load. Giving up
    after a couple of tries silently drops a whole day from the archive, and
    across a two-month harvest that is a scatter of missing days that only
    shows up later as broken training windows. The backoff is therefore both
    longer and more patient than the per-request retry inside mosdac_io.
    """
    for i in range(attempts):
        try:
            return M.search(dataset_id,
                            start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            bbox=bbox, count="100", timeout=45)
        except Exception as e:                              # noqa: BLE001
            if i == attempts - 1:
                print("    search FAILED after %d attempts: %s"
                      % (attempts, str(e)[:80]))
                return {}
            time.sleep(min(3.0 * (2 ** i), 45.0))
    return {}


def harvest(start_day, end_day, cfg, keep_granules=False, channel="TIR1"):
    mc = cfg["mosdac"]
    os.makedirs(SCRATCH, exist_ok=True)
    token, _ = M.get_token(mc["username"], mc["password"])
    if not token:
        print("could not obtain a MOSDAC token")
        return 1

    total_new = total_have = total_fail = 0
    missed_days = []
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

        if not dated:
            # an empty day is either a genuine gap in the archive or a search
            # that never succeeded; either way it must be visible, not silent
            missed_days.append(day.strftime("%Y-%m-%d"))
        todo = [(t, e) for t, e in dated if t.strftime("%H%M") not in store]
        total_have += len(dated) - len(todo)

        # Download in parallel, crop serially.
        #
        # A granule is ~100 MB and a single transfer does not saturate the
        # link: measured sequentially this is about one granule every four
        # minutes, which puts two months of data at eight days of wall clock.
        # Several concurrent transfers run at close to the same per-file rate,
        # so the throughput multiplies. The pool is kept small because MOSDAC
        # drops large transfers under load, and the retry path is what makes
        # the whole thing survive that.
        for batch_start in range(0, len(todo), WORKERS):
            batch = todo[batch_start:batch_start + WORKERS]
            got = {}

            def _fetch(item):
                t_, e_ = item
                ident_ = e_.get("identifier")
                try:
                    M.download(e_.get("id"), ident_, token, SCRATCH)
                    return t_, ident_, None
                except Exception as exc:                    # noqa: BLE001
                    return t_, ident_, exc

            with cf.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                for t_, ident_, err in pool.map(_fetch, batch):
                    got[t_] = (ident_, err)

            # one token refresh per batch, then retry whatever failed
            if any(err is not None for _, err in got.values()):
                try:
                    token, _ = M.get_token(mc["username"], mc["password"],
                                           force=True)
                except Exception:                           # noqa: BLE001
                    pass
                for t_, (ident_, err) in list(got.items()):
                    if err is None:
                        continue
                    try:
                        M.download(dict(dated)[t_].get("id"), ident_, token,
                                   SCRATCH)
                        got[t_] = (ident_, None)
                    except Exception as exc:                # noqa: BLE001
                        got[t_] = (ident_, exc)

            for t_ in sorted(got):
                ident_, err = got[t_]
                key = t_.strftime("%H%M")
                local = os.path.join(SCRATCH, ident_)
                if err is not None:
                    total_fail += 1
                    print("    %s  FAILED  %s" % (key, str(err)[:70]))
                else:
                    try:
                        tb, _meta = read_tb(local, box=BOX, channel=channel)
                        store[key] = tb.astype(np.float32)
                        total_new += 1
                        print("    %s  Tb %.0f-%.0f K"
                              % (key, tb.min(), tb.max()))
                    except Exception as exc:                # noqa: BLE001
                        total_fail += 1
                        print("    %s  CROP FAILED  %s" % (key, str(exc)[:60]))
                if not keep_granules and os.path.exists(local):
                    for _ in range(3):
                        try:
                            os.remove(local)
                            break
                        except OSError:
                            time.sleep(0.5)

            # checkpoint after every batch: a day is ~45 min and an
            # interrupted run should not lose it
            if len(store) > before:
                save_day(day, store)

        if len(store) > before:
            save_day(day, store)
            print("    saved %d frames -> %s"
                  % (len(store), os.path.basename(day_path(day))))
        day = nxt

    print("\nharvest complete: %d new, %d already held, %d failed"
          % (total_new, total_have, total_fail))
    if missed_days:
        print("days with no granules (%d): %s"
              % (len(missed_days), ", ".join(missed_days)))
        print("re-run with --start/--end over those dates to fill the gaps")
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
