# -*- coding: utf-8 -*-
"""Harvest INSAT-3R L2B Hydro-Estimator granules.

This supersedes harvest.py for the live input channel. Two reasons, and the
second one is why it matters more than it looks.

Size. A HEM granule is 9.6 MB against L1C's 85 MB, because it carries one
derived field instead of six raw radiance channels. Two months of INSAT drops
from roughly thirty hours of downloading to under four.

Substance. HEM is ISRO's operational Hydro-Estimator rain rate, which is the
product the plan actually specifies as the live input. The L1C path downloaded
brightness temperature and applied a published power-law fit as a stand-in for
the retrieval. That approximation is defensible but it is not what ISRO runs:
the real Hydro-Estimator has precipitable-water-dependent coefficients, an
orographic correction and a warm-cloud correction built into it. Using HEM
removes an entire layer of approximation from the pipeline, and it is also the
product the benchmark is supposed to be measured against.

Usage:
    python harvest_hem.py --start 2026-08-01 --end 2026-09-30
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

# The MOSDAC client and its credentials live outside this repository.
NIRA = os.environ.get("MEGHDOOT_MOSDAC_DIR", r"D:\NIRA")
sys.path.insert(0, NIRA)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mosdac_io as M                                      # noqa: E402
import hem                                                 # noqa: E402

CONFIG = os.path.join(NIRA, "config.yaml")
ARCHIVE = r"E:\sih\data\archive"
SCRATCH = r"E:\sih\data\_scratch_hem"
DATASET = "3RIMG_L2B_HEM"

# HEM granules are ~9.6 MB, so a wider pool than the L1C harvester is safe.
WORKERS = int(os.environ.get("MEGHDOOT_HEM_WORKERS", "4"))

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
    return os.path.join(ARCHIVE, "hem_%s.npz" % day.strftime("%Y%m%d"))


def load_day(day):
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


def _search_paged(start, end, bbox, page=100, max_pages=12):
    """Follow startIndex until the server's totalResults is satisfied.

    The API caps a response at 100 entries. A two-day window holds more than
    that, and results arrive newest-first, so an unpaged request silently drops
    the tail of the earliest day.
    """
    entries, total, idx = [], None, 1
    for _ in range(max_pages):
        res = M.search(DATASET, start=start.strftime("%Y-%m-%d"),
                       end=end.strftime("%Y-%m-%d"), bbox=bbox,
                       count=str(page), start_index=idx, timeout=45)
        got = res.get("entries") or []
        entries.extend(got)
        if total is None:
            try:
                total = int(res.get("totalResults") or 0)
            except (TypeError, ValueError):
                total = 0
        if len(got) < page or len(entries) >= total:
            break
        idx += page
    return {"entries": entries, "totalResults": total}


def search_window(bbox, start, end, attempts=2, budget=90.0):
    """Search, widening the window on dates the server refuses outright.

    Bounded by wall clock, not just attempt count. Some dates 500 no matter
    what is tried - 2026-08-01 has no granules at all - and the previous
    version spent several minutes per such date retrying and widening before
    giving up. Across a sixty-day harvest that is an hour lost to dates that
    hold nothing. The budget caps that: when it is spent, the day is reported
    empty and the harvest moves on.
    """
    t0 = time.time()
    for w_start, w_end in ((start, end),
                           (start, end + dt.timedelta(days=1)),
                           (start - dt.timedelta(days=1),
                            end + dt.timedelta(days=1))):
        for i in range(attempts):
            if time.time() - t0 > budget:
                print("    search gave up on %s after %.0fs"
                      % (start.strftime("%Y-%m-%d"), time.time() - t0))
                return {}
            try:
                return _search_paged(w_start, w_end, bbox)
            except Exception:                               # noqa: BLE001
                if i < attempts - 1:
                    time.sleep(3.0)
    print("    search FAILED for %s" % start.strftime("%Y-%m-%d"))
    return {}


def harvest(start_day, end_day, cfg, keep_granules=False):
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

        res = search_window(mc["bbox"], day, nxt)
        entries = res.get("entries") or []
        dated = [(parse_stamp(e.get("identifier", "")), e) for e in entries]
        dated = sorted([(t, e) for t, e in dated
                        if t and t.date() == day.date()], key=lambda p: p[0])

        print("%s  %3d granules listed, %d already stored"
              % (day.strftime("%Y-%m-%d"), len(dated), before))
        if not dated:
            missed_days.append(day.strftime("%Y-%m-%d"))

        todo = [(t, e) for t, e in dated if t.strftime("%H%M") not in store]
        total_have += len(dated) - len(todo)
        by_time = dict(dated)

        for i in range(0, len(todo), WORKERS):
            batch = todo[i:i + WORKERS]
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

            # one token refresh per batch, then retry what failed
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
                        M.download(by_time[t_].get("id"), ident_, token,
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
                        rain, _meta = hem.read_box(local)
                        store[key] = rain.astype(np.float32)
                        total_new += 1
                        if rain.max() > 1.0:
                            print("    %s  max %5.1f mm/h  %.2f%% >= 4"
                                  % (key, rain.max(),
                                     100.0 * float((rain >= 4).mean())))
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

            if len(store) > before:
                save_day(day, store)

        if len(store) > before:
            print("    saved %d frames -> %s"
                  % (len(store), os.path.basename(day_path(day))))
        day = nxt

    print("\nHEM harvest: %d new, %d already held, %d failed"
          % (total_new, total_have, total_fail))
    if missed_days:
        print("days with no granules (%d): %s"
              % (len(missed_days), ", ".join(missed_days)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--keep-granules", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    today = dt.datetime.now(dt.UTC).replace(tzinfo=None, hour=0, minute=0,
                                            second=0, microsecond=0)
    if args.start:
        start = dt.datetime.strptime(args.start, "%Y-%m-%d")
        end = (dt.datetime.strptime(args.end, "%Y-%m-%d")
               if args.end else start)
    else:
        end = today
        start = end - dt.timedelta(days=args.days - 1)

    print("harvesting HEM %s .. %s"
          % (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return harvest(start, end, cfg, keep_granules=args.keep_granules)


if __name__ == "__main__":
    raise SystemExit(main())
