# -*- coding: utf-8 -*-
"""Harvest IMERG rain labels for a date range.

IMERG granules are ~8 MB against INSAT's ~88 MB, and the cropped box is a few
kilobytes, so these are cheap to pull. They are the training label: the model
learns INSAT brightness temperature -> IMERG rain rate.

Early Run by default. IMERG Final propagates microwave observations backward in
time, so a Final frame can carry an overpass that happens after its own
timestamp; training on it reports skill that cannot be reproduced at inference.
Early is the run whose ~4 h latency a nowcast can live with.

Usage:
    python harvest_imerg.py --start 2026-09-24 --end 2026-09-25
    python harvest_imerg.py --days 3
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imerg                                               # noqa: E402

ARCHIVE = r"E:\sih\data\archive"


def day_path(day):
    return os.path.join(ARCHIVE, "imerg_%s.npz" % day.strftime("%Y%m%d"))


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


def harvest(start_day, end_day, keep_granules=False):
    # warm the Earthdata handshake once; see imerg._session for why
    session = imerg.warm_up(imerg._session())
    total_new = total_have = total_fail = 0

    day = start_day
    while day <= end_day:
        store = load_day(day)
        before = len(store)
        print("%s" % day.strftime("%Y-%m-%d"))

        for slot in range(48):                      # 48 half-hours per day
            t = day + dt.timedelta(minutes=30 * slot)
            key = t.strftime("%H%M")
            if key in store:
                total_have += 1
                continue
            try:
                path = imerg.download(t, session=session)
                field, _ = imerg.read_box(path)
                store[key] = imerg.upsample_to(field, (128, 128))
                total_new += 1
                if field.max() > 1.0:
                    print("    %s  max %5.1f mm/h   %.2f%% >= 4"
                          % (key, field.max(), 100 * (field >= 4).mean()))
                if not keep_granules and os.path.exists(path):
                    os.remove(path)
                # checkpoint every few slots: a day takes ~15 min and an
                # interrupted run should not lose it
                if total_new % 8 == 0:
                    save_day(day, store)
            except FileNotFoundError:
                pass                                 # not published yet
            except Exception as exc:                 # noqa: BLE001
                total_fail += 1
                print("    %s  FAILED %s" % (key, str(exc)[:70]))

        if len(store) > before:
            save_day(day, store)
            print("    saved %d frames -> %s"
                  % (len(store), os.path.basename(day_path(day))))
        day += dt.timedelta(days=1)

    print("\nIMERG harvest: %d new, %d already held, %d failed"
          % (total_new, total_have, total_fail))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--keep-granules", action="store_true")
    args = ap.parse_args()

    today = dt.datetime.now(dt.UTC).replace(tzinfo=None, hour=0, minute=0,
                                            second=0, microsecond=0)
    if args.start:
        start = dt.datetime.strptime(args.start, "%Y-%m-%d")
        end = (dt.datetime.strptime(args.end, "%Y-%m-%d")
               if args.end else start)
    else:
        end = today
        start = end - dt.timedelta(days=args.days - 1)

    print("harvesting IMERG %s .. %s"
          % (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return harvest(start, end, keep_granules=args.keep_granules)


if __name__ == "__main__":
    raise SystemExit(main())
