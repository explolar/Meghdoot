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
import concurrent.futures as cf
import datetime as dt
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imerg                                               # noqa: E402

ARCHIVE = r"E:\sih\data\archive"

# IMERG granules are ~8 MB, an order of magnitude smaller than INSAT's, so a
# wider pool is comfortable here.
WORKERS = int(os.environ.get("MEGHDOOT_IMERG_WORKERS", "4"))


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

        slots = [day + dt.timedelta(minutes=30 * s) for s in range(48)]
        slots = [t for t in slots if t.strftime("%H%M") not in store]
        total_have += 48 - len(slots)

        # Download in parallel, crop serially. Each worker gets its own
        # session: the Earthdata cookie is what makes a request fast, and a
        # session is not thread-safe to share.
        for i in range(0, len(slots), WORKERS):
            batch = slots[i:i + WORKERS]

            def _fetch(t_, _sessions={}):
                tid = threading.get_ident()
                if tid not in _sessions:
                    _sessions[tid] = imerg.warm_up(imerg._session())
                try:
                    return t_, imerg.download(t_, session=_sessions[tid]), None
                except Exception as exc:              # noqa: BLE001
                    return t_, None, exc

            with cf.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                for t_, path, err in pool.map(_fetch, batch):
                    key = t_.strftime("%H%M")
                    if err is not None:
                        if not isinstance(err, FileNotFoundError):
                            total_fail += 1
                            print("    %s  FAILED %s" % (key, str(err)[:70]))
                        continue
                    try:
                        field, _ = imerg.read_box(path)
                        store[key] = imerg.upsample_to(field, (128, 128))
                        total_new += 1
                        if field.max() > 1.0:
                            print("    %s  max %5.1f mm/h   %.2f%% >= 4"
                                  % (key, field.max(),
                                     100 * (field >= 4).mean()))
                    except Exception as exc:          # noqa: BLE001
                        total_fail += 1
                        print("    %s  CROP FAILED %s" % (key, str(exc)[:60]))
                    finally:
                        if not keep_granules and os.path.exists(path):
                            try:
                                os.remove(path)
                            except OSError:
                                pass

            if len(store) > before:
                save_day(day, store)

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
