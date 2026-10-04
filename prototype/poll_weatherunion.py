# -*- coding: utf-8 -*-
"""Poll Weather Union on a fixed interval and append to the archive.

Weather Union exposes current conditions only - there is no historical
endpoint - so the archive has to be accumulated by polling. Every hour this
does not run is an hour of ground truth that cannot be recovered later, which
is why the plan says to start archiving immediately rather than when the
verification work begins.

The interval matches the satellite cadence. INSAT-3R scans every 30 minutes,
and a gauge reading is only useful here if it can be bucketed against a scan,
so polling faster buys nothing for verification. It is set slightly under 30
minutes so a scan never falls in a gap.

Usage:
    python poll_weatherunion.py                  # poll until stopped
    python poll_weatherunion.py --once           # single snapshot
    python poll_weatherunion.py --stations 20    # explicit station count
"""
import argparse
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weatherunion as wu                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=1740,
                    help="seconds between polls (default 29 min)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--hours", type=float, default=0,
                    help="stop after this many hours; 0 means run forever")
    ap.add_argument("--stations", type=int, default=None,
                    help="how many stations to poll; default fits the budget")
    ap.add_argument("--all-stations", action="store_true",
                    help="poll every station in the box, ignoring the budget")
    args = ap.parse_args()

    try:
        stations = (wu.stations_in_box() if args.all_stations
                    else wu.verification_set(n=args.stations))
    except RuntimeError as e:
        print(e)
        return 1

    # The free tier allows 1000 calls a day and each poll costs one call per
    # station, so the two knobs trade against each other. Print the arithmetic
    # rather than leaving it implicit: an over-budget poller fails silently
    # partway through the day, which is the worst way to find out.
    per_day = len(stations) * (86400.0 / args.interval)
    print("polling %d stations every %d s -> %.0f calls/day (budget %d)"
          % (len(stations), args.interval, per_day, wu.DAILY_CALL_BUDGET),
          flush=True)
    if per_day > wu.DAILY_CALL_BUDGET:
        print("  WARNING: over budget. Reduce --stations or raise --interval.",
              flush=True)

    deadline = (time.time() + args.hours * 3600) if args.hours else None
    polls = 0
    while True:
        t0 = time.time()
        try:
            n, live, path = wu.snapshot(stations)
            polls += 1
            print("%s  %2d/%d reporting  (poll %d)"
                  % (dt.datetime.now().strftime("%d %b %H:%M"), live, n, polls),
                  flush=True)
        except Exception as e:                              # noqa: BLE001
            # a failed poll is a gap in the archive, not a reason to stop
            print("%s  poll failed: %s"
                  % (dt.datetime.now().strftime("%d %b %H:%M"), str(e)[:70]),
                  flush=True)

        if args.once:
            break
        if deadline and time.time() >= deadline:
            break
        # sleep the remainder of the interval, not the whole interval, so a
        # slow poll does not drift the schedule
        time.sleep(max(5.0, args.interval - (time.time() - t0)))

    print("%d polls recorded" % polls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
