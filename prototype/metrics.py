# -*- coding: utf-8 -*-
"""Verification metrics: CSI, POD, FAR, HSS, bias, and multi-scale FSS.

Why not accuracy. Rain is rare. On a domain where 8% of cells rain, forecasting
"no rain" everywhere scores 92% accuracy with zero skill. CSI excludes correct
negatives, so the same forecast correctly scores 0.00.

Why FSS as well as CSI. A forecast can be right about the storm and wrong about
the exact pixel, which CSI punishes as both a miss and a false alarm - the
"double penalty". FSS pools over a neighbourhood before comparing, so it
reports the *scale* at which the forecast has skill. That is what turns
"4 km output" into a claim that can be checked rather than asserted.
"""
import numpy as np


# --------------------------------------------------------------------------
def contingency(pred, obs, threshold):
    """Hits, misses, false alarms and correct negatives at one threshold."""
    p = pred >= threshold
    o = obs >= threshold
    return (int(np.sum(p & o)), int(np.sum(~p & o)),
            int(np.sum(p & ~o)), int(np.sum(~p & ~o)))


def scores(pred, obs, threshold):
    """Every categorical score at one threshold, from one contingency table."""
    h, m, f, c = contingency(pred, obs, threshold)
    tot = h + m + f + c
    out = {
        "hits": h, "misses": m, "false_alarms": f, "correct_neg": c,
        "events": h + m,
        "CSI": h / (h + m + f) if (h + m + f) else np.nan,
        "POD": h / (h + m) if (h + m) else np.nan,
        "FAR": f / (h + f) if (h + f) else np.nan,
        "BIAS": (h + f) / (h + m) if (h + m) else np.nan,
    }
    # Heidke skill score: skill against random chance, not against zero
    exp = ((h + m) * (h + f) + (c + m) * (c + f)) / tot if tot else 0.0
    out["HSS"] = ((h + c - exp) / (tot - exp)) if (tot - exp) else np.nan
    return out


# --------------------------------------------------------------------------
def _pool(field, n):
    """Fractional coverage in an n x n neighbourhood, via a summed-area table.

    An integral image makes this O(1) per pixel regardless of n, which matters
    because FSS is evaluated at several window sizes on every batch.
    """
    if n <= 1:
        return field.astype(np.float32)
    pad = n // 2
    p = np.pad(field.astype(np.float32), pad, mode="constant")
    ii = p.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
    h, w = field.shape
    out = (ii[n:n + h, n:n + w] - ii[0:h, n:n + w]
           - ii[n:n + h, 0:w] + ii[0:h, 0:w])
    return out / float(n * n)


def fss(pred, obs, threshold, window):
    """Fractions Skill Score at one threshold and one neighbourhood size.

    1.0 is perfect, 0.0 is no skill. The usual "useful" line is 0.5.
    """
    pf = _pool(pred >= threshold, window)
    of = _pool(obs >= threshold, window)
    num = np.mean((pf - of) ** 2)
    den = np.mean(pf ** 2) + np.mean(of ** 2)
    if den == 0:
        return np.nan            # neither forecast nor observation had events
    return 1.0 - num / den


def fss_multiscale(pred, obs, threshold, windows=(1, 3, 5, 9, 16),
                   dx_km=4.0):
    """FSS across neighbourhood sizes, reported in kilometres.

    windows are in grid cells; at 4 km these are 4, 12, 20, 36 and 64 km, which
    are the scales the evaluation plan names.
    """
    out = {}
    for w in windows:
        out[int(round(w * dx_km))] = fss(pred, obs, threshold, w)
    return out


# --------------------------------------------------------------------------
def evaluate(pred, obs, thresholds=(1, 4, 8, 16, 20), leads=None,
             dx_km=4.0, fss_windows=(1, 3, 5, 16)):
    """Score a full (N, L, H, W) forecast against observations.

    Returns {lead_index: {threshold: {metric: value}}}, with FSS folded in.
    """
    assert pred.shape == obs.shape, (pred.shape, obs.shape)
    n_lead = pred.shape[1]
    leads = leads or list(range(n_lead))
    results = {}

    for li in range(n_lead):
        p = pred[:, li]
        o = obs[:, li]
        per_thresh = {}
        for t in thresholds:
            s = scores(p, o, t)
            # FSS is a spatial score, so it is computed per sample and averaged
            # rather than on the flattened stack
            fvals = {}
            for w in fss_windows:
                vals = [fss(p[i], o[i], t, w) for i in range(p.shape[0])]
                vals = [v for v in vals if np.isfinite(v)]
                fvals[int(round(w * dx_km))] = float(np.mean(vals)) if vals else np.nan
            s["FSS"] = fvals
            per_thresh[t] = s
        results[leads[li]] = per_thresh
    return results


def summary_table(results, thresholds=(1, 4, 8, 16, 20), metric="CSI",
                  lead_minutes=30):
    """A printable table of one metric, leads down the side."""
    lines = []
    head = "lead    " + "".join("%9s" % ("%g mm/h" % t) for t in thresholds)
    lines.append(head)
    lines.append("-" * len(head))
    for li in sorted(results):
        row = "%4d min" % ((li + 1) * lead_minutes)
        for t in thresholds:
            v = results[li][t].get(metric, np.nan)
            row += "%9s" % ("%.3f" % v if np.isfinite(v) else "-")
        lines.append(row)
    return "\n".join(lines)


def event_counts(obs, thresholds=(1, 4, 8, 16, 20)):
    """How many pixels exceed each threshold. A score over a handful of
    events is noise, so this belongs beside every results table."""
    return {t: int(np.sum(obs >= t)) for t in thresholds}
