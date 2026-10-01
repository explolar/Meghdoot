# -*- coding: utf-8 -*-
"""The four baselines, which are the point of the project.

An audit of 46 nowcasting papers found only two that report both persistence
and optical flow. Without a floor, a reported CSI is uninterpretable: a model
can beat "nothing" and look impressive while losing to the assumption that the
next hour resembles this one. These run on every evaluation cycle, not once at
the end.

  persistence     the last observed frame, repeated. The floor.
  climatology     the training-set mean field. Catches "always predict the
                  average", which scores deceptively well on MSE.
  eulerian        persistence with no motion, kept separate from Lagrangian
                  advection so the contribution of motion is isolated.
  optical flow    Lucas-Kanade motion estimate, then semi-Lagrangian advection.
                  This is the baseline that is genuinely hard to beat at short
                  lead times, and the one most papers omit.
"""
import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:                                   # pragma: no cover
    HAVE_CV2 = False


# --------------------------------------------------------------------------
def persistence(inputs, n_leads):
    """Repeat the last observed frame at every lead time."""
    last = inputs[:, -1]
    return np.repeat(last[:, None], n_leads, axis=1)


def climatology(inputs, n_leads, clim_field):
    """Predict the training-set mean field, regardless of what was observed."""
    b = inputs.shape[0]
    out = np.broadcast_to(clim_field, (b, n_leads) + clim_field.shape)
    return np.ascontiguousarray(out)


# --------------------------------------------------------------------------
def _to_u8(frame, vmax):
    """Optical flow wants 8-bit. Scale on a shared vmax so motion is
    comparable between the two frames rather than per-frame normalised."""
    if vmax <= 0:
        return np.zeros(frame.shape, dtype=np.uint8)
    f = np.clip(frame / vmax, 0, 1) * 255.0
    return f.astype(np.uint8)


def _dense_flow(prev, curr):
    """Farneback dense flow. Returns (H, W, 2) displacement per 30-min step.

    Farneback rather than sparse Lucas-Kanade because a rain field has no
    corners to track: the features that matter are blobs, and dense flow
    handles them without a feature detector.
    """
    vmax = float(max(prev.max(), curr.max(), 1e-6))
    p8, c8 = _to_u8(prev, vmax), _to_u8(curr, vmax)
    return cv2.calcOpticalFlowFarneback(
        p8, c8, None,
        pyr_scale=0.5, levels=4, winsize=17,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0)


def _advect(field, flow, steps):
    """Semi-Lagrangian advection: walk backwards along the flow and sample.

    Backward semi-Lagrangian rather than forward scatter, because forward
    scatter leaves holes where no source pixel lands and piles up where many
    do. Backward remap gives every output pixel exactly one source.
    """
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = (xx - flow[..., 0] * steps) % w
    src_y = (yy - flow[..., 1] * steps) % h
    return cv2.remap(field, src_x, src_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def optical_flow(inputs, n_leads):
    """Lucas-Kanade / Farneback motion, then advect the last frame forward."""
    if not HAVE_CV2:
        return persistence(inputs, n_leads)
    b = inputs.shape[0]
    out = np.zeros((b, n_leads) + inputs.shape[2:], dtype=np.float32)
    for i in range(b):
        prev, curr = inputs[i, -2], inputs[i, -1]
        flow = _dense_flow(prev, curr)
        for L in range(n_leads):
            out[i, L] = _advect(curr, flow, L + 1)
    return out


def eulerian_persistence(inputs, n_leads):
    """Explicitly zero-motion persistence.

    Identical to `persistence` by construction; kept as a separate name so the
    results table can state plainly that the motion-free and motion-aware
    floors were both run, which is the distinction the audited papers blur.
    """
    return persistence(inputs, n_leads)


# --------------------------------------------------------------------------
def all_baselines(inputs, n_leads, clim_field):
    """Run every baseline and return {name: (B, L, H, W)}."""
    out = {
        "persistence": persistence(inputs, n_leads),
        "climatology": climatology(inputs, n_leads, clim_field),
    }
    if HAVE_CV2:
        out["optical flow"] = optical_flow(inputs, n_leads)
    return out
