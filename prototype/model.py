# -*- coding: utf-8 -*-
"""The U-Net described on the technical-approach slide, in code.

Encoder-decoder with skip connections, a residual head, and one output head per
lead time. Every choice here is one the deck claims, so the code is the check on
the claim:

  * skip connections       full-resolution detail bypasses the bottleneck
  * residual head          the net predicts the *change* from the last frame,
                           so persistence is the floor it starts from
  * direct multi-lead      six heads, one forward pass, no recursion
  * balanced loss          weights by rain rate, because MSE and CSI at
                           30 mm/h have a rank correlation of -0.01

Input is three rain frames plus the INSAT 10.8 um brightness temperature, which
is the "T-2, T-1, T + IR" the architecture panel shows.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Conv 3x3 -> BN -> ReLU, twice. The blue bars in the architecture figure."""

    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetNowcast(nn.Module):
    """U-Net with a residual head and one output head per lead time.

    Parameters
    ----------
    in_ch   : input channels (3 rain frames + 1 brightness temperature = 4)
    leads   : how many lead times to predict, each from its own head
    base    : channel width at full resolution; doubles at each downsample
    """

    def __init__(self, in_ch=4, leads=6, base=32, dropout=0.5,
                 last_idx=2, base_space="mm"):
        super().__init__()
        self.leads = leads
        # which input channel is the newest frame, and which space the residual
        # base is added in. Inputs are log1p-normalised but the output is
        # scored in mm/h, so the base must be converted back with expm1 or the
        # "persistence floor" is really log1p(rain): at 30 mm/h it starts at
        # 3.4. base_space="log" reproduces that earlier behaviour for ablation.
        self.last_idx = last_idx
        self.base_space = base_space

        self.enc1 = ConvBlock(in_ch, base)            # 128 -> 128
        self.enc2 = ConvBlock(base, base * 2)         #  64
        self.enc3 = ConvBlock(base * 2, base * 4)     #  32
        self.enc4 = ConvBlock(base * 4, base * 8)     #  16
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)

        self.bridge = ConvBlock(base * 8, base * 16)  #   8

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        # one 1x1 head per lead time: no recursion, so errors cannot compound
        self.heads = nn.ModuleList(
            [nn.Conv2d(base, 1, 1) for _ in range(leads)])

    def forward(self, x):
        """x: (B, C, H, W) with channel -1 the brightness temperature.

        Returns (B, leads, H, W) of rain rate in mm/h.
        """
        # the most recent rain frame, which the residual head adds back
        last = x[:, self.last_idx:self.last_idx + 1]
        if self.base_space == "mm":
            last = torch.expm1(last)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.drop(self.enc4(self.pool(e3)))
        b = self.drop(self.bridge(self.pool(e4)))

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        # each head predicts a *delta*; adding the last frame makes persistence
        # the zero-effort solution, so the network only has to learn the change
        outs = [F.relu(last + head(d1)) for head in self.heads]
        return torch.cat(outs, dim=1)


# --------------------------------------------------------------------------
# Weight schemes, in mm/h so they port between products.
#
# TrajGRU's published weights (1/2/5/10/30) were derived from Hong Kong radar.
# Measured on this archive they are far too weak for the tail: the 10-30 band
# holds 0.51% of pixels and the >30 band 0.03%, so pure inverse frequency would
# ask for 185x and 3166x against the 1x baseline - the published scheme
# under-weights the heaviest band by about 106x.
#
# The first run showed exactly the failure that predicts: CSI 0.000 at 20 mm/h
# against persistence at 0.089. The model paid almost nothing for missing the
# events the system exists to warn about.
#
# Pure inverse frequency is not the fix either - a 3166x weight lets a handful
# of pixels dominate every gradient and training destabilises. These schemes
# step between the two so the trade can be measured rather than guessed.
WEIGHT_SCHEMES = {
    "trajgru": (1.0, 2.0, 5.0, 10.0, 30.0),        # published, the baseline
    "moderate": (1.0, 5.0, 20.0, 60.0, 150.0),     # ~5x the published tail
    "strong": (1.0, 10.0, 40.0, 150.0, 500.0),     # ~17x
    "sqrt_inv": (1.0, 4.7, 8.6, 13.6, 56.3),       # sqrt of inverse frequency
    "inv_freq": (1.0, 22.4, 73.2, 185.3, 3166.3),  # pure inverse frequency
}


def balanced_weights(y, bins=(2.0, 5.0, 10.0, 30.0),
                     weights=None, scheme="trajgru"):
    """Per-pixel weights by rain rate, in mm/h so they port between products.

    Bands are < 2, 2-5, 5-10, 10-30 and >= 30 mm/h. `scheme` picks a row from
    WEIGHT_SCHEMES; `weights` overrides it directly.
    """
    if weights is None:
        weights = WEIGHT_SCHEMES[scheme]
    w = torch.full_like(y, weights[0])
    for edge, wt in zip(bins, weights[1:]):
        w = torch.where(y >= edge, torch.full_like(y, wt), w)
    return w


def balanced_loss(pred, target, under_penalty=1.3, scheme="trajgru"):
    """Weighted MSE + MAE, with an extra penalty for under-forecasting.

    The asymmetry matters for a warning system: predicting 5 mm/h when 20 fell
    is a missed warning, while predicting 20 when 5 fell is a false alarm. They
    are not equally costly, so they do not get equal gradient.
    """
    w = balanced_weights(target, scheme=scheme)
    err = pred - target
    under = (err < 0).float() * (under_penalty - 1.0) + 1.0
    w = w * under
    return (w * (err ** 2)).mean() + (w * err.abs()).mean()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
