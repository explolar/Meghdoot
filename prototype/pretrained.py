# -*- coding: utf-8 -*-
"""Transfer learning from a small public precipitation-nowcasting checkpoint.

This is deliberately *not* the foundation-model route the plan declines in
section 7.5. Aurora and Prithvi-WxC are disqualified on interface grounds - no
precipitation in pretraining, 6-hour timesteps, ERA5 latency, 0.25 degree grids
- and the measured result there is that a from-scratch U-Net beats every
adapter scheme on a 1.3B backbone.

SmaAt-UNet is a different proposition: 4.0M parameters, trained on
precipitation, published with weights under MIT, predicting rain fields from
rain fields. None of the four disqualifiers apply. The question it answers is
narrow: with only ~900 training windows, does a checkpoint that already knows
what rain looks like beat random initialisation?

The architecture below is reconstructed from the published state dict, and
`verify_architecture` checks it tensor-for-tensor against the file. That check
exists because reading a checkpoint by eye is error-prone: the depthwise layers
have weight shape (cin * kernels_per_layer, 1, k, k), so the second dimension is
always 1 and says nothing about the input width. The true input width comes
from the pointwise layer after it, divided by kernels_per_layer.

Interfaces, as read from the checkpoint:

    input    4 channels      matches ours (three past rain frames + live HEM)
    output   4 channels      we need 6 leads, so the head is replaced

Only the output head is not transferred. Its four outputs are the checkpoint's
own future steps at its own cadence, which have no meaningful correspondence to
our +30 ... +180 minute leads, so it trains from scratch. Everything upstream of
it is copied.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_DIR = r"E:\sih\data\pretrained"
SMAAT_URL = ("https://github.com/Rogue-Juan/MAD-SmaAt-GNet/raw/main/"
             "checkpoints/statedict_SmaAt_UNet.pt")
SMAAT_PATH = os.path.join(CKPT_DIR, "statedict_SmaAt_UNet.pt")

# the checkpoint's own interface
NATIVE_IN, NATIVE_OUT, KPL = 4, 4, 2


# --------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, cin, cout, kernels_per_layer=KPL, kernel_size=3,
                 padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(cin, cin * kernels_per_layer, kernel_size,
                                   padding=padding, groups=cin)
        self.pointwise = nn.Conv2d(cin * kernels_per_layer, cout, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DoubleConvDS(nn.Module):
    def __init__(self, cin, cout, mid=None, kernels_per_layer=KPL):
        super().__init__()
        mid = mid or cout
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(cin, mid, kernels_per_layer),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid, cout, kernels_per_layer),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class _Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class ChannelAttention(nn.Module):
    """Linear-MLP channel attention, as published (not the conv variant)."""

    def __init__(self, ch, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.MLP = nn.Sequential(
            _Flatten(),
            nn.Linear(ch, ch // reduction_ratio),
            nn.ReLU(),
            nn.Linear(ch // reduction_ratio, ch),
        )

    def forward(self, x):
        out = self.MLP(self.avg_pool(x)) + self.MLP(self.max_pool(x))
        return x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        out = self.bn(self.conv(torch.cat([avg, mx], dim=1)))
        return x * torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, ch, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(ch, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class DownDS(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                          DoubleConvDS(cin, cout))

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=True)
        self.conv = DoubleConvDS(cin, cout, mid=cin // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.conv(x)


class SmaAtUNet(nn.Module):
    """SmaAt-UNet with a residual output head.

    n_channels / n_classes default to our task (4 in, 6 leads). Pass 4 / 4 to
    get the checkpoint's native shape, which is how the architecture is
    verified against the published file.
    """

    def __init__(self, n_channels=4, n_classes=6, residual=True):
        super().__init__()
        self.residual = residual
        self.inc = DoubleConvDS(n_channels, 64)
        self.cbam1 = CBAM(64)
        self.down1 = DownDS(64, 128)
        self.cbam2 = CBAM(128)
        self.down2 = DownDS(128, 256)
        self.cbam3 = CBAM(256)
        self.down3 = DownDS(256, 512)
        self.cbam4 = CBAM(512)
        self.down4 = DownDS(512, 512)
        self.cbam5 = CBAM(512)
        self.up1 = UpDS(1024, 256)
        self.up2 = UpDS(512, 128)
        self.up3 = UpDS(256, 64)
        self.up4 = UpDS(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        last = x[:, 2:3]                        # newest rain frame

        x1 = self.inc(x)
        x1a = self.cbam1(x1)
        x2 = self.down1(x1)
        x2a = self.cbam2(x2)
        x3 = self.down2(x2)
        x3a = self.cbam3(x3)
        x4 = self.down3(x3)
        x4a = self.cbam4(x4)
        x5 = self.down4(x4)
        x5a = self.cbam5(x5)

        y = self.up1(x5a, x4a)
        y = self.up2(y, x3a)
        y = self.up3(y, x2a)
        y = self.up4(y, x1a)
        out = self.outc(y)

        # the same residual convention as the from-scratch U-Net, so the two
        # differ in backbone and initialisation and in nothing else
        return F.relu(last + out) if self.residual else F.relu(out)


# --------------------------------------------------------------------------
def download(url=SMAAT_URL, path=SMAAT_PATH, timeout=180):
    """Fetch the checkpoint if it is not already on disk."""
    import requests
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    os.replace(tmp, path)
    return path


def _read_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def verify_architecture(path=SMAAT_PATH):
    """True iff SmaAtUNet(4, 4) matches the checkpoint tensor-for-tensor.

    Returns (ok, missing, extra, bad_shapes) so a failure says what differs.
    """
    sd = _read_state(path)
    own = SmaAtUNet(NATIVE_IN, NATIVE_OUT).state_dict()
    missing = [k for k in sd if k not in own]
    extra = [k for k in own if k not in sd]
    bad = [k for k in sd if k in own and sd[k].shape != own[k].shape]
    return (not (missing or extra or bad)), missing, extra, bad


def load_pretrained(model, path=SMAAT_PATH, verbose=True):
    """Copy every shape-compatible tensor from the checkpoint into `model`.

    Returns (n_copied, n_scratch, scratch_names). A transfer experiment that
    does not state how much of the model actually came from the checkpoint is
    not interpretable, so the caller reports these.
    """
    sd = _read_state(path)
    own = model.state_dict()
    new_sd, copied, scratch = {}, 0, []
    for k, v in own.items():
        if k in sd and sd[k].shape == v.shape:
            new_sd[k] = sd[k]
            copied += 1
        else:
            new_sd[k] = v
            scratch.append(k)
    model.load_state_dict(new_sd)
    if verbose:
        n_par = sum(own[k].numel() for k in own if k not in scratch)
        n_all = sum(v.numel() for v in own.values())
        print("pretrained: %d/%d tensors copied (%.1f%% of parameters); "
              "from scratch: %s"
              % (copied, len(own), 100.0 * n_par / n_all,
                 ", ".join(sorted({s.rsplit('.', 1)[0] for s in scratch}))
                 or "none"))
    return copied, len(scratch), scratch


def freeze_encoder(model, freeze=True):
    """Freeze the encoder and attention blocks, leaving the decoder trainable.

    The T2 variant: the encoder keeps what the checkpoint learned about storm
    structure and only the decoder adapts to our product. Preferred when the
    archive is thin, because fine-tuning every weight on ~900 windows mostly
    overwrites the thing being transferred.
    """
    enc = ("inc", "down1", "down2", "down3", "down4",
           "cbam1", "cbam2", "cbam3", "cbam4", "cbam5")
    n = 0
    for name, p in model.named_parameters():
        if name.split(".")[0] in enc:
            p.requires_grad = not freeze
            n += 1
    return n


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
