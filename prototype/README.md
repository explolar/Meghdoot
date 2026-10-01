# MEGHDOOT prototype

Working code for **SIH26077** — satellite-only rainfall nowcasting over eastern
India. This is the pipeline behind the technical-approach slide, running on real
data from two live sources.

## What it does

```
INSAT-3R TIR1 (10.8 µm)  ──►  U-Net encoder–decoder  ──►  +30 … +180 min
GPM IMERG Early (label)       skip connections,            six direct heads
                              residual head
                                      │
                                      ▼
                        scored against four baselines,
                        every evaluation cycle
```

INSAT brightness temperature is the **input**; IMERG rain rate is the **label**.
That split is the design: INSAT sees cloud tops every 30 minutes over all of
India at 4 km but does not measure rain; IMERG measures rain from a
passive-microwave constellation but at ~11 km with latency. Training one onto
the other buys IMERG's physical grounding at INSAT's resolution and refresh.

## Files

| File | What it does |
|---|---|
| `insat.py` | read INSAT-3R L1C HDF5, crop to the Kolkata box, counts → brightness temperature |
| `imerg.py` | download and crop GPM IMERG Early, align to INSAT scan times |
| `harvest.py` | build an INSAT archive: download → crop → delete, so disk stays flat |
| `harvest_imerg.py` | the same for IMERG labels |
| `dataset.py` | pair the two products, normalise, split by day |
| `model.py` | the U-Net, the balanced loss, the residual head |
| `baselines.py` | persistence, climatology, Eulerian, optical flow |
| `metrics.py` | CSI, POD, FAR, HSS, bias, multi-scale FSS |
| `train.py` | train, then score the model against every baseline |
| `data.py` | synthetic fields, so the pipeline runs without network access |

## Running it

```bash
# no credentials needed - proves the pipeline end to end
python train.py --synthetic --epochs 20

# real data
python harvest_imerg.py --start 2026-09-22 --end 2026-09-26
python harvest.py --start 2026-09-22 --end 2026-09-26
python train.py --epochs 30
```

### Credentials

Nothing secret is committed. Two accounts are needed, both free:

- **MOSDAC** (INSAT): `mosdac_io.py` and a `config.yaml` holding the login live
  outside this repository. Point `MEGHDOOT_MOSDAC_DIR` at that directory.
- **NASA Earthdata** (IMERG): put the login in `~/.netrc` as
  `machine urs.earthdata.nasa.gov login USER password PASS`, and approve
  *NASA GESDISC DATA ARCHIVE* once under Earthdata → Authorized Apps.
  Without that approval every download returns 401.

## Design decisions the code implements

**No recursion.** Six output heads, one per lead time, in a single forward
pass. Feeding a prediction back in costs 77% of correlation by the final step
in the published comparison, because each pass consumes its own errors.

**Residual head.** The network predicts the *change* from the last observed
frame, so persistence is the floor it starts from rather than something it has
to rediscover.

**Balanced loss, not MSE.** Weights of 1 / 2 / 5 / 10 / 30 by rain rate in
mm/h. Rank correlation between MSE and CSI at 30 mm/h is −0.01, so optimising
squared error is uncorrelated with detecting heavy rain: on a field where heavy
rain is rare, the cheapest way to cut squared error is to predict near-zero
everywhere.

**IMERG Early, not Final.** Final propagates microwave observations *backward*
in time, so a Final frame can contain an overpass that happens after its own
timestamp. Training on it reports skill that cannot be reproduced at inference.

**Baselines every cycle.** An audit of 46 nowcasting papers found only two that
report both persistence and optical flow. Without a floor, a reported CSI is
uninterpretable.

**Day-based splits.** Random windows leak: the same storm lands in train and
test at different offsets, and the reported skill is partly memorisation.

## Honest limits

- `insat.tb_to_rain` is a published power-law fit to cloud-top temperature, not
  ISRO's Hydro-Estimator. The real HE adds precipitable-water-dependent
  coefficients, an orographic correction and a warm-cloud correction that
  cannot be reproduced from the L1C file. It is used only for diagnostics;
  IMERG is the training label.
- Nearest-neighbour is used to put IMERG on the INSAT grid. Bilinear would
  invent gradients between 11 km cells that the instrument never resolved.
- The archive built so far spans days, not years. Numbers from it are a
  demonstration that the pipeline works, not a measurement of the method.
