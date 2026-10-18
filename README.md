# Physics-Constrained Synthetic Solar PV Generation

Conditional VAEs that generate realistic **daily hourly PV profiles**, with a clear-sky physical envelope built directly into the decoder, so generated power is zero at night and can never exceed the physical limit. Built and evaluated end to end in Google Colab.


> **Scope in one line:** one site, one fixed-tilt 1 kWp PV system, daily 24-hour profiles. The model is *physics-constrained* (hard output limit from `pvlib`), not a PINN: it does not solve differential equations.

---

## Why this project

Synthetic energy data is useful when real data is scarce, private, or expensive to collect. A generator is only useful if its output is (1) statistically close to reality, (2) physically valid, (3) genuinely new rather than memorized, and (4) good enough to train downstream models. This project tests all four, and compares against two honest controls: a trivial seasonal bootstrap and post-hoc clipping.

## Key results

All numbers are on held-out real days (last 2 years of data). "Seeds" means 3 independently trained models.

| Question | Result |
|---|---|
| Does the unconstrained decoder respect physics? | **No.** It broke a physical limit on 100% of days. |
| Does the in-decoder envelope fix that? | **Yes**, 0% hard violations by construction. |
| Is it better than just clipping afterwards? | **Yes, modestly and consistently.** Wasserstein distance to real data **0.0022 ± 0.0002** (envelope) vs **0.0030 ± 0.0006** (unconstrained + clipping): about 27% lower, better in all 3 seeds. |
| Does it memorize? | **No.** Novelty about 0.08 vs 0.092 for real held-out days (bootstrap baseline: 0). |
| Is the synthetic data useful? | **Yes.** Afternoon forecaster trained on synthetic days: RMSE **74.3 W/kWp** vs **72.8** trained on real days (naive baseline: 80.9). |
| Can the decoder be small? | A 24k-parameter decoder (about 97 KB) matched the 81k-parameter one within seed variation (single seed). |
| Does a ramp-rate penalty help? | **No measurable effect** (see below). |
| Known weakness | Extreme hourly ramps are about **25% too small** (see Limitations). |

### Fidelity across 3 seeds (mean ± std)

| Model | Wasserstein, power values | Wasserstein, hourly ramps | 99th-pct hourly ramp (real: about 0.41) |
|---|---|---|---|
| A + post-hoc clipping | 0.0030 ± 0.0006 | 0.0050 ± 0.0006 | 0.294 ± 0.011 |
| **B: envelope in decoder** | **0.0022 ± 0.0002** | **0.0043 ± 0.0002** | 0.310 ± 0.006 |
| C: envelope + ramp penalty | 0.0023 ± 0.0002 | 0.0042 ± 0.0002 | 0.304 ± 0.009 |

B lower than clipped A in every seed (0.0024/0.0036, 0.0023/0.0031, 0.0020/0.0024). With three seeds this is consistent evidence, not a significance test. The ramp-distance gap is within A's seed spread and should be read as suggestive only.

### Full diagnostic table (seed 0, test set)

| | wd_value | wd_ramp | novelty | hard violations | ramp exceed | hourly-ramp quantiles (50/90/99/99.9%) |
|---|---|---|---|---|---|---|
| Real test days (reference) | 0 | 0 | 0.0922 | 0.0082 | 0.0004 | 0.116, 0.217, 0.412, 0.608 |
| Real train tail (noise floor) | 0.0015 | 0.0007 | n/a* | 0.0150 | 0.0001 | 0.117, 0.219, 0.421, 0.638 |
| A: unconstrained | 0.0041 | 0.0059 | 0.0860 | 1.0000 | 0.0000 | 0.119, 0.195, 0.312, 0.457 |
| A + post-hoc clip | 0.0036 | 0.0055 | 0.0849 | 0.0041 | 0.0000 | 0.119, 0.195, 0.307, 0.444 |
| B: + envelope | 0.0024 | 0.0044 | 0.0794 | 0.0000 | 0.0000 | 0.120, 0.197, 0.306, 0.473 |
| C: + envelope + ramp | 0.0025 | 0.0043 | 0.0786 | 0.0000 | 0.0000 | 0.118, 0.197, 0.295, 0.483 |

\*Novelty is 0 for the noise-floor row only because those days are part of the training set.
Real data itself breaks the hard rules on 0.8 to 1.5% of days (likely cloud-edge brightening above the envelope); the constrained models cannot produce that by design.

### Decoder-width sweep (variant C, single seed)

| Width | Decoder params | Size (float32) | Wasserstein | CPU ms / 1000 days |
|---|---|---|---|---|
| 8 | 576 | 2.3 KB | 0.0060 | 0.31 |
| 16 | 1,256 | 5.0 KB | 0.0044 | 0.39 |
| 32 | 3,000 | 12 KB | 0.0041 | 0.36 |
| 64 | 8,024 | 32 KB | 0.0033 | 0.57 |
| **128** | **24,216** | **97 KB** | **0.0023** | **1.19** |
| 256 | 81,176 | 325 KB | 0.0022 | 2.48 |

All variants sample 1000 days in under 3 ms on CPU, so the practical speed gain is small; the point is that the model compresses well.

### Likelihood sharpness sweep (variant B, single seed)

| sigma | wd_value | wd_ramp | novelty | ramp quantiles (50/90/99/99.9%) |
|---|---|---|---|---|
| **0.05 (used)** | 0.0024 | 0.0044 | 0.079 | 0.120, 0.197, 0.306, 0.473 |
| 0.03 | 0.0035 | 0.0044 | 0.089 | 0.119, 0.200, 0.296, 0.431 |
| 0.02 | 0.0039 | 0.0032 | 0.103 | 0.119, 0.210, 0.324, 0.444 |

A sharper likelihood improves moderate ramps (90th percentile reaches 0.210 vs 0.217 real) but not the extremes, and degrades the distribution of power values. Novelty above the real reference (0.103 vs 0.092) together with worse fidelity indicates samples drifting off the data manifold, not extra diversity. Sigma stayed at 0.05.

---

## Method

**Data.** Hourly PV output of a 1 kWp fixed-tilt system and ambient temperature from [PVGIS](https://joint-research-centre.ec.europa.eu/pvgis_en) (via `pvlib.iotools.get_pvgis_hourly`), 2005 to 2020. Timestamps are shifted to local solar time so each "day" is one sun cycle. The last 2 years (731 days) are held out for testing; 5,113 days are used for training. *(Adjust these figures if you change the site or years in Notebook 1.)*

**Physical envelope.** Clear-sky plane-of-array irradiance from `pvlib` (Ineichen model), scaled to PV watts with a constant `K` fitted on training data (99.5th percentile of output / clear-sky irradiance). The envelope is deterministic physics, not learned.

**Model.** Conditional VAE with MLP encoder and decoder (2 hidden layers of 128, 8-dim latent).
- Input / target: daily profile divided by peak capacity.
- Conditions: sin/cos of day of year, daily clearness index (actual energy / clear-sky energy), and the 24-hour envelope.
- Loss: Gaussian reconstruction (sigma = 0.05) + KL divergence (beta = 1 with a 30% warm-up).
- Optimizer: Adam, lr 1e-3, 300 epochs, batch 128.

**Variants (identical architecture; only the physics differs).**

| Variant | Decoder output | Extra loss |
|---|---|---|
| A: unconstrained | free linear output | none |
| B: + envelope | `1.1 x envelope x sigmoid(.)` | none |
| C: + envelope + ramp | same as B | penalty on hourly ramps above a limit (weight 200) |

**Controls.** (1) Seasonal block bootstrap: resample real training days within +/-15 days of the target date. It scores near-perfectly on fidelity and physics but has zero novelty, since it only replays history. (2) Post-hoc clipping: take the unconstrained model's output and clip to `[0, 1.1 x envelope]`, to test whether the constraint needs to live inside the model.

## Evaluation

| Metric | Definition |
|---|---|
| `wd_value` | Wasserstein-1 distance between pooled hourly power values (kW), synthetic vs real test |
| `wd_ramp` | Same, on hour-to-hour differences |
| `profile_rmse` | RMSE between the mean daily profiles |
| Hard violation rate | Share of days with any of: output above 1 W when the envelope is below 1 W; negative output; output above 1.1 x envelope |
| Ramp exceed rate | Share of daytime hour pairs whose ramp exceeds the 99.9th percentile of real training ramps |
| Novelty | Mean Euclidean distance from each synthetic day (1,500 sampled) to its nearest *training* day |
| TSTR | Train a forecaster on synthetic days, test on real days |

**Metric validation.** Notebook 2 contains negative controls (shuffled hours, night power added, output scaled by 1.5, white noise, flat profiles, near-copies of training days) with assertions that the right metric reacts to each corruption.

**TSTR task.** Predict hours 12 to 23 from hours 0 to 11, day of year, and the afternoon envelope, using an MLP regressor (64, 64). Compared against clear-sky persistence (scale the afternoon envelope by how sunny the morning was).

**Generation protocol.** Synthetic days are generated for each test day's conditions (day of year, clearness index, envelope) with a random latent, then compared with the real test set.

## Repository layout

```
notebooks/
  01_data_clearsky_baseline.ipynb   data, clear-sky envelope, metrics, bootstrap baseline
  02_cvae_variants.ipynb            metric sanity checks, A/B/C variants x 3 seeds
  03_tstr_efficiency_cv.ipynb       TSTR, decoder-width sweep, conditioning check
figures/                            real-vs-synthetic plots
results/                            results_nb2.csv and summary tables
```

Run the notebooks in order. Notebook 1 writes `solar_data.pkl`, Notebook 2 writes `results_nb2.csv`, `cvae_models.pt` and `synthetic_test_conditions.npz`, and Notebook 3 reads them.

## Reproducing

1. Open Notebook 1 in Colab and set the site (`LAT, LON`), panel tilt and years in the config cell. Any site covered by PVGIS works. If the download fails, set `RASTER_DB = "PVGIS-ERA5"`.
2. Run all cells. Notebooks 2 and 3 are fastest on a GPU runtime but also run on CPU.
3. Seeds are fixed (0, 1, 2) for the CVAE variants.

Dependencies: `pvlib`, `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`.

## Limitations

- **Extreme ramps are too gentle.** The 99th-percentile hourly ramp is about 0.30 to 0.31 in all models vs about 0.41 in real data (roughly 25% low); the 99.9th percentile is also low. Neither a ramp penalty nor a sharper likelihood fixed this, so it is likely a property of this model class (MLP VAE with a Gaussian latent).
- **The ramp penalty is inert.** Generated ramps never reach the limit, so the penalty never activates. Variant C is therefore equivalent to B.
- **The envelope slightly over-restricts.** About 1% of real days exceed it, which the constrained models cannot reproduce.
- **Evaluation conditions on the held-out day's own clearness index.** A user generating data would not know it in advance, so absolute numbers are somewhat optimistic for fully unconditional use.
- **Single site, single PV configuration, daily profiles.** No multi-site, multi-year continuity, demand, wind, or grid data.
- **TSTR has limited headroom.** A real-trained forecaster beats the naive baseline by only about 10%, so TSTR supports the generator but cannot separate good generators from great ones.
- **Efficiency and sigma sweeps use a single seed.**
- Three seeds give consistent evidence of an effect, not a formal significance test.

## Future work

- Generators better at tails: diffusion models or adversarial training with physical constraints.
- Multiple sites and climates; demand and wind data.
- Fully unconditional evaluation using clearness indices sampled from the training distribution.
- Physics-informed terms beyond the envelope, such as energy balance and temperature-dependent derating.

## Author

Man-W-Salwa Qazi

## License

The code in this repository is released under the MIT License. See LICENSE for details.
