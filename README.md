# GTA VI Twitch Viewer Forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Angga-Nugraha/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Angga-Nugraha/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

A forecaster for the [make.fun GTA VI Twitch viewer forecast](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) (gta6-twitch-ccu-forecast-01). Given Twitch viewers in the first hour of GTA VI's release day (2026-11-19 UTC), it forecasts viewers for each of hours 2–24 as five quantiles (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the log scale.

## Strategy

Comparable-launch quantile forecaster. A launch's release day has a shape: a night dip after hour 1, then a ramp as timezones unlock, peaking in US prime time (hours 19–21). Persistence from hour 1 misses that shape badly — the house baseline's typical miss on the practice cases is ~83%.

For every comparable past launch shipped in the case file, take its log-ratio curve `r_h = ln(viewers_h / viewers_1)`. Two views of the conditional distribution of `r_h` given the target's hour-1 size:
- **Kernel kNN**: Gaussian-kernel weighted empirical quantiles (bandwidth 0.8 on ln hour-1 viewers) — launches of similar size have similar shapes.
- **Log-size regression**: per hour, least-squares fit of `r_h` on ln(hour-1 viewers); centre at the fitted value, spread from residual quantiles.

Blend 0.75 kNN + 0.25 regression in log space, add a +0.10 log bias correction, smooth each quantile across neighbouring hours. Pure function of the case file: no clock, no network, no randomness — two runs write identical bytes.

## Local validation (leave-one-out, same pinball metric)

Scored on the 37 comparable launches with a full 24-hour release day, each held out in turn:

| Subset | This model | House baseline (persistence) |
|---|---|---|
| GTA-like (comparable tier, hour-1 ≥ 20k, n=17) | **0.252** mean / 0.176 median | 0.347 / 0.252 |
| Hour-1 ≥ 50k (n=13) | **0.181** / 0.157 | 0.243 |
| All 37 full-curve comparables | **0.396** / 0.282 | 0.930 / 0.325 |

Beats the house baseline on 14 of the 17 GTA-like launches. The kit states the house baseline scores ~0.387 on the four practice cases; the server shows the practice score only after an entry lands, so this entry is submitted to read it, then iterated if needed.

## Files

- `main.py` — the forecaster (Python 3.13, standard library only).
