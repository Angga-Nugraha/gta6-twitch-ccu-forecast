# GTA VI Twitch Viewer Forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Angga-Nugraha/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Angga-Nugraha/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

An entry for the [make.fun GTA VI Twitch viewer forecast](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) (gta6-twitch-ccu-forecast-01). On release day — 2026-11-19 UTC — GTA VI's first hour of Twitch viewership is the only thing known; this program turns that single number into a full-day forecast, one row per hour from hour 2 through hour 24, each row holding five quantiles (5%, 25%, 50%, 75%, 95%). Scoring is pinball loss on the log scale.

## The idea

A release day has a shape. Viewership usually sags overnight after the first hour, climbs again as regions wake up and unlock, and tops out in US prime time around hours 19–21. Simply carrying hour 1 forward misses that shape entirely — the house persistence baseline is typically off by ~83% on the practice cases.

So instead of extrapolating from one number, the forecaster borrows shapes from history. The case file carries dozens of comparable past launches, each with its own release-day curve. Every comparable is converted to log-ratios against its own hour 1, `r_h = ln(viewers_h / viewers_1)`, which strips out launch size and leaves pure shape. The forecast for GTA VI is then built from those shapes in two complementary ways:

- **Kernel kNN.** Each comparable is weighted by how close its hour-1 audience was to GTA VI's, using a Gaussian kernel (bandwidth 0.8 on log viewers). The weighted empirical quantiles of the log-ratios say what similarly sized launches did next.
- **Log-size regression.** Separately, each hour's log-ratio is regressed on log hour-1 size across all comparables. The fitted value at GTA VI's size is the centre; the residuals' quantiles supply the spread. This view borrows strength across the whole pool, including launches far in size from the target.

The two views are averaged in log space — 75% kNN, 25% regression — nudged up by a +0.10 log bias correction (leave-one-out testing showed the raw median under-predicts), and smoothed across neighbouring hours so no single hour jumps out of line. Quantiles are forced non-decreasing, and if a case ever ships with no usable comparable at all, the program degrades gracefully to the house persistence baseline.

Everything is a pure function of the case file: Python 3.13 standard library only, no network, no clock, no randomness. Run it twice and the bytes are identical.

## How well does it do?

Leave-one-out over the 37 comparable launches with a complete 24-hour release day — each launch held out in turn and forecast from the rest, scored with the contest's own pinball metric on the log scale (see `MODEL.md` for the full variant comparison):

| Subset | This model | House baseline (persistence) |
|---|---|---|
| GTA-like (comparable tier, hour-1 ≥ 20k, n=17) | **0.252** mean / 0.176 median | 0.347 / 0.252 |
| Hour-1 ≥ 50k (n=13) | **0.181** / 0.157 | 0.243 |
| All 37 full-curve comparables | **0.396** / 0.282 | 0.930 / 0.325 |

It beats the house baseline on 14 of the 17 GTA-like launches. The kit notes the house baseline scores ~0.387 on the four practice cases; the server only reveals the practice score once an entry is submitted, so this entry is submitted to read it and iterated from there if needed.

## Running it

```sh
python3 main.py
```

It reads every `*.in` file from `/input` (or `$MAKEFUN_INPUT_DIR`) and writes a matching `*.out` next to it in `/output` (or `$MAKEFUN_OUTPUT_DIR`), one JSON object per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Files

- `main.py` — the forecaster, self-contained.
- `MODEL.md` — the model notes: variant selection, decisions, and caveats.
