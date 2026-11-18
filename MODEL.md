# Model notes — GTA VI Twitch viewer forecaster

## Approach

The program reads the case's own comparables at runtime (42 in the final,
38 in practice) and builds release-day log-ratio curves
`r_h = ln(viewers_h / viewers_1)`, h = 2..24, for every comparable with a
usable release day (hour 1 present, ≥ 12 hours). Two views of the
conditional distribution of `r_h` given the target's hour-1 size `v1`:

- **Kernel kNN** (weight 0.75): Gaussian-kernel weighted empirical
  quantiles of `r_h`, weights `exp(-0.5 * ((ln v1_comp - ln v1) / 0.8)^2)`.
  Launches of similar hour-1 size have similar day shapes.
- **Log-size regression** (weight 0.25): per hour, least-squares fit of
  `r_h` on `ln(v1_comp)`; centre at the fitted value for the target's size,
  spread from the residuals' empirical quantiles.

Quantiles are blended in log space, shifted by a +0.10 log bias correction,
smoothed across neighbouring hours (window ±1), and enforced
non-decreasing. With no usable comparable the program falls back to the
house persistence baseline. Deterministic, stdlib only, no network, no
clock, no randomness; ~0.1 s per case.

## Variant selection (leave-one-out, 37 full-curve comparables)

Mean / median pinball loss on the log scale. GTA-like subset = comparable
tier + hour-1 viewers ≥ 20k (n=17); ALL = all 37.

| Variant | GTA-like mean / median | ALL mean / median |
|---|---|---|
| House baseline (persistence starter) | 0.3473 / 0.2516 | 0.9297 / 0.3248 |
| kNN hard k=8 | 0.2871 / 0.1720 | 0.4181 / 0.2833 |
| kNN hard k=10 | 0.2718 / 0.1602 | 0.4359 / 0.2600 |
| kNN hard k=12 | 0.2753 / 0.1590 | 0.4502 / 0.2645 |
| kNN hard k=16 | 0.2796 / 0.1691 | 0.4896 / 0.2613 |
| kNN hard k=20 | 0.2880 / 0.1903 | 0.5188 / 0.2618 |
| kNN kernel bw=0.8, smoothed | 0.2730 / 0.1573 | 0.4093 / 0.2529 |
| kNN-12 restricted to target's unlock class | 0.4236 / 0.3789 | 0.6867 / 0.4047 |
| Log-size regression (all comparables) | 0.3311 / 0.2744 | 0.4377 / 0.3386 |
| **Hybrid blend 0.75 kNN + 0.25 regression, shift +0.10 (SHIPPED)** | **0.2515 / 0.1755** | **0.3962 / 0.2824** |
| Hybrid, blend 0.60 / shift +0.10 | 0.2563 / 0.1939 | 0.3953 / 0.2840 |
| Hybrid, blend 0.85 / shift +0.10 | 0.2535 / 0.1699 | 0.4000 / 0.2831 |
| Hybrid, shift +0.00 | 0.2614 / 0.1707 | 0.3958 / 0.2451 |
| Hybrid, shift +0.15 | 0.2509 / 0.1819 | 0.3994 / 0.3034 |

Decisions:

- **Unlock-class restriction was rejected**: it scores far worse
  (0.424 vs 0.275 for plain kNN-12 on the GTA-like subset). The rolling
  pool is only n=8 and the class labels are noisy (many unlock texts are
  "unverified"); size similarity carries more signal than the class label.
  GTA VI itself unlocks rolling midnight local, but the unrestricted
  size-weighted pool beats the rolling-only pool.
- **Shift +0.10 kept over +0.15**: +0.15 wins the GTA-like mean by 0.0006
  but loses median (0.182 vs 0.176) and ALL (0.399 vs 0.396) — noise, not
  a real gain.
- **Blend 0.75 kept over 0.60/0.85**: best GTA-like mean; 0.60 wins ALL
  mean by 0.0007, again noise.
- The shipped hybrid beats the baseline on 14 of the 17 GTA-like
  launches, with 90% coverage 0.887 (GTA-like) / 0.847 (ALL).
  On the hour-1 ≥ 50k subset (n=13): 0.1813 mean / 0.1567 median vs
  baseline 0.2426.

## Caveats

- LOO is a proxy: the final case is one launch (GTA VI, console-only,
  possibly a larger hour-1 than any comparable). The kernel falls back
  gracefully — weights decay with log-size distance, so an out-of-range
  hour-1 is effectively forecast from the largest comparables.
- The kit states the house baseline scores ~0.387 on the four practice
  cases; the server grades the practice score only after submission.
