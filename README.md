# GTA VI Twitch CCU Forecaster

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), from the viewer count recorded up
to the end of the day's first hour.

Entry for make.fun bounty `gta6-twitch-ccu-forecast-01` — program/v1 harness,
Python 3.13 stdlib only, offline, deterministic.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out`:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2–24.

## Model — "kernel-blend"

Past big launches' release-day curves (shipped in each case as
`comparables`) are the training set. For each comparable, take the
log-ratio of each hour's viewers to its hour-1 print. Weight comparables
by how close their hour-1 print is to this case's (Gaussian kernel in log
space, bandwidth 1.0 nat): a game whose hour 1 was already huge stays
near flat, a game whose hour 1 was tiny can grow a hundredfold, and the
kernel is what separates those regimes. Per hour, blend two estimates of
the five quantiles of the log-ratio — a normal estimate from the
kernel-weighted mean/sd and the kernel-weighted empirical quantiles —
smooth across neighbouring hours, and forecast
`q = hour1_print * exp(quantile)`.

Leave-one-out over the 37 usable comparables in the kit: mean pinball
loss 0.393 (0.207 on launches with hour-1 ≥ 50k, the regime GTA VI will
be in) vs the house persistence baseline's 0.930 (0.251). Full method and
backtests are documented in the header of `main.py`.

## Files

- `main.py` — the entry (the only code file).
- `LICENSE` — MIT.
- `.makefun/owner.json` — payout wallet.
