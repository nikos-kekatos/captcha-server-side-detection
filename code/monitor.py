"""E5 (v2, after the 2026-10-20 red-team review): runtime monitor for automated
CAPTCHA solving over server-side telemetry.

A client key (account, session or IP prefix) emits events e_i = (t_i, L_i, c_i):
submission time, solve latency, correctness. With w the window, Delta the rate
window and n the observation horizon, the monitor checks

  P1 latency floor     : always  L_i >= tau1
  P2 median floor      : always  median(L_{i-w+1..i}) >= tau2            (i >= w)
  P3 regularity floor  : always  CV(L_{i-w+1..i}) >= c3                  (i >= w)
  P4 rate ceiling      : always  #{ j : t_i - t_j <= Delta } <= k4
  P5 accuracy ceiling  : at n    sum_{j<=n} c_j <= a5
  P6 dispersion ceiling: at n    SD(log L_{1..n}) <= s6

P1-P4 are evaluated after every event, P5-P6 when the key reaches n events.
Two implementations are provided: a vectorised batch evaluator (`violations`)
used for the experiments, and an event-by-event online monitor (`OnlineMonitor`)
with memory bounded by w + k4 + O(1) per key. `check_online` verifies that both
give the same first-violation index.

Thresholds are calibrated on human streams so that each property alone has a
false-positive rate of at most fpr/6 per stream of n challenges (union bound fpr).
"""
from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
SEARLES = ROOT / "data_ext" / "searles2023_text_times.csv"
PROPS = ["P1", "P2", "P3", "P4", "P5", "P6"]
W, DELTA = 5, 3600.0


def load_humans():
    rows = list(csv.DictReader(open(SEARLES)))
    return np.array([[float(r["simple_s"]), float(r["moving_s"]), float(r["masked_s"])] for r in rows])


def within_sigma(T):
    """Pooled within-participant SD of log time after removing the CAPTCHA-type effect (ddof=1)."""
    Lg = np.log(T)
    res = Lg - Lg.mean(0)
    return float(np.sqrt(np.mean(res.var(1, ddof=1))))


def beta_ab(mean, sd):
    k = mean * (1 - mean) / sd ** 2 - 1
    return mean * k, (1 - mean) * k


# ---------------------------------------------------------------- profiles
class Profile:
    """lat(rng, n) -> latencies; acc: float or callable(rng) -> per-key accuracy;
    gap: mean idle time (s) between challenges, or callable(rng, L) -> gaps."""

    def __init__(self, name, lat, acc, gap, group="bot"):
        self.name, self.lat, self.acc, self.gap, self.group = name, lat, acc, gap, group

    def stream(self, n, rng):
        L = self.lat(rng, n)
        acc = self.acc(rng) if callable(self.acc) else self.acc
        c = rng.random(n) < acc
        if callable(self.gap):
            g = self.gap(rng, L)
        else:
            g = rng.exponential(self.gap, n) if self.gap > 0 else np.zeros(n)
        t = np.cumsum(g + L)
        return t, L, c


def person_lat(T, sigma):
    """One participant per key: log-normal around the participant's log-mean."""
    mu = np.log(T).mean(1)

    def f(rng, n):
        return rng.lognormal(mu[rng.integers(len(mu))], sigma, n)
    return f


def pooled_lat(T):
    pool = T.ravel()
    return lambda rng, n: rng.choice(pool, n)


def human_acc(mean=0.84, sd=0.10):
    if sd <= 0:
        return mean
    a, b = beta_ab(mean, sd)
    return lambda rng: rng.beta(a, b)


# ---------------------------------------------------------------- batch evaluator
def _roll(x, w, fn):
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = fn(np.lib.stride_tricks.sliding_window_view(x, w))
    return out


def signals(t, L, c, n):
    med = _roll(L, W, lambda v: np.median(v, 1))
    cv = _roll(L, W, lambda v: v.std(1) / v.mean(1))
    cnt = np.arange(len(t)) - np.searchsorted(t, t - DELTA, side="left") + 1
    acc = c[:n].sum()
    sd = np.log(L[:n]).std(ddof=1)
    return med, cv, cnt, acc, sd


def violations(t, L, c, th, n):
    med, cv, cnt, acc, sd = signals(t, L, c, n)
    end = np.zeros(len(L), bool)
    with np.errstate(invalid="ignore"):
        v = {"P1": L < th["tau1"], "P2": med < th["tau2"], "P3": cv < th["c3"],
             "P4": cnt > th["k4"]}
    p5, p6 = end.copy(), end.copy()
    if len(L) >= n:
        p5[n - 1] = acc > th["a5"]
        p6[n - 1] = sd > th["s6"]
    v["P5"], v["P6"] = p5, p6
    return v


def first(v):
    idx = {k: (int(np.argmax(x)) if x.any() else None) for k, x in v.items()}
    hit = [i for i in idx.values() if i is not None]
    return (min(hit) if hit else None), idx


def calibrate(human, n, fpr, rng, n_streams=4000):
    q = fpr / len(PROPS)
    s = {k: [] for k in PROPS}
    for _ in range(n_streams):
        t, L, c = human.stream(n, rng)
        med, cv, cnt, acc, sd = signals(t, L, c, n)
        s["P1"].append(L.min())
        s["P2"].append(np.nanmin(med) if n >= W else np.inf)
        s["P3"].append(np.nanmin(cv) if n >= W else np.inf)
        s["P4"].append(cnt.max())
        s["P5"].append(acc)
        s["P6"].append(sd)
    # floors: lower q-quantile; ceilings: upper (1-q)-quantile on the observed values
    return dict(tau1=float(np.quantile(s["P1"], q, method="lower")),
                tau2=float(np.quantile(s["P2"], q, method="lower")),
                c3=float(np.quantile(s["P3"], q, method="lower")),
                k4=int(np.quantile(s["P4"], 1 - q, method="higher")),
                a5=int(np.quantile(s["P5"], 1 - q, method="higher")),
                s6=float(np.quantile(s["P6"], 1 - q, method="higher")), n=n)


def run_profile(p, th, n, rng, n_eval, keep=False):
    det, delay, fp = 0, [], {k: 0 for k in PROPS}
    streams = []
    for _ in range(n_eval):
        t, L, c = p.stream(n, rng)
        f, idx = first(violations(t, L, c, th, n))
        if keep:
            streams.append((t, L, c, f))
        if f is not None:
            det += 1
            delay.append(f + 1)
            for k, i in idx.items():
                if i == f:
                    fp[k] += 1
    out = dict(k=det, n=n_eval, rate=det / n_eval, wilson95=list(wilson(det, n_eval)),
               median_events_to_alarm=(float(np.median(delay)) if delay else None), first_property=fp)
    return (out, streams) if keep else out


# ---------------------------------------------------------------- online monitor
class OnlineMonitor:
    """Event-by-event monitor for one key. Memory: w latencies, at most k4+1 timestamps,
    one counter and three running sums (Welford for the log-latency SD)."""

    def __init__(self, th):
        self.th = th
        self.win = deque(maxlen=W)
        self.ts = deque()
        self.i = 0
        self.correct = 0
        self.m, self.m2 = 0.0, 0.0

    def step(self, t, L, c):
        th = self.th
        self.i += 1
        bad = L < th["tau1"]
        self.win.append(L)
        if len(self.win) == W:
            v = sorted(self.win)
            mean = sum(v) / W
            sd = math.sqrt(sum((x - mean) ** 2 for x in v) / W)
            bad |= v[W // 2] < th["tau2"] or sd / mean < th["c3"]
        self.ts.append(t)
        while self.ts and t - self.ts[0] > DELTA:
            self.ts.popleft()
        bad |= len(self.ts) > th["k4"]
        self.correct += int(c)
        x = math.log(L)
        d = x - self.m
        self.m += d / self.i
        self.m2 += d * (x - self.m)
        if self.i == th["n"]:
            sd = math.sqrt(self.m2 / (self.i - 1)) if self.i > 1 else 0.0
            bad |= self.correct > th["a5"] or sd > th["s6"]
        return bad


def online_first(t, L, c, th):
    mon = OnlineMonitor(th)
    for i in range(len(L)):
        if mon.step(float(t[i]), float(L[i]), bool(c[i])):
            return i
    return None


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
