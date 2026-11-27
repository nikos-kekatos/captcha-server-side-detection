"""E5 v2 driver. Writes ../results/e5.json.

Measured inputs: E1 (CRNN accuracy on L2, inference time), E2 (accuracy after
fine-tuning on 1,000 G1 labels), E4 (VLM accuracy and latency on L2).
Published inputs: Searles et al. 2023 raw human text times; Motoyama et al.
2010 farm latency (log-normal, median 14 s, Q3 20 s).
Modelled (varied in the misspecification analysis): human accuracy (Beta,
mean 0.84, SD 0.10 across keys), human gap (exponential, mean 300 s),
automation overhead U(0.3, 1.5) s.
"""
from __future__ import annotations

import glob
import json
import math
import time
from pathlib import Path

import numpy as np

import monitor as M

R = M.R
HZ = [10, 20, 30, 50, 100]
FPR = 0.01
N_SPLITS = 20


def measured():
    e1 = json.loads((R / "e1.json").read_text())
    l2 = [r["test"]["L2"] for r in e1 if r["train_level"] == 2]
    crnn_acc = float(np.mean([x["acc"] for x in l2]))
    crnn_ms = float(np.mean([x["ms_per_img"] for x in l2]))
    e2acc = None
    if (R / "e2.json").exists():
        v = [r["finetune"]["acc"] for r in json.loads((R / "e2.json").read_text())
             if r["src"] == "G2mix" and r["k"] == 1000]
        e2acc = float(np.mean(v)) if v else None
    vlm = {}
    for f in sorted(glob.glob(str(R / "e4_*.json"))):
        rows = [r for r in json.loads(Path(f).read_text()) if r["cond"] == "L2"]
        if rows:
            vlm[Path(f).stem[3:]] = dict(acc=float(np.mean([r["ok"] for r in rows])),
                                         lat=[r["latency_s"] for r in rows], n=len(rows))
    return crnn_acc, crnn_ms, e2acc, vlm


def humans(T, sigma, acc_mean=0.84, acc_sd=0.10, gap=300.0, mode="person", name="human"):
    lat = M.person_lat(T, sigma) if mode == "person" else M.pooled_lat(T)
    return M.Profile(name, lat, M.human_acc(acc_mean, acc_sd), gap, "human")


def bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th, gap_h=300.0):
    ov = lambda r, n: r.uniform(0.3, 1.5, n) + crnn_ms / 1000
    hacc = M.human_acc()
    farm_mu, farm_sg = math.log(14.0), math.log(20.0 / 14.0) / 0.6744897501960817
    period = M.DELTA / th["k4"] * 1.001          # just under k4 events per Delta
    paced_gap = lambda r, L: np.maximum(period - L, 0.0)
    P = [M.Profile("CRNN, direct", ov, crnn_acc, 1.0)]
    for name, v in sorted(vlm.items()):
        lat = np.array(v["lat"])
        P.append(M.Profile(f"VLM {name}, direct",
                           lambda r, n, lat=lat: r.choice(lat, n) + r.uniform(0.3, 1.5, n), v["acc"], 1.0))
    P += [
        M.Profile("Fixed delay 6-12 s", lambda r, n: ov(r, n) + r.uniform(6, 12, n), crnn_acc, 1.0),
        M.Profile("Fixed delay, human gap", lambda r, n: ov(r, n) + r.uniform(6, 12, n), crnn_acc, gap_h),
        M.Profile("Farm relay", lambda r, n: r.lognormal(farm_mu, farm_sg, n), hacc, 1.0),
        M.Profile("Farm relay, human gap", lambda r, n: r.lognormal(farm_mu, farm_sg, n), hacc, gap_h),
        M.Profile("Pooled mimic, full accuracy", M.pooled_lat(Tcal), crnn_acc, gap_h),
        M.Profile("Pooled mimic, human accuracy", M.pooled_lat(Tcal), hacc, gap_h),
        M.Profile("Person mimic, full accuracy", M.person_lat(Tcal, sigma), crnn_acc, gap_h),
        M.Profile("Person mimic, human accuracy", M.person_lat(Tcal, sigma), hacc, gap_h),
        M.Profile("Rate-paced mimic, human accuracy", M.person_lat(Tcal, sigma), hacc, paced_gap),
    ]
    if e2acc is not None:
        P.append(M.Profile("Person mimic, E2 solver accuracy", M.person_lat(Tcal, sigma), e2acc, gap_h))
    return P


def rate_limit_only(human, n, fpr, rng, n_streams=4000):
    """Baseline (red-team 2026-10-21, change 1): P4 alone, calibrated to the whole false-positive
    budget; the other properties are disabled."""
    s = []
    for _ in range(n_streams):
        t, L, c = human.stream(n, rng)
        s.append(M.signals(t, L, c, n)[2].max())
    k4 = int(np.quantile(s, 1 - fpr, method="higher"))
    return dict(tau1=-np.inf, tau2=-np.inf, c3=-np.inf, k4=k4, a5=n, s6=np.inf, n=n)


def throughput(p, rng, keys=200, n=500):
    """Mean correct solves per key-hour over many keys (per-key accuracy draws differ)."""
    v = []
    for _ in range(keys):
        t, L, c = p.stream(n, rng)
        v.append(3600 * c.sum() / t[-1])
    return float(np.mean(v))


def main():
    T = M.load_humans()
    sigma = M.within_sigma(T)          # replaced per split below; kept for the inputs record
    crnn_acc, crnn_ms, e2acc, vlm = measured()
    out = dict(inputs=dict(sigma=sigma, crnn_acc_L2=crnn_acc, crnn_ms=crnn_ms, e2_acc_k1000=e2acc,
                           vlm={k: dict(acc=v["acc"], n=v["n"], median_lat=float(np.median(v["lat"])))
                                for k, v in vlm.items()},
                           n_participants=len(T), splits=N_SPLITS, horizons=HZ, fpr=FPR),
               splits=[], misspec=[], online_check=None, monitor_rate=None, throughput={})
    t0 = time.time()
    for sp in range(N_SPLITS):
        perm = np.random.default_rng(1000 + sp).permutation(len(T))
        Tcal, Ttest = T[perm[: len(T) // 2]], T[perm[len(T) // 2:]]
        # red-team 2026-10-21 item 8: estimate the spread on the calibration half only, and generate
        # held-out people with the spread estimated on the held-out half
        sigma, sigma_test = M.within_sigma(Tcal), M.within_sigma(Ttest)
        n_eval = 2000 if sp == 0 else 1000
        rec = dict(split=sp, sigma_cal=sigma, sigma_test=sigma_test, horizons={})
        for n in HZ:
            th = M.calibrate(humans(Tcal, sigma), n, FPR, np.random.default_rng(10 * sp + 1))
            res = {"Human, held-out": M.run_profile(humans(Ttest, sigma_test), th, n,
                                                    np.random.default_rng(10 * sp + 2), n_eval)}
            for p in bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
                res[p.name] = M.run_profile(p, th, n, np.random.default_rng(10 * sp + 3), n_eval)
            th_rl = rate_limit_only(humans(Tcal, sigma), n, FPR, np.random.default_rng(10 * sp + 4))
            res_rl = {"Human, held-out": M.run_profile(humans(Ttest, sigma_test), th_rl, n,
                                                       np.random.default_rng(10 * sp + 2), n_eval)}
            for p in bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
                res_rl[p.name] = M.run_profile(p, th_rl, n, np.random.default_rng(10 * sp + 3), n_eval)
            rec["horizons"][str(n)] = dict(thresholds=th, results=res, rate_limit=dict(thresholds=th_rl, results=res_rl))
            if sp == 0 and n == 50:
                # the online monitor must reproduce the batch verdicts on every stream
                mism, total = 0, 0
                for p in [humans(Ttest, sigma_test)] + bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
                    _, streams = M.run_profile(p, th, n, np.random.default_rng(99), 300, keep=True)
                    for (t, L, c, f) in streams:
                        total += 1
                        mism += int(M.online_first(t, L, c, th) != f)
                out["online_check"] = dict(streams=total, mismatches=mism)
                hp = humans(Ttest, sigma_test)
                ev = [hp.stream(n, np.random.default_rng(5 + i)) for i in range(400)]
                tt = time.time()
                for (t, L, c) in ev:
                    mon = M.OnlineMonitor(th)
                    for i in range(n):
                        mon.step(float(t[i]), float(L[i]), bool(c[i]))
                out["monitor_rate"] = dict(events=400 * n, events_per_s=400 * n / (time.time() - tt))
                rng = np.random.default_rng(3)
                for p in [humans(Ttest, sigma_test)] + bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
                    out["throughput"][p.name] = float(throughput(p, rng))
                out["throughput"]["k4_at_50"] = th["k4"]
        out["splits"].append(rec)
        hum = {n: rec["horizons"][str(n)]["results"]["Human, held-out"]["rate"] for n in HZ}
        print(f"split {sp} {time.time()-t0:.0f}s human FPR", {k: round(v, 4) for k, v in hum.items()}, flush=True)
        (R / "e5.json").write_text(json.dumps(out, indent=1))
    perm = np.random.default_rng(1000).permutation(len(T))
    Tcal, Ttest = T[perm[: len(T) // 2]], T[perm[len(T) // 2:]]
    sigma = M.within_sigma(Tcal)
    out["inputs"]["sigma"] = sigma
    out["inputs"]["sigma_all"] = M.within_sigma(T)
    # k4 and the throughput bound depend on the challenge rate assumed for the site's users
    out["gap_sensitivity"] = []
    for gap in [60.0, 120.0, 300.0, 900.0, 1800.0]:
        th = M.calibrate(humans(Tcal, sigma, gap=gap), 50, FPR, np.random.default_rng(1))
        out["gap_sensitivity"].append(dict(gap_s=gap, k4=th["k4"], bound=th["k4"] * crnn_acc))
        print("gap", gap, th["k4"], flush=True)
    alts = [("nominal", {}), ("sigma 0.32", dict(sigma=0.32)), ("sigma 0.48", dict(sigma=0.48)),
            ("accuracy mean 0.71", dict(acc_mean=0.71)), ("accuracy mean 0.93", dict(acc_mean=0.93, acc_sd=0.05)),
            ("accuracy homogeneous", dict(acc_sd=0.0)), ("accuracy SD 0.15", dict(acc_sd=0.15)),
            ("gap 60 s", dict(gap=60.0)), ("gap 1800 s", dict(gap=1800.0)), ("pooled latencies", dict(mode="pool"))]
    for n in [20, 50]:
        th = M.calibrate(humans(Tcal, sigma), n, FPR, np.random.default_rng(1))
        for name, kw in alts:
            kw = dict(kw)
            sg = kw.pop("sigma", sigma)
            r = M.run_profile(humans(Ttest, sg, **kw), th, n, np.random.default_rng(7), 2000)
            out["misspec"].append(dict(horizon=n, alternative=name, **r))
            print("misspec", n, name, round(r["rate"], 4), r["first_property"], flush=True)
    # trade-off: calibrating on homogeneous human accuracy (all keys at 0.84) makes P5 usable,
    # at the cost of false positives when real accuracy varies across people
    out["homog_calibration"] = []
    for n in [20, 50, 100]:
        th = M.calibrate(humans(Tcal, sigma, acc_sd=0.0), n, FPR, np.random.default_rng(1))
        rows = {"Human, homogeneous (as calibrated)": humans(Ttest, sigma, acc_sd=0.0),
                "Human, heterogeneous SD 0.10": humans(Ttest, sigma)}
        for p in bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
            if p.name.startswith("Person mimic"):
                rows[p.name] = p
        res = {k: M.run_profile(p, th, n, np.random.default_rng(8), 2000) for k, p in rows.items()}
        out["homog_calibration"].append(dict(horizon=n, thresholds=th, results=res))
        print("homog", n, {k: round(v["rate"], 4) for k, v in res.items()}, flush=True)
    (R / "e5.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
