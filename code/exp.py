"""Experiments E1-E3 (solver side). Usage: python exp.py e1|e1b|e2|e3

Every result is written to ../results/<exp>.json; models to ../results/models/.
Seeds: data seeds are disjoint between train (1000+), test (900000+) and
fine-tune (500000+) sets.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)  # several experiments share the machine

import gen
import solver as S

R = Path(__file__).resolve().parent.parent / "results"
MD = R / "models"
MD.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]
NTEST = 2000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def epochs_for(n, target_steps=1800, bs=128):
    return max(12, math.ceil(target_steps / max(1, n // bs)))


def save(name, obj):
    (R / f"{name}.json").write_text(json.dumps(obj, indent=1))


def testsets():
    T = {}
    for lv in range(4):
        T[f"L{lv}"] = gen.make_set("G2", NTEST, 900000 + lv, lv)
        T[f"L{lv}_heldout_fonts"] = gen.make_set("G2", NTEST, 910000 + lv, lv, fonts="heldout")
    T["G1"] = gen.make_set("G1", NTEST, 920000)
    return T


def e1():
    """Train on one hardening level, test on all levels, held-out fonts and G1."""
    T = testsets()
    out = []
    for lv in range(4):
        X, Y = gen.make_set("G2", 20000, 1000 + 10 * lv, lv)
        for s in SEEDS:
            # different seeds also use different training images
            if s:
                X, Y = gen.make_set("G2", 20000, 1000 + 10 * lv + s, lv)
            t = time.time()
            m = S.train(X, Y, epochs=12, seed=s)
            tt = time.time() - t
            torch.save(m.state_dict(), MD / f"e1_L{lv}_s{s}.pt")
            res = {k: S.evaluate(m, *v) for k, v in T.items()}
            out.append(dict(train_level=lv, seed=s, n_train=20000, train_s=tt, test=res))
            log("E1", lv, s, {k: round(v["acc"], 3) for k, v in res.items()})
            save("e1", out)


def e1b():
    """Sample efficiency at the hardest level."""
    Xt, Yt = gen.make_set("G2", NTEST, 900003, 3)
    old = (R / "e1b.json")
    out = json.loads(old.read_text()) if old.exists() else []
    done = {(r["n_train"], r["seed"]) for r in out}
    for n in [250, 1000, 2000, 5000, 10000]:
        for s in SEEDS:
            if (n, s) in done:
                continue
            X, Y = gen.make_set("G2", n, 2000 + 100 * s + n % 97, 3)
            bs = min(128, n // 4)
            ep = epochs_for(n, bs=bs)  # ~1,800 optimiser steps for every n
            t = time.time()
            m = S.train(X, Y, epochs=ep, seed=s, bs=bs)
            r = S.evaluate(m, Xt, Yt)
            out.append(dict(n_train=n, seed=s, epochs=ep, train_s=time.time() - t, **r))
            log("E1b", n, s, round(r["acc"], 3))
            save("e1b", out)


def e2():
    """Cross-scheme transfer and few-shot adaptation with k purchased labels."""
    Tg1 = gen.make_set("G1", NTEST, 920000)
    Tg2 = gen.make_set("G2", NTEST, 900002, 2, fonts="heldout")
    out = []
    for s in SEEDS:
        # source A: G2 mixture of levels 0-3, train fonts
        parts = [gen.make_set("G2", 5000, 3000 + 10 * s + lv, lv) for lv in range(4)]
        XA = np.concatenate([p[0] for p in parts])
        YA = sum([p[1] for p in parts], [])
        mA, mB = S.CRNN().to(S.DEV), S.CRNN().to(S.DEV)
        pA, pB = MD / f"e2_srcG2mix_s{s}.pt", MD / f"e2_srcG1_s{s}.pt"
        if pA.exists():  # cached from an earlier, interrupted run (same seed and data)
            mA.load_state_dict(torch.load(pA, map_location=S.DEV))
        else:
            mA = S.train(XA, YA, epochs=12, seed=s)
            torch.save(mA.state_dict(), pA)
        if pB.exists():
            mB.load_state_dict(torch.load(pB, map_location=S.DEV))
        else:
            XB, YB = gen.make_set("G1", 20000, 4000 + s)
            mB = S.train(XB, YB, epochs=12, seed=s)
            torch.save(mB.state_dict(), pB)
        mA.eval(), mB.eval()
        for src, m0, tgt_name, tgt_test, tgt_gen in [
                ("G2mix", mA, "G1", Tg1, ("G1", 0, "train")),
                ("G1", mB, "G2L2_heldout", Tg2, ("G2", 2, "heldout"))]:
            for k in [0, 100, 300, 1000, 3000]:
                if k == 0:
                    m = m0
                else:
                    Xf, Yf = gen.make_set(tgt_gen[0], k, 500000 + 10 * s + k, tgt_gen[1],
                                          fonts=tgt_gen[2] if tgt_gen[0] == "G2" else "train")
                    # held-out fonts at test time are new images, same font family set:
                    # this measures adaptation to the scheme, not to the exact images
                    m = S.CRNN().to(S.DEV)
                    m.load_state_dict(m0.state_dict())
                    bf = min(64, max(8, k // 4))
                    m = S.train(Xf, Yf, epochs=epochs_for(k, 600, bs=bf), seed=s,
                                bs=bf, lr=5e-4, model=m)
                r = S.evaluate(m, *tgt_test)
                # scratch baseline: same k labels, no source model
                if k >= 1000:  # small-k scratch curve is covered by E1b
                    bsc = min(128, max(8, k // 4))
                    ms = S.train(Xf, Yf, epochs=epochs_for(k, bs=bsc), seed=s, bs=bsc)
                    rs = S.evaluate(ms, *tgt_test)
                else:
                    rs = None
                out.append(dict(src=src, tgt=tgt_name, k=k, seed=s, finetune=r, scratch=rs))
                log("E2", src, "->", tgt_name, k, s, round(r["acc"], 3),
                    None if rs is None else round(rs["acc"], 3))
                save("e2", out)


def e2s():
    """Scratch baselines for E2 at k in {100, 300} (added after red-team review 2026-10-20).
    Same data seeds, batch rule and step budget as the k>=1000 scratch runs in e2()."""
    Tg1 = gen.make_set("G1", NTEST, 920000)
    Tg2 = gen.make_set("G2", NTEST, 900002, 2, fonts="heldout")
    out = []
    for s in SEEDS:
        for src, tgt_name, tgt_test, tgt_gen in [
                ("G2mix", "G1", Tg1, ("G1", 0, "train")),
                ("G1", "G2L2_heldout", Tg2, ("G2", 2, "heldout"))]:
            for k in [100, 300]:
                Xf, Yf = gen.make_set(tgt_gen[0], k, 500000 + 10 * s + k, tgt_gen[1],
                                      fonts=tgt_gen[2] if tgt_gen[0] == "G2" else "train")
                bsc = min(128, max(8, k // 4))
                ms = S.train(Xf, Yf, epochs=epochs_for(k, bs=bsc), seed=s, bs=bsc)
                rs = S.evaluate(ms, *tgt_test)
                out.append(dict(src=src, tgt=tgt_name, k=k, seed=s, scratch=rs))
                log("E2s", src, "->", tgt_name, k, s, round(rs["acc"], 3))
                save("e2s", out)


def e3q():
    """Check of 8-bit quantisation for E3 (red-team item 18): regenerate PGD CAPTCHAs with the
    same surrogates and evaluate them after rounding to 8 bits. No fine-tuning."""
    def load(name):
        m = S.CRNN().to(S.DEV)
        m.load_state_dict(torch.load(MD / name, map_location=S.DEV))
        return m.eval()
    Xt, Yt = gen.make_set("G2", 1000, 930002, 2)
    out = []
    for s_def, s_att in [(0, 1), (1, 2), (2, 0)]:
        sur, att = load(f"e1_L2_s{s_def}.pt"), load(f"e1_L2_s{s_att}.pt")
        for e255 in [2, 4, 8, 16]:
            Xadv = S.pgd(sur, Xt, Yt, e255 / 255, steps=10)
            Xq = (Xadv * 255).round() / 255
            row = dict(eps255=e255, def_seed=s_def, att_seed=s_att,
                       surrogate=S.evaluate(sur, Xq, Yt), transfer=S.evaluate(att, Xq, Yt),
                       surrogate_float=S.evaluate(sur, Xadv, Yt), transfer_float=S.evaluate(att, Xadv, Yt))
            out.append(row)
            log("E3q", s_def, e255, {k: round(v["acc"], 3) for k, v in row.items() if isinstance(v, dict)})
            save("e3q", out)


def e3():
    """Adversarial CAPTCHAs: transfer, cheap purification, adaptive fine-tuning."""
    def load(name):
        m = S.CRNN().to(S.DEV)
        m.load_state_dict(torch.load(MD / name, map_location=S.DEV))
        return m.eval()
    Xt, Yt = gen.make_set("G2", 1000, 930002, 2)
    out = []
    for s_def, s_att in [(0, 1), (1, 2), (2, 0)]:
        sur = load(f"e1_L2_s{s_def}.pt")
        att = load(f"e1_L2_s{s_att}.pt")
        for e255 in [0, 2, 4, 8, 16]:
            eps = e255 / 255
            Xadv = S.pgd(sur, Xt, Yt, eps, steps=10)
            psnr = float(10 * np.log10(1 / max(1e-12, ((Xadv - S.to_tensor(Xt)) ** 2).mean().item())))
            row = dict(eps255=e255, def_seed=s_def, att_seed=s_att, psnr=psnr,
                       surrogate=S.evaluate(sur, Xadv, Yt),
                       transfer=S.evaluate(att, Xadv, Yt),
                       transfer_median=S.evaluate(att, Xadv, Yt, preprocess=S.median3),
                       transfer_quant=S.evaluate(att, Xadv, Yt, preprocess=S.quantise))
            if e255:
                # attacker buys 1000 labelled adversarial CAPTCHAs served by the defender
                Xf, Yf = gen.make_set("G2", 1000, 600000 + 10 * s_def + e255, 2)
                Xf_adv = (S.pgd(sur, Xf, Yf, eps, steps=10) * 255).round().byte().squeeze(1).numpy()
                m = S.CRNN().to(S.DEV)
                m.load_state_dict(att.state_dict())
                # mix with clean data the attacker already owns to avoid forgetting
                Xc, Yc = gen.make_set("G2", 1000, 610000 + s_att, 2)
                m = S.train(np.concatenate([Xf_adv, Xc]), Yf + Yc, epochs=epochs_for(2000, 600),  # = 40 epochs, as run
                            seed=s_att, bs=64, lr=5e-4, model=m)
                row["adaptive_ft"] = S.evaluate(m, Xadv, Yt)
                row["adaptive_ft_clean"] = S.evaluate(m, Xt, Yt)
            out.append(row)
            log("E3", s_def, s_att, e255, round(psnr, 1), {k: round(v["acc"], 3) for k, v in row.items()
                                                          if isinstance(v, dict)})
            save("e3", out)


if __name__ == "__main__":
    {"e1": e1, "e1b": e1b, "e2": e2, "e2s": e2s, "e3": e3, "e3q": e3q}[sys.argv[1]]()
