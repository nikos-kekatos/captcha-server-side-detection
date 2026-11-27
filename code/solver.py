"""CRNN-CTC text-CAPTCHA solver (segmentation-free), training, evaluation, PGD.

The architecture follows the CNN + recurrent + CTC family that most text
CAPTCHA solvers since 2018 use. It is small on purpose: the study asks what a
low-budget attacker obtains, not what the best possible solver obtains.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gen import ALPHABET

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NCLS = len(ALPHABET) + 1  # 0 = CTC blank
C2I = {c: i + 1 for i, c in enumerate(ALPHABET)}


class CRNN(nn.Module):
    def __init__(self, width: int = 1):
        super().__init__()
        c = [32 * width, 64 * width, 128 * width, 128 * width]

        def blk(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(pool))
        self.cnn = nn.Sequential(blk(1, c[0], 2), blk(c[0], c[1], 2),
                                 blk(c[1], c[2], (2, 1)), blk(c[2], c[3], (2, 1)))
        # 48x160 -> 3x40
        self.rnn = nn.GRU(c[3] * 3, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, NCLS)

    def forward(self, x):  # x: [B,1,48,160] in [0,1]
        f = self.cnn(x)
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        f, _ = self.rnn(f)
        return self.fc(f)  # [B,T,NCLS]


def to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(X).float().div(255.0).unsqueeze(1)


def encode(labels):
    tg = torch.tensor([C2I[c] for t in labels for c in t], dtype=torch.long)
    ln = torch.tensor([len(t) for t in labels], dtype=torch.long)
    return tg, ln


def ctc_loss(logits, labels):
    lp = logits.log_softmax(2).permute(1, 0, 2)  # T,B,C
    tg, ln = encode(labels)
    T = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
    # CTC is not implemented on MPS; compute on CPU
    return F.ctc_loss(lp.cpu(), tg, T, ln, blank=0, zero_infinity=True)


def decode(logits) -> list[str]:
    best = logits.argmax(2).cpu().numpy()
    out = []
    for row in best:
        s, prev = [], 0
        for k in row:
            if k != prev and k != 0:
                s.append(ALPHABET[k - 1])
            prev = k
        out.append("".join(s))
    return out


def augment(x: torch.Tensor) -> torch.Tensor:
    # light photometric jitter; no geometric augmentation
    g = 1 + 0.2 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5)
    b = 0.1 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5)
    return (x * g + b).clamp(0, 1)


def train(X, Y, epochs=20, bs=128, lr=2e-3, seed=0, width=1, model=None, log=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = (model or CRNN(width)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Y)
    steps = epochs * max(1, n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Xt = to_tensor(X)
    t0, step = time.time(), 0
    m.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n - bs + 1 if n >= bs else 1, bs):
            idx = perm[i:i + bs]
            xb = augment(Xt[idx].to(DEV))
            loss = ctc_loss(m(xb), [Y[j] for j in idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            if step < steps - 1:
                sch.step()
            step += 1
        if log:
            log(f"  ep {ep+1}/{epochs} loss {loss.item():.3f} {time.time()-t0:.0f}s")
    m.eval()
    return m


def cer(a: str, b: str) -> float:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)] / max(1, len(b))


@torch.no_grad()
def predict(m, X, bs=500):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs] if torch.is_tensor(X) else to_tensor(X[i:i + bs])
        out += decode(m(x.to(DEV)))
    return out


def evaluate(m, X, Y, preprocess=None):
    Xt = to_tensor(X) if not torch.is_tensor(X) else X
    if preprocess is not None:
        Xt = preprocess(Xt)
    t = time.time()
    P = predict(m, Xt)
    dt = (time.time() - t) / len(Y)
    acc = float(np.mean([p == y for p, y in zip(P, Y)]))
    c = float(np.mean([cer(p, y) for p, y in zip(P, Y)]))
    return dict(acc=acc, cer=c, n=len(Y), k=int(round(acc * len(Y))), ms_per_img=1000 * dt)


def pgd(m, X, Y, eps, steps=10, alpha=None, bs=250):
    """L-inf PGD that maximises the CTC loss of the true label (untargeted)."""
    if eps == 0:
        return to_tensor(X)
    alpha = alpha or 2.5 * eps / steps
    out = []
    m.eval()
    # cuDNN-style RNNs need train mode for backward on some backends; GRU on MPS is fine in eval
    for i in range(0, len(Y), bs):
        x0 = to_tensor(X[i:i + bs]).to(DEV)
        y = Y[i:i + bs]
        x = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
        for _ in range(steps):
            x.requires_grad_(True)
            loss = ctc_loss(m(x), y)
            g, = torch.autograd.grad(loss, x)
            x = x.detach() + alpha * g.sign()
            x = torch.min(torch.max(x, x0 - eps), x0 + eps).clamp(0, 1)
        out.append(x.detach().cpu())
    return torch.cat(out)


def median3(x: torch.Tensor) -> torch.Tensor:
    p = F.pad(x, (1, 1, 1, 1), mode="replicate")
    u = p.unfold(2, 3, 1).unfold(3, 3, 1)
    return u.reshape(*u.shape[:4], 9).median(-1).values


def quantise(x: torch.Tensor, levels=8) -> torch.Tensor:
    return torch.round(x * (levels - 1)) / (levels - 1)
