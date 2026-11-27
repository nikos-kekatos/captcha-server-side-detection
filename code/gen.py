"""Synthetic text-CAPTCHA generators used in the study.

G1 = the open-source `captcha` library (ImageCaptcha), a scheme the solver
     author does not control.
G2 = our parameterised generator; `level` 0..3 adds the hardening features
     that the literature reports as the classical defences (rotation,
     touching/overlapping glyphs, occluding arcs, clutter, elastic warp).

All CAPTCHAs are generated locally from a seed. No live service is queried.
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 0/O and 1/I/L are removed, as most deployed schemes do.
ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
H, W = 48, 160
SUP = Path("/System/Library/Fonts/Supplemental")
TRAIN_FONTS = [SUP / f for f in ["Arial.ttf", "Arial Bold.ttf", "Verdana.ttf",
                                  "Georgia.ttf", "Times New Roman.ttf", "Tahoma Bold.ttf"]]
HELDOUT_FONTS = [SUP / f for f in ["Trebuchet MS.ttf", "Comic Sans MS.ttf",
                                    "Courier New Bold.ttf", "Impact.ttf"]]

LEVELS = {
    #      rot  spacing  arcs  dots  warp  clutter
    0: dict(rot=0,  space=0.95, arcs=0, dots=0,   warp=0.0, clutter=0),
    1: dict(rot=15, space=0.85, arcs=1, dots=100, warp=0.0, clutter=0),
    2: dict(rot=25, space=0.70, arcs=2, dots=250, warp=1.5, clutter=3),
    3: dict(rot=35, space=0.55, arcs=4, dots=500, warp=2.5, clutter=6),
}


def random_text(rng: random.Random, lo=4, hi=6) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(lo, hi)))


def _elastic(img: Image.Image, amp: float, rng: random.Random) -> Image.Image:
    if amp <= 0:
        return img
    a = np.asarray(img).astype(np.float32)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ph1, ph2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    f1, f2 = rng.uniform(0.04, 0.09), rng.uniform(0.08, 0.15)
    dx = amp * np.sin(yy * f2 + ph2)
    dy = amp * np.sin(xx * f1 + ph1)
    xs = np.clip(np.rint(xx + dx), 0, w - 1).astype(int)
    ys = np.clip(np.rint(yy + dy), 0, h - 1).astype(int)
    return Image.fromarray(a[ys, xs].astype(np.uint8))


def g2(text: str, level: int, fonts, rng: random.Random) -> np.ndarray:
    p = LEVELS[level]
    big = Image.new("L", (W * 2, H * 2), 255)
    font_path = rng.choice(fonts)
    size = rng.randint(54, 66)
    x = rng.randint(6, 20)
    for ch in text:
        font = ImageFont.truetype(str(font_path), size)
        bbox = font.getbbox(ch)
        gw, gh = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8
        glyph = Image.new("L", (gw, gh), 0)
        ImageDraw.Draw(glyph).text((4 - bbox[0], 4 - bbox[1]), ch, font=font, fill=255)
        if p["rot"]:
            glyph = glyph.rotate(rng.uniform(-p["rot"], p["rot"]), expand=True,
                                 resample=Image.BILINEAR)
        y = (H * 2 - glyph.size[1]) // 2 + rng.randint(-8, 8)
        ink = rng.randint(0, 90)
        big.paste(ink, (x, max(0, y)), glyph)
        x += int(glyph.size[0] * p["space"])
    # crop the used width so the text fills the frame, then resize
    used = min(W * 2, x + 20)
    img = big.crop((0, 0, used, H * 2)).resize((W, H), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    for _ in range(p["arcs"]):
        x0, y0 = rng.randint(-20, 40), rng.randint(-10, H)
        x1, y1 = rng.randint(W - 40, W + 20), rng.randint(0, H + 10)
        y0, y1 = min(y0, y1), max(y0, y1)
        d.arc([x0, y0 - 20, x1, y1 + 20], rng.randint(0, 180), rng.randint(180, 360),
              fill=rng.randint(0, 80), width=rng.randint(1, 3))
    for _ in range(p["clutter"]):
        cx, cy, r = rng.randint(0, W), rng.randint(0, H), rng.randint(3, 9)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=rng.randint(40, 140), width=1)
    for _ in range(p["dots"]):
        d.point((rng.randint(0, W - 1), rng.randint(0, H - 1)), fill=rng.randint(0, 160))
    img = _elastic(img, p["warp"], rng)
    if level >= 2:
        img = img.filter(ImageFilter.GaussianBlur(0.5))
    return np.asarray(img, dtype=np.uint8)


_G1 = None


def g1(text: str, rng: random.Random) -> np.ndarray:
    global _G1
    from captcha.image import ImageCaptcha
    if _G1 is None:
        _G1 = ImageCaptcha(width=160, height=60)
    random.seed(rng.random())  # ImageCaptcha uses the global RNG
    im = _G1.generate_image(text).convert("L").resize((W, H), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def make_set(gen: str, n: int, seed: int, level: int = 0, fonts="train"):
    """Return (images uint8 [n,H,W], labels list[str])."""
    rng = random.Random(seed)
    fl = TRAIN_FONTS if fonts == "train" else HELDOUT_FONTS
    X = np.empty((n, H, W), dtype=np.uint8)
    Y = []
    for i in range(n):
        t = random_text(rng)
        X[i] = g1(t, rng) if gen == "G1" else g2(t, level, fl, rng)
        Y.append(t)
    return X, Y


def to_png_bytes(a: np.ndarray, scale: int = 2) -> bytes:
    im = Image.fromarray(a).resize((a.shape[1] * scale, a.shape[0] * scale), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = []
    for lv in range(4):
        X, Y = make_set("G2", 4, 7, lv)
        rows.append(np.concatenate(list(X), axis=1))
    X, Y = make_set("G2", 4, 7, 2, fonts="heldout")
    rows.append(np.concatenate(list(X), axis=1))
    X, Y = make_set("G1", 4, 7)
    rows.append(np.concatenate(list(X), axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / "samples.png")
    print("ok")
