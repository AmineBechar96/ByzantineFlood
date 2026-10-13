"""Image preprocessing and augmentation (numpy / OpenCV, no deep-learning deps).

All functions take and return uint8 RGB arrays of shape (H, W, 3). Random
functions take a numpy Generator so every augmentation is reproducible from a
seed.

Pipeline (training):  load -> [fisheye_warp] -> nuisance augs -> letterbox
Pipeline (evaluation): load -> letterbox
Fisheye warping is applied only to non-fisheye sources (DeepFlood,
AlleyFloodNet, STURM); FishEye8K and Grok images are already fisheye.
"""
from __future__ import annotations

import cv2
import numpy as np

# ------------------------------------------------------------- geometry --


def to_rgb(img) -> np.ndarray:
    """PIL image or array (gray, RGBA, CMYK-converted) -> uint8 RGB array."""
    if hasattr(img, "convert"):
        img = np.asarray(img.convert("RGB"))
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return np.ascontiguousarray(a.astype(np.uint8))


def letterbox(img: np.ndarray, size: int = 384, fill: int = 0) -> np.ndarray:
    """Scale so the longest side equals `size`, then pad to size x size.
    Nothing is cropped, so the waterline at the bottom of a photo survives."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def fisheye_warp(img: np.ndarray, rng: np.random.Generator, size: int = 512,
                 hfov_deg: tuple = (60.0, 100.0), coverage: tuple = (0.75, 1.0)) -> np.ndarray:
    """Re-project a perspective photo into an equidistant fisheye image inside a
    black circle, like FishEye8K frames.

    A pixel at radius r from the output centre sees the ray at angle
    theta = r / f_e; in the source (pinhole, horizontal FOV ~ hfov) that ray
    lands at radius f_p * tan(theta). `coverage` sets how much of the source
    half-diagonal the circle edge reaches (< 1: corners are cut, as in real
    fisheye frames). Pixels outside the circle or the source stay black.
    """
    h, w = img.shape[:2]
    hfov = np.radians(rng.uniform(*hfov_deg))
    f_p = (w / 2) / np.tan(hfov / 2)
    theta_d = np.arctan(np.hypot(w, h) / 2 / f_p)
    R = size / 2
    f_e = R / (theta_d * rng.uniform(*coverage))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = xx - (size - 1) / 2, yy - (size - 1) / 2
    r = np.hypot(dx, dy)
    theta = np.minimum(r / f_e, np.radians(89.0))
    rs = f_p * np.tan(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(r > 0, rs / r, 0.0)
    map_x = (w - 1) / 2 + dx * scale
    map_y = (h - 1) / 2 + dy * scale
    out = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out[r > R] = 0
    return out

# ---------------------------------------------------- nuisance factors --


def low_light(img, rng):
    """Dusk/night colour camera: darker (gamma), blue-shifted, sensor noise."""
    f = (img.astype(np.float32) / 255.0) ** rng.uniform(2.0, 3.2) * rng.uniform(0.4, 0.7)
    f *= np.array([rng.uniform(0.75, 0.9), rng.uniform(0.85, 0.95), 1.0], dtype=np.float32)
    f += rng.normal(0, rng.uniform(0.015, 0.04), f.shape).astype(np.float32)
    return np.clip(f * 255, 0, 255).astype(np.uint8)


def infrared_night(img, rng, strength=None):
    """Monochrome infrared night look of FishEye8K night frames.
    strength None: random (training); s in [0, 1]: fixed, s = 1 darkest
    (gamma 1.8, gain 0.8)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gamma = rng.uniform(1.2, 1.8) if strength is None else 1.2 + 0.6 * strength
    gain = rng.uniform(0.8, 1.2) if strength is None else 1.2 - 0.4 * strength
    g = np.clip((g ** gamma) * gain, 0, 1)
    g += rng.normal(0, 0.03, g.shape).astype(np.float32)
    g = np.clip(g * 255, 0, 255).astype(np.uint8)
    return np.stack([g] * 3, axis=-1)


def glare(img, rng):
    """Bright blob from the sun or a street light."""
    h, w = img.shape[:2]
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    sig = rng.uniform(0.08, 0.25) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))[..., None]
    f = img.astype(np.float32) + rng.uniform(80, 200) * blob
    return np.clip(f, 0, 255).astype(np.uint8)


def rain_streaks(img, rng, strength=None):
    """Thin bright motion-blurred streaks. strength None: random (training);
    s in [0, 1]: fixed density, length and brightness (s = 1 the upper end of
    each training range); positions and angle stay random."""
    def u(lo, hi):
        return rng.uniform(lo, hi) if strength is None else lo + strength * (hi - lo)
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    n = int(u(0.002, 0.006) * h * w)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    layer[ys, xs] = 1.0
    length = int(u(0.03, 0.08) * h) | 1
    k = np.zeros((length, length), np.float32)
    k[:, length // 2] = 1.0 / length
    M = cv2.getRotationMatrix2D((length / 2, length / 2), rng.uniform(-20, 20), 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    layer = cv2.filter2D(layer, -1, k) * u(6, 12)
    layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=0.7)
    f = img.astype(np.float32) * 0.92 + 255 * np.clip(layer, 0, 1)[..., None] * u(0.45, 0.8)
    return np.clip(f, 0, 255).astype(np.uint8)


def lens_droplets(img, rng, strength=None):
    """Water drops on the lens: small regions locally blurred and brightened.
    strength None: random (training); s in [0, 1]: fixed blur, number and
    size of drops (s = 1: 14 drops of radius 0.10, blur 16); positions random."""
    def u(lo, hi):
        return rng.uniform(lo, hi) if strength is None else lo + strength * (hi - lo)
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=u(8, 16))
    mask = np.zeros((h, w), np.float32)
    count = int(rng.integers(5, 15)) if strength is None else 5 + int(round(9 * strength))
    for _ in range(count):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(mask, c, int(u(0.04, 0.10) * max(h, w)), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3)[..., None]
    f = img * (1 - mask) + (blurred.astype(np.float32) * 1.15 + 10) * mask
    return np.clip(f, 0, 255).astype(np.uint8)


def jpeg(img, rng):
    q = int(rng.integers(30, 90))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def color_jitter(img, rng):
    f = img.astype(np.float32)
    f = (f - f.mean()) * rng.uniform(0.8, 1.2) + f.mean() * rng.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.75, 1.25)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


NUISANCE = {"low_light": (low_light, 0.15), "infrared_night": (infrared_night, 0.10),
            "glare": (glare, 0.15), "rain_streaks": (rain_streaks, 0.20),
            "lens_droplets": (lens_droplets, 0.10), "jpeg": (jpeg, 0.30),
            "color_jitter": (color_jitter, 0.50)}


def augment(img: np.ndarray, rng: np.random.Generator, is_fisheye: bool, size: int = 384,
            p_fisheye: float = 0.7, p_flip: float = 0.5, nuisance: bool = True) -> tuple[np.ndarray, list]:
    """Training transform. Returns the letterboxed image and the list of
    augmentations applied (for logging and ablations). Low light and infrared
    are mutually exclusive."""
    applied = []
    if not is_fisheye and rng.random() < p_fisheye:
        img = fisheye_warp(img, rng)
        applied.append("fisheye_warp")
    if rng.random() < p_flip:
        img = np.ascontiguousarray(img[:, ::-1])
        applied.append("flip")
    night = False
    for name, (fn, p) in (NUISANCE.items() if nuisance else ()):
        if name in ("low_light", "infrared_night") and night:
            continue
        if rng.random() < p:
            img = fn(img, rng)
            applied.append(name)
            night |= name in ("low_light", "infrared_night")
    return letterbox(img, size), applied


def evaluate_transform(img: np.ndarray, size: int = 384) -> np.ndarray:
    return letterbox(img, size)


def adverse_transform(img: np.ndarray, rng: np.random.Generator, size: int = 384,
                      strength: float = 1.0) -> np.ndarray:
    """Fixed adverse condition (exp13, condition model): infrared night, then
    rain streaks, then lens droplets, each at `strength` (1 = full), then the
    evaluation letterbox. Only positions and the streak angle are random."""
    img = infrared_night(img, rng, strength)
    img = rain_streaks(img, rng, strength)
    img = lens_droplets(img, rng, strength)
    return letterbox(img, size)
