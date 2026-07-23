import os
import math
import csv
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import time
import numpy as np
from PIL import Image


# ----------------------------
# I/O and preprocessing
# ----------------------------

def load_grayscale(path: str, max_side: int = 800) -> np.ndarray:
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        img = img.resize((int(round(w / scale)), int(round(h / scale))), resample=Image.BILINEAR)
    return (np.asarray(img, dtype=np.float32) / 255.0)


def normalize_contrast(I: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.median(I)
    mad = np.median(np.abs(I - med)) + eps
    Z = (I - med) / mad
    return (0.5 * (np.tanh(Z / 3.0) + 1.0)).astype(np.float32)


# ----------------------------
# From-scratch Sobel edges
# ----------------------------

def convolve2d_same(I: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = I.shape
    kH, kW = K.shape
    pad_h, pad_w = kH // 2, kW // 2
    P = np.pad(I, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            patch = P[y:y + kH, x:x + kW]
            out[y, x] = float(np.sum(patch * K))
    return out


def sobel_edges(I: np.ndarray) -> np.ndarray:
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    gx = convolve2d_same(I, Kx)
    gy = convolve2d_same(I, Ky)
    mag = np.sqrt(gx * gx + gy * gy)
    return (mag / (float(np.max(mag)) + 1e-8)).astype(np.float32)


def edge_map(I: np.ndarray, thresh: float = 0.25) -> np.ndarray:
    mag = sobel_edges(I)
    return np.clip((mag - thresh) / (1.0 - thresh), 0.0, 1.0).astype(np.float32)


# ----------------------------
# Symmetry scoring
# ----------------------------

def make_weight_mask(H: int, W: int, top_frac: float = 0.10, bottom_frac: float = 0.15) -> np.ndarray:
    w = np.ones((H, W), dtype=np.float32)
    t = int(round(H * top_frac))
    b = int(round(H * bottom_frac))
    if t > 0:
        w[:t, :] = 0.0
    if b > 0:
        w[H - b:, :] = 0.0
    return w


def reflect_points(xs: np.ndarray, ys: np.ndarray, theta: float, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny = math.cos(theta), math.sin(theta)
    d = nx * xs + ny * ys - rho
    return xs - 2.0 * d * nx, ys - 2.0 * d * ny


def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    H, W = img.shape
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    def inside(ix, iy):
        return (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

    Ia = np.zeros_like(x, dtype=np.float32)
    Ib = np.zeros_like(x, dtype=np.float32)
    Ic = np.zeros_like(x, dtype=np.float32)
    Id = np.zeros_like(x, dtype=np.float32)

    m = inside(x0, y0); Ia[m] = img[y0[m], x0[m]]
    m = inside(x0, y1); Ib[m] = img[y1[m], x0[m]]
    m = inside(x1, y0); Ic[m] = img[y0[m], x1[m]]
    m = inside(x1, y1); Id[m] = img[y1[m], x1[m]]

    return (wa * Ia + wb * Ib + wc * Ic + wd * Id).astype(np.float32)


def reflection_score(E: np.ndarray,
                     theta: float,
                     rho: float,
                     wmask: Optional[np.ndarray] = None,
                     eps: float = 1e-6,
                     sample_step: int = 2) -> float:
    H, W = E.shape
    ys, xs = np.mgrid[0:H:sample_step, 0:W:sample_step]
    xs = xs.astype(np.float32); ys = ys.astype(np.float32)

    w = 1.0 if wmask is None else wmask[0:H:sample_step, 0:W:sample_step].astype(np.float32)

    x2, y2 = reflect_points(xs, ys, theta, rho)
    Er = bilinear_sample(E, x2, y2)

    a = w * E[0:H:sample_step, 0:W:sample_step]
    num = float(np.sum(a * Er))
    den = float(np.sum(a * a)) + eps
    return num / den


@dataclass
class AxisResult:
    theta: float
    rho: float
    score: float


def search_reflection_axis(E: np.ndarray,
                           theta_range_deg: Tuple[float, float] = (-15.0, 15.0),
                           theta_step_deg: float = 2.0,
                           rho_step_px: int = 8,
                           sample_step: int = 4) -> AxisResult:
    H, W = E.shape
    wmask = make_weight_mask(H, W)
    diag = math.hypot(W, H)
    rhos = np.arange(-diag, diag + 1e-9, rho_step_px, dtype=np.float32)
    thetas = np.deg2rad(np.arange(theta_range_deg[0], theta_range_deg[1] + 1e-9, theta_step_deg))

    best = AxisResult(theta=0.0, rho=0.0, score=-1.0)
    for th in thetas:
        for rho in rhos:
            s = reflection_score(E, float(th), float(rho), wmask=wmask, sample_step=sample_step)
            if s > best.score:
                best = AxisResult(theta=float(th), rho=float(rho), score=float(s))
    return best


# ----------------------------
# Periodicity via autocorrelation
# ----------------------------

def autocorr_1d(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x)) + eps
    n = x.shape[0]

    m = 1
    while m < 2 * n:
        m *= 2
    X = np.fft.rfft(x, n=m)
    A = np.fft.irfft(X * np.conj(X), n=m)[:n]
    return (A / denom).astype(np.float32)


def periodicity_score_1d(profile: np.ndarray, lag_min: int, lag_max: int) -> Tuple[float, int]:
    A = autocorr_1d(profile)
    lag_max = min(lag_max, A.shape[0] - 1)
    if lag_max <= lag_min:
        return 0.0, lag_min
    region = A[lag_min:lag_max + 1]
    k = int(np.argmax(region)) + lag_min
    return float(A[k]), int(k)


def extract_features(image_path: str,
                     edge_thresh: float = 0.25,
                     theta_range_deg: Tuple[float, float] = (-15.0, 15.0),
                     theta_step_deg: float = 1.0,
                     rho_step_px: int = 4,
                     sample_step: int = 2,
                     lag_min: int = 8,
                     lag_max_frac: float = 0.33) -> Dict[str, float]:
    I = normalize_contrast(load_grayscale(image_path))
    E = edge_map(I, thresh=edge_thresh)
    H, W = E.shape

    axis = search_reflection_axis(
        E,
        theta_range_deg=theta_range_deg,
        theta_step_deg=theta_step_deg,
        rho_step_px=rho_step_px,
        sample_step=sample_step,
    )

    # periodicity in x (repeated window bays) and y (repeated floors)
    px = np.sum(E, axis=0)
    py = np.sum(E, axis=1)
    lag_max_x = max(lag_min + 1, int(round(W * lag_max_frac)))
    lag_max_y = max(lag_min + 1, int(round(H * lag_max_frac)))

    Sper_x, lag_x = periodicity_score_1d(px, lag_min=lag_min, lag_max=lag_max_x)
    Sper_y, lag_y = periodicity_score_1d(py, lag_min=lag_min, lag_max=lag_max_y)

    # add a couple of simple interpretable extras
    edge_density = float(np.mean(E > 0.1))
    anisotropy = float(np.mean(np.sum(E, axis=0)) / (np.mean(np.sum(E, axis=1)) + 1e-6))  # crude x vs y structure

    return {
        "S_ref": float(axis.score),
        "theta_deg": float(np.rad2deg(axis.theta)),
        "S_per_x": float(Sper_x),
        "per_x_lag": float(lag_x),
        "S_per_y": float(Sper_y),
        "per_y_lag": float(lag_y),
        "edge_density": float(edge_density),
        "anisotropy": float(anisotropy),
        "H": float(H),
        "W": float(W),
    }


def write_features_csv(image_dir: str, out_csv: str, exts=(".jpg", ".jpeg", ".png")) -> None:
    paths = []
    for fn in sorted(os.listdir(image_dir)):
        if fn.lower().endswith(exts):
            paths.append(os.path.join(image_dir, fn))

    print(f"Scanning directory: {image_dir}")
    print(f"Found {len(paths)} images")
    print("Starting feature extraction...\n")

    rows = []
    t_start = time.time()

    for i, p in enumerate(paths, 1):
        img_name = os.path.basename(p)
        t0 = time.time()

        feats = extract_features(p)
        feats["image"] = img_name
        rows.append(feats)

        dt = time.time() - t0
        elapsed = time.time() - t_start

        print(f"[{i:4d}/{len(paths)}] Finished {img_name} "
              f"in {dt:6.2f}s | elapsed {elapsed:7.1f}s")

    print(f"\nWriting CSV to: {out_csv}")

    fieldnames = [
        "image",
        "S_ref", "theta_deg",
        "S_per_x", "per_x_lag",
        "S_per_y", "per_y_lag",
        "edge_density", "anisotropy",
        "H", "W"
    ]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    total = time.time() - t_start
    print(f"\nDone. Wrote {out_csv}")
    print(f"Total time: {total:.1f}s | Avg per image: {total / max(len(paths),1):.2f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="features.csv")
    args = parser.parse_args()
    write_features_csv(args.image_dir, args.out_csv)
    print(f"Wrote {args.out_csv}")
