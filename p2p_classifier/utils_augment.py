"""Standard IMU augmentation baseline (Um et al., 2017, ICMI, "Data augmentation of wearable
sensor data for Parkinson's disease monitoring using convolutional neural networks").

Literature-default strengths, taken from the authors' reference implementation
(github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data), NOT tuned on our data:
  jitter            additive N(0, sigma^2), sigma = 0.05 g  -> 0.05 * 9.81 m/s^2 (our data are m/s^2)
  scaling           per-channel factor ~ N(1, 0.1^2)
  magnitude warping smooth per-channel factor curve: cubic spline through 4 knots ~ N(1, 0.2^2)
  time warping      smooth time distortion: cubic spline through 4 knots ~ N(1, 0.2^2), cumulative
  rotation          uniformly random 3-D rotation (random axis, angle ~ U(-pi, pi))
Composition: every transform is applied independently with probability 0.5, in the order above
(random subset per sample, on the fly). Applied to TRAINING windows only.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

G = 9.81
UM2017 = dict(jitter_sigma=0.05 * G, scale_sigma=0.1, magwarp_sigma=0.2, timewarp_sigma=0.2,
              knots=4, p=0.5)


def _smooth_curves(rng, n_ch, T, sigma, knots):
    """(n_ch, T) random smooth curves around 1 (Um et al. GenerateRandomCurves)."""
    xx = np.linspace(0, T - 1, knots + 2)
    yy = rng.normal(1.0, sigma, size=(n_ch, knots + 2))
    t = np.arange(T)
    return np.stack([CubicSpline(xx, yy[c])(t) for c in range(n_ch)], 0)


def jitter(x, rng, sigma):
    return x + rng.normal(0.0, sigma, size=x.shape)


def scaling(x, rng, sigma):
    return x * rng.normal(1.0, sigma, size=(x.shape[0], 1))


def magnitude_warp(x, rng, sigma, knots):
    return x * _smooth_curves(rng, x.shape[0], x.shape[1], sigma, knots)


def time_warp(x, rng, sigma, knots):
    C, T = x.shape
    tt = np.cumsum(_smooth_curves(rng, C, T, sigma, knots), axis=1)  # distorted time, per channel
    tt = tt * (T - 1) / tt[:, -1:]                                     # rescale to [0, T-1]
    t = np.arange(T)
    return np.stack([np.interp(t, tt[c], x[c]) for c in range(C)], 0)


def random_rotation(x, rng):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi, np.pi)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R @ x


def augment_um2017(x, rng, cfg=UM2017):
    """x: (3, T) accelerometer window in m/s^2. Returns a new array."""
    x = np.asarray(x, dtype=np.float64)
    p = cfg['p']
    if rng.random() < p: x = jitter(x, rng, cfg['jitter_sigma'])
    if rng.random() < p: x = scaling(x, rng, cfg['scale_sigma'])
    if rng.random() < p: x = magnitude_warp(x, rng, cfg['magwarp_sigma'], cfg['knots'])
    if rng.random() < p: x = time_warp(x, rng, cfg['timewarp_sigma'], cfg['knots'])
    if rng.random() < p: x = random_rotation(x, rng)
    return x


class AugmentedDataset(Dataset):
    """Wraps a dataset returning {'data': FloatTensor(3, T), 'label': ...}; augments 'data' on the fly.
    Uses a per-worker numpy Generator so DataLoader workers do not share random streams."""

    def __init__(self, base, cfg=UM2017, seed=0):
        self.base, self.cfg, self.seed = base, cfg, seed
        self._rng = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self._rng is None:
            import torch.utils.data as tud
            info = tud.get_worker_info()
            self._rng = np.random.default_rng([self.seed, 0 if info is None else info.id + 1])
        import torch
        s = self.base[idx]
        s = dict(s)
        s['data'] = torch.FloatTensor(augment_um2017(s['data'].numpy(), self._rng, self.cfg))
        return s
