"""Phase I feature space, KDE samplers and movement kinematics (shared by phase1b and analysis/table2_metrics.py).

Feature space (phase1a output, feature_extraction/Subject{nnn}_MoCapTraj_feat_v2.npy, one row per real movement):
  19-D z = [log10 T, PVDVVP_x * (2*hand - 1), PVDVVP_y, PVDVVP_z]; PVDVVP = position (mid), velocity (mid), displacement,
  velocities at both ends and end position of the minimum-jerk quintic fitted per axis, velocities in normalized-time units.
Samplers (Model(tf, h)): Gaussian KDE with bandwidth h in a transformed space, fitted per subject:
  std     per-dimension standardization, isotropic kernel (= the submitted paper, h = 0.05)
  white   + global whitening (multivariate KDE with H = h^2 Sigma)
  local   white + per-movement kernel covariance from its k = 30 nearest neighbours, shrunk 10% to I
          (adaptive/anisotropic KDE; the adopted sampler, h = 0.35)
  copula  per-dimension rank -> normal-score map + whitening
  shape   velocities relative to the displacement amplitude + whitening
"""
import os
import numpy as np, pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from scipy.spatial import cKDTree
from sklearn.neighbors import KernelDensity

# --- quintic bases: PVDVVP <-> polynomial coefficients <-> 6 keyframes on normalized time ---
P = np.array([[1, 0, 0, 0, 0, 0], [1, 1/5, 1/25, 1/125, 1/625, 1/3125], [1, 2/5, 4/25, 8/125, 16/625, 32/3125],
              [1, 3/5, 9/25, 27/125, 81/625, 243/3125], [1, 4/5, 16/25, 64/125, 256/625, 1024/3125], [1, 1, 1, 1, 1, 1]])
Q = np.array([[1, 1/2, 1/4, 1/8, 1/16, 1/32], [0, 1, 1, 3/4, 1/2, 5/16], [0, 1, 1, 1, 1, 1],
              [0, 1, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], [1, 1/2, 1/2, 1/2, 1/2, 1/2]])
Q_inv = np.linalg.inv(Q)          # PVDVVP -> coefficients
TM = P @ Q_inv                    # PVDVVP -> keyframes
# trajectories per class needed by the reduced ablation grid (healthy S_tr in {1, 3, 5}); --n 0 in phase1b
NEED = {1: 20000, 2: 6666, 3: 20000, 4: 20000, 5: 20000, 6: 20000, 8: 20000, 10: 20000, 11: 4000, 12: 20000,
        13: 6666, 14: 6666, 15: 20000, 16: 6666, 17: 6666, 18: 20000, 19: 20000, 20: 6666}  # groups for S_tr in {1,3,5}
SUBJECTS = list(NEED)              # the 18 healthy subjects with MoCap (subject 7 excluded)
TAU = np.linspace(0, 1, 101)


def load_features(feat_dir, subj, kind):
    """Real movements of one subject in the 19-D space; kind = 'p2p' (label < 3) or 'neg' (non-P2P)."""
    a = np.load(os.path.join(feat_dir, f'Subject{subj:03d}_MoCapTraj_feat_v2.npy'))
    a = a[a[:, 2] < 3] if kind == 'p2p' else a[a[:, 2] >= 3]
    return real_space(np.hstack([a[:, :2], a[:, 3:]]))


def real_space(a):  # a: rows of feat_v2 without label col -> [T, hand, 18]
    return np.concatenate([a[:, 0:1], a[:, 2:8] * (2 * a[:, 1:2] - 1), a[:, 8:]], 1)


def to_coeff_space(Z):  # 19-D PVDVVP space -> [T, 18 coeffs] (linear per axis)
    return np.concatenate([Z[:, :1]] + [Z[:, 1 + 6 * k:7 + 6 * k] @ Q_inv.T for k in range(3)], 1)


def skeleton_from_pvdvvp(Z, hands):
    out = [Z[:, :1], hands]
    for k in range(3):
        f = Z[:, 1 + 6 * k:7 + 6 * k] * ((2 * hands - 1) if k == 0 else 1)
        out.append(f @ TM.T)
    return np.concatenate(out, 1)


def skeleton_from_coeff(C, hands):
    out = [C[:, :1], hands]
    for k in range(3):
        c = C[:, 1 + 6 * k:7 + 6 * k] * ((2 * hands - 1) if k == 0 else 1)
        out.append(c @ P.T)
    return np.concatenate(out, 1)


def kde_sample_sklearn(X, h, n, seed):
    """Standardized isotropic KDE as in the submitted code (sklearn), used for the coefficient-space ablation."""
    mu, sd = X.mean(0), X.std(0)
    return KernelDensity(bandwidth=h).fit((X - mu) / sd).sample(n, random_state=seed) * sd + mu


TFS = ['std', 'white', 'local', 'copula', 'shape']
HS = [0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0]
D_IDX = [3, 9, 15]; V_IDX = [2, 4, 5, 8, 10, 11, 14, 16, 17]; EPS = 0.02
KNN = 30; SHRINK = 0.1


def pwl(x, y, v):
    """piecewise-linear monotone map through knots (x, y) with linear extrapolation; returns value and log slope."""
    i = np.clip(np.searchsorted(x, v) - 1, 0, len(x) - 2)
    s = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return y[i] + s * (v - x[i]), np.log(s)


class Model:
    def __init__(self, tf, h): self.tf, self.h = tf, h

    def fit(self, Z):
        self.mu, self.sd = Z.mean(0), Z.std(0)
        return self.fit_s((Z - self.mu) / self.sd)

    def fit_s(self, Zs):                       # everything below lives in the standardized space Zs
        tf = self.tf; Y = Zs
        if tf == 'copula':
            self.kx = [np.unique(Zs[:, j]) for j in range(Zs.shape[1])]
            self.ky = [norm.ppf((np.arange(len(k)) + 0.5) / len(k)) for k in self.kx]
        if tf == 'shape':
            S = self._shape(Zs); self.smu, self.ssd = S.mean(0), S.std(0)
        Y, _ = self.fwd(Zs, stop_at_white=True)
        if tf == 'std':
            self.W, self.ym, self.ldW = np.eye(Y.shape[1]), np.zeros(Y.shape[1]), 0.0
        else:
            self.ym = Y.mean(0); L = np.linalg.cholesky(np.cov(Y.T) + 1e-6 * np.eye(Y.shape[1]))
            self.W = np.linalg.inv(L).T; self.ldW = -np.log(np.diag(L)).sum()
        self.X = (Y - self.ym) @ self.W
        if tf == 'local':
            _, nn = cKDTree(self.X).query(self.X, min(KNN, len(self.X) - 1))
            C = np.stack([np.cov(self.X[k].T) for k in nn]); C = (1 - SHRINK) * C + SHRINK * np.eye(C.shape[1])
            self.Lc = np.linalg.cholesky(C); self.Lci = np.linalg.inv(self.Lc)
            self.ldc = np.log(np.diagonal(self.Lc, axis1=1, axis2=2)).sum(1)
        return self

    # --- Zs <-> X, with log|dX/dZs| ---
    def _shape(self, Zs):
        Z = Zs * self.sd + self.mu; A = np.sqrt((Z[:, D_IDX] ** 2).sum(1) + EPS ** 2)
        S = Z.copy(); S[:, V_IDX] /= A[:, None]; return S

    def fwd(self, Zs, stop_at_white=False):
        jac = np.zeros(len(Zs))
        if self.tf == 'copula':
            Y = np.empty_like(Zs)
            for j in range(Zs.shape[1]):
                Y[:, j], ls = pwl(self.kx[j], self.ky[j], Zs[:, j]); jac += ls
        elif self.tf == 'shape':
            Z = Zs * self.sd + self.mu; A = np.sqrt((Z[:, D_IDX] ** 2).sum(1) + EPS ** 2)
            Y = (self._shape(Zs) - self.smu) / self.ssd
            jac += np.log(self.sd).sum() - np.log(self.ssd).sum() - len(V_IDX) * np.log(A)
        else:
            Y = Zs
        if stop_at_white: return Y, jac
        return (Y - self.ym) @ self.W, jac + self.ldW

    def back(self, X):
        Y = X @ np.linalg.inv(self.W) + self.ym
        if self.tf == 'copula':
            return np.stack([pwl(self.ky[j], self.kx[j], Y[:, j])[0] for j in range(Y.shape[1])], 1)
        if self.tf == 'shape':
            S = Y * self.ssd + self.smu; A = np.sqrt((S[:, D_IDX] ** 2).sum(1) + EPS ** 2)
            Z = S.copy(); Z[:, V_IDX] *= A[:, None]; return (Z - self.mu) / self.sd
        return Y

    # --- KDE in X ---
    def sample_s(self, n, rng):
        i = rng.integers(0, len(self.X), n); e = rng.standard_normal((n, self.X.shape[1]))
        if self.tf == 'local': e = np.einsum('nij,nj->ni', self.Lc[i], e)
        return self.back(self.X[i] + self.h * e)

    def sample(self, n, rng): return self.sample_s(n, rng) * self.sd + self.mu

    def logpdf_s(self, Zs):
        X, jac = self.fwd(Zs); d = X.shape[1]; h = self.h
        if self.tf == 'local':
            out = np.empty((len(X), len(self.X)))
            for k in range(len(self.X)):
                u = (X - self.X[k]) @ self.Lci[k].T
                out[:, k] = -0.5 * (u ** 2).sum(1) / h ** 2 - self.ldc[k]
        else:
            out = -0.5 * ((X[:, None, :] - self.X[None]) ** 2).sum(-1) / h ** 2
        return logsumexp(out, 1) - np.log(len(self.X)) - d * np.log(h) - 0.5 * d * np.log(2 * np.pi) + jac


def kinematics(Z):
    """Z: 19-D [log10 T, PVDVVP x(signed), y, z] -> metrics per movement (positions in m)."""
    T = 10 ** Z[:, 0]
    C = np.stack([Z[:, 1 + 6 * k:7 + 6 * k] @ Q_inv.T for k in range(3)], 1)       # (n, 3, 6) coefficients in tau
    V = np.vander(TAU, 6, increasing=True)                                             # (101, 6)
    dV = np.c_[np.zeros(len(TAU)), V[:, :5] * np.arange(1, 6)]
    ddV = np.c_[np.zeros((len(TAU), 2)), V[:, :4] * (np.arange(2, 6) * np.arange(1, 5))]
    p = np.einsum('tk,nak->nta', V, C)
    v = np.einsum('tk,nak->nta', dV, C) / T[:, None, None]
    acc = np.einsum('tk,nak->nta', ddV, C) / T[:, None, None] ** 2
    return pd.DataFrame(dict(T=T, disp=np.linalg.norm(p[:, -1] - p[:, 0], axis=1), vpeak=np.linalg.norm(v, axis=2).max(1),
                             apeak=np.linalg.norm(acc, axis=2).max(1), reach=np.linalg.norm(p, axis=2).max(1)))


