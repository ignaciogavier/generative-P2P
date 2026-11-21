"""Table II: realism metrics of the Phase I samplers, every transform x bandwidth (kde_samplers.Model) + KDE on the raw
quintic coefficients ('coeff', ablation B).

Reproduced from synthetic-pipeline/extracting_skeleton_mocapAll.ipynb (cells 17-21):
  distance = |p(1) - p(0)| (m), duration = T (s); speed |v| and radius of curvature |v|^3 / |v x a| along each movement,
  sampled at the MoCap rate (1/150 s); slopes = least-squares log10(duration) ~ log10(distance) and
  log10(speed) ~ log10(radius) (the notebook's 'Loglog regression').
  kl_pub_*  the notebook's estimator, verbatim: 1-D histogram, 50 LINEAR bins over the pooled range, KL(real || synth).
            NOTE (documented, not fixed): for the 'joint' entries the notebook passes a stacked (n, 2) array, which
            np.histogram FLATTENS -> it is a 1-D histogram of distances and durations pooled together, not a joint KL;
            with linear bins the heavy-tailed curvature puts almost all mass in one bin (published KL curvene 4e-5).
  kl_*      corrected estimator: JOINT 2-D histogram of the log10 values, 30 x 30 bins over the real [0.5, 99.5]th
            percentiles (out-of-range values clipped to the edge bins), Laplace smoothing 1e-6, KL(real || synth).
The paper's Table II used ONE KDE pooled over subjects; the training data were generated per subject. Here synthetic =
per subject (as generated), 2,000 P2P samples per subject, pooled; rows 'pooled_std' h=0.05 reproduce the notebook setting.
  km_*      the same KLs at MATCHED sample sizes with a Jeffreys
            pseudo-count (0.5 per bin): over R = 20 random halvings of the real movements, KL(half A || half B) is the noise
            floor (row 'real_half', km_*_ref) and KL(half A || synthetic subsample of len(B) movements / len(B) speed-curvature
            samples) the variant's value; reported as the mean over the 20 splits (and km_*_excess = value - floor).
Usage: python table2_metrics.py --feat_dir ../generative_pipeline/feature_extraction --out_dir results/sampler  (after
       sampler_selection.py; multi-core). Output: table2_metrics.csv (one row per sampler x h), joined with novelty /
       plausibility from sampler_selection.csv. Quote the km_* columns with the real_half row as the noise floor.
"""
import os, sys, argparse
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'generative_pipeline'))
import kde_samplers as ks  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument('--feat_dir', required=True); ap.add_argument('--out_dir', default='results/sampler')
A_ = ap.parse_args(); HERE = A_.out_dir
DT = 1 / 150; NS = 2000; NCURV = 300          # movements per subject used for the (per-sample) speed/curvature metrics
D_IDX = ks.D_IDX


def descr(Z, ncurv, rng):
    """Z (n, 19) -> distance, duration (per movement); speed, radius (per sample, from ncurv movements)."""
    T = 10 ** Z[:, 0]; dist = np.linalg.norm(Z[:, D_IDX], axis=1)
    sp, rc = [], []
    for i in rng.choice(len(Z), min(ncurv, len(Z)), replace=False):
        C = np.stack([Z[i, 1 + 6 * k:7 + 6 * k] @ ks.Q_inv.T for k in range(3)])       # (3, 6) coefficients in tau
        n = max(int(T[i] / DT), 3); tau = np.arange(n) / (n - 1)
        V = np.vander(tau, 6, increasing=True)
        dV = np.c_[np.zeros(n), V[:, :5] * np.arange(1, 6)]; ddV = np.c_[np.zeros((n, 2)), V[:, :4] * (np.arange(2, 6) * np.arange(1, 5))]
        v = dV @ C.T / T[i]; a = ddV @ C.T / T[i] ** 2
        s = np.linalg.norm(v, axis=1); cr = np.linalg.norm(np.cross(v, a), axis=1)
        ok = (cr > 0) & (s > 0); sp.append(s[ok]); rc.append(s[ok] ** 3 / cr[ok])
    return dist, T, np.concatenate(sp), np.concatenate(rc)


def slope(x, y):
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    A = np.c_[np.ones(ok.sum()), np.log10(x[ok])]; return float(np.linalg.lstsq(A, np.log10(y[ok]), rcond=None)[0][1])


def kl_pub(x1, x2, bins=50, eps=1e-10):            # verbatim from the notebook (kl_hist_1d)
    r = (min(np.nanmin(x1), np.nanmin(x2)), max(np.nanmax(x1), np.nanmax(x2)))
    p, _ = np.histogram(x1, bins=bins, range=r, density=True); q, _ = np.histogram(x2, bins=bins, range=r, density=True)
    p = np.clip(p / p.sum(), eps, None); q = np.clip(q / q.sum(), eps, None); return float(np.sum(p * np.log(p / q)))


def kl2(r1, r2, s1, s2, bins=30, a=1e-6):
    lx, ly = np.log10(r1), np.log10(r2); ex = np.nanpercentile(lx, [.5, 99.5]); ey = np.nanpercentile(ly, [.5, 99.5])
    h = lambda x, y: np.histogram2d(np.clip(np.log10(x), *ex), np.clip(np.log10(y), *ey), bins=bins, range=[ex, ey])[0].ravel() + a
    p = h(r1, r2); q = h(s1, s2); p /= p.sum(); q /= q.sum(); return float(np.sum(p * np.log(p / q)))


def kl1(r, s, bins=50, a=1e-6):
    e = np.nanpercentile(np.log10(r), [.5, 99.5])
    h = lambda x: np.histogram(np.clip(np.log10(x), *e), bins=bins, range=e)[0] + a
    p = h(r); q = h(s); p = p / p.sum(); q = q / q.sum(); return float(np.sum(p * np.log(p / q)))


def kl2m(r1, r2, s1, s2, bins=30, a=0.5):        # matched-size variant: Jeffreys pseudo-count per bin
    lx, ly = np.log10(r1), np.log10(r2); ex = np.nanpercentile(lx, [.5, 99.5]); ey = np.nanpercentile(ly, [.5, 99.5])
    h = lambda x, y: np.histogram2d(np.clip(np.log10(x), *ex), np.clip(np.log10(y), *ey), bins=bins, range=[ex, ey])[0].ravel() + a
    p = h(r1, r2); q = h(s1, s2); p /= p.sum(); q /= q.sum(); return float(np.sum(p * np.log(p / q)))


def kl1m(r, s, bins=50, a=0.5):
    e = np.nanpercentile(np.log10(r), [.5, 99.5])
    h = lambda x: np.histogram(np.clip(np.log10(x), *e), bins=bins, range=e)[0] + a
    p = h(r); q = h(s); p = p / p.sum(); q = q / q.sum(); return float(np.sum(p * np.log(p / q)))


KM = ['km_durdist', 'km_curvel', 'km_dist', 'km_dur', 'km_curv', 'km_speed']


def km(A, B):
    """A, B = (dist, T, speed, radius) with per-movement and per-sample arrays."""
    (rd, rT, rs, rc), (sd_, sT, ss, sc) = A, B
    return dict(km_durdist=kl2m(rd, rT, sd_, sT), km_curvel=kl2m(rc, rs, sc, ss), km_dist=kl1m(rd, sd_), km_dur=kl1m(rT, sT),
                km_curv=kl1m(rc, sc), km_speed=kl1m(rs, ss))


def subsample(syn, n_mov, n_smp, rng):
    i = rng.choice(len(syn[0]), n_mov, replace=False); j = rng.choice(len(syn[2]), min(n_smp, len(syn[2])), replace=False)
    return syn[0][i], syn[1][i], syn[2][j], syn[3][j]


def coeff_sample(Z, h, n, rng):
    Zc = ks.to_coeff_space(Z); mu, sd = Zc.mean(0), Zc.std(0)
    X = (Zc - mu) / sd; i = rng.integers(0, len(X), n); S = (X[i] + h * rng.standard_normal((n, X.shape[1]))) * sd + mu
    return np.concatenate([S[:, :1]] + [S[:, 1 + 6 * k:7 + 6 * k] @ ks.Q.T for k in range(3)], 1)


REAL = {s: ks.load_features(A_.feat_dir, s, 'p2p') for s in ks.SUBJECTS}


def run(cfg):
    tf, h = cfg; parts = []
    for s in ks.SUBJECTS:
        rng = np.random.default_rng(1000 * s + 29 + int(1000 * h))
        Z = REAL[s]
        if tf == 'coeff': S = coeff_sample(Z, h, NS, rng)
        elif tf == 'pooled_std': continue
        else: S = ks.Model(tf, h).fit(Z).sample(NS, rng)
        parts.append(descr(S, NCURV, rng))
    if tf == 'pooled_std':                          # notebook setting: one KDE over all subjects
        rng = np.random.default_rng(29); Zall = np.concatenate(list(REAL.values()))
        parts = [descr(ks.Model('std', h).fit(Zall).sample(NS * len(ks.SUBJECTS), rng), NCURV * len(ks.SUBJECTS), rng)]
    return tf, h, [np.concatenate([p[k] for p in parts]) for k in range(4)]


def metrics(real, syn):
    (rd, rT, rs, rc), (sd_, sT, ss, sc) = real, syn
    return dict(slope_durdist=slope(sd_, sT), slope_curvel=slope(sc, ss),
                kl_durdist=kl2(rd, rT, sd_, sT), kl_curvel=kl2(rc, rs, sc, ss),
                kl_dist=kl1(rd, sd_), kl_dur=kl1(rT, sT), kl_curv=kl1(rc, sc), kl_speed=kl1(rs, ss),
                kl_pub_durdist=kl_pub(np.stack([rd, rT], 1), np.stack([sd_, sT], 1)),
                kl_pub_curvel=kl_pub(np.stack([rc, rs], 1), np.stack([sc, ss], 1)),
                kl_pub_dist=kl_pub(rd, sd_), kl_pub_dur=kl_pub(rT, sT), kl_pub_curv=kl_pub(rc, sc), kl_pub_speed=kl_pub(rs, ss))


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    parts = [descr(REAL[s], 10 ** 9, rng) for s in ks.SUBJECTS]          # all real movements
    real = [np.concatenate([p[k] for p in parts]) for k in range(4)]
    rows = [dict(transform='real', h=np.nan, n_mov=len(real[0]), slope_durdist=slope(real[0], real[1]), slope_curvel=slope(real[3], real[2]))]
    # noise floor: random halves of the real movements (per-movement split, curvature samples follow their movement)
    Zall = np.concatenate(list(REAL.values())); idx = rng.permutation(len(Zall)); A, B = Zall[idx[::2]], Zall[idx[1::2]]
    ra, rb = descr(A, 10 ** 9, rng), descr(B, 10 ** 9, rng)
    rows.append(dict(transform='real_half', h=np.nan, n_mov=len(B), **metrics(ra, rb)))
    # matched-size noise floor: R random halvings (each half described once; reused for every synthetic variant)
    R_SPLITS = 20; SPL = []
    for r in range(R_SPLITS):
        idx = np.random.default_rng(100 + r).permutation(len(Zall)); rgen = np.random.default_rng(200 + r)
        a_, b_ = descr(Zall[idx[::2]], 10 ** 9, rgen), descr(Zall[idx[1::2]], 10 ** 9, rgen); SPL.append((a_, b_, km(a_, b_)))
    REF = {k: float(np.mean([s_[2][k] for s_ in SPL])) for k in KM}
    rows[-1].update({**REF, **{f'{k}_sd': float(np.std([s_[2][k] for s_ in SPL], ddof=1)) for k in KM}})
    cfgs = [(tf, h) for tf in ks.TFS + ['coeff'] for h in ks.HS] + [('pooled_std', 0.05)]
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))) as p:
        for tf, h, syn in p.imap(run, cfgs):
            rows.append(dict(transform=tf, h=h, n_mov=len(syn[0]), **metrics(real, syn)))
            vals = [km(a_, subsample(syn, len(b_[0]), len(b_[2]), np.random.default_rng(300 + r))) for r, (a_, b_, _) in enumerate(SPL)]
            for k in KM:
                v = [x[k] for x in vals]; rows[-1].update({k: float(np.mean(v)), f'{k}_sd': float(np.std(v, ddof=1)), f'{k}_excess': float(np.mean(v)) - REF[k]})
            print(tf, h, 'done', flush=True)
    R = pd.DataFrame(rows)
    A = pd.read_csv(f'{HERE}/sampler_selection.csv')[['transform', 'h', 'novelty', 'any', 'll_per_dim']]
    R = R.merge(A, on=['transform', 'h'], how='left'); R.to_csv(f'{HERE}/table2_metrics.csv', index=False)
    pd.set_option('display.width', 250); print(R.round(4).to_string(index=False))
