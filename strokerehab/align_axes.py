"""label-free alignment of the Noraxon hand-sensor frame (StrokeRehab) to our wrist-IMU frame.

Unknowns: the axis correspondence (a signed permutation) and the sign of the recovered vertical. With a candidate signed
permutation M (48 of them; s = det M, P = s M is a proper rotation) the reconstructed accelerometer signal is
    a_tot = P a_lin + s G0 P g_dir = s M a_lin + G0 M g_dir                     (specific force, + up at rest)
We pick the M whose distribution of a_tot DIRECTIONS (50 Hz, 10 Hz low-pass, as in the paper) is closest (Jensen-Shannon on
an equal-area sphere binning) to that of our REAL HEALTHY RIGHT-WRIST recordings (training-side data, all 18 subjects).
No StrokeRehab labels are used. Robustness: the same choice per StrokeRehab subject and on low-motion samples only.
Only right-arm trials are aligned (the reference is the right wrist; 469 of 474 exact trials).
"""
import os, sys, glob, json, itertools
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strokerehab as sr

DATA = os.environ.get('P2P_DATA', 'data')   # our real healthy windows (data_real_th0.4_singleP2P_man/)
G0 = 9.80665
HEALTHY = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
B, A_ = butter(4, 10, fs=100)


def signed_perms():
    out = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3)); M[range(3), p] = s; out.append(M)
    return out


def fib_sphere(n=600):
    i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


CENT = fib_sphere()


def hist(u):
    h = np.bincount(np.argmax(u @ CENT.T, 1), minlength=len(CENT)).astype(float) + 0.5
    return h / h.sum()


def js(p, q):
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def unit(x): return x / np.linalg.norm(x, axis=1, keepdims=True)


def reference(stride=3):
    a = [np.load(f)[::stride] for s in HEALTHY
         for f in sorted(glob.glob(f'{DATA}/data_real_th0.4_singleP2P_man/S{s}/*_racc.npy'))]
    return np.concatenate(a)


def stroke(stride=6):
    """50 Hz after 10 Hz low-pass, then every 3rd sample (stride counted at 100 Hz)."""
    idx = pd.read_csv(f'{sr.OUT}/cache_hand_index.csv'); idx = idx[idx.side == 'rt']
    al, g, subj = [], [], []
    for _, r in idx.iterrows():
        z = np.load(f'{sr.ROOT}/cache_hand/{r["name"]}.npz')
        al.append(filtfilt(B, A_, z['acc_lin'], axis=0)[::stride]); g.append(filtfilt(B, A_, z['g_dir'], axis=0)[::stride])
        subj += [r.subject] * len(al[-1])
    return np.concatenate(al), unit(np.concatenate(g)), np.array(subj)


if __name__ == '__main__':
    ref = reference(); ref_low = ref[np.abs(np.linalg.norm(ref, axis=1) - G0) < 0.5]
    href, href_low = hist(unit(ref)), hist(unit(ref_low))
    al, g, subj = stroke()
    low = np.linalg.norm(al, axis=1) < 0.5
    rows = []
    Ms = signed_perms()
    for k, M in enumerate(Ms):
        s = round(np.linalg.det(M))
        at = s * al @ M.T + G0 * g @ M.T
        u = unit(at)
        r = dict(k=k, M=' '.join(map(str, M.astype(int).ravel())), det=s, js_all=js(hist(u), href),
                 js_low=js(hist(u[low]), href_low),
                 mean_dir_angle=float(np.rad2deg(np.arccos(np.clip(unit(u.mean(0, keepdims=True)) @ unit(ref.mean(0, keepdims=True)).T, -1, 1)))[0, 0]))
        for sj in np.unique(subj):
            r[f'js_{sj}'] = js(hist(u[subj == sj]), href)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values('js_all')
    sj_cols = [c for c in df.columns if c.startswith('js_s')]
    wins = {c[3:]: int(df.loc[df[c].idxmin(), 'k']) for c in sj_cols}
    df.to_csv(f'{sr.OUT}/align_candidates.csv', index=False)
    best = df.iloc[0]
    print(f'reference: {len(ref)} samples (right wrist, 18 healthy); stroke: {len(al)} samples, {len(np.unique(subj))} subjects, {low.mean():.2f} low-motion')
    print(df[['k', 'M', 'det', 'js_all', 'js_low', 'mean_dir_angle']].head(8).to_string(index=False))
    print('best per subject:', wins)
    print('best on low-motion:', int(df.loc[df.js_low.idxmin(), 'k']))
    json.dump(dict(M=np.array(best.M.split(), int).reshape(3, 3).tolist(), det=int(best.det), k=int(best.k),
                   js_all=best.js_all, js_runner_up=float(df.iloc[1].js_all), per_subject_best=wins,
                   low_motion_best=int(df.loc[df.js_low.idxmin(), 'k'])),
              open(f'{sr.OUT}/align_choice.json', 'w'), indent=1)

    # figure: per-axis distributions, reference vs best and runner-up candidate
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.5), sharey='row')
    for row, (kk, lab) in enumerate([(int(best.k), 'best'), (int(df.iloc[1].k), 'runner-up')]):
        M = Ms[kk]; s = round(np.linalg.det(M)); at = s * al @ M.T + G0 * g @ M.T
        for j in range(3):
            bins = np.linspace(-20, 20, 81)
            ax[row, j].hist(ref[:, j], bins, density=True, alpha=.5, label='Healthy wrist (ours)')
            ax[row, j].hist(at[:, j], bins, density=True, alpha=.5, label=f'StrokeRehab hand, {lab}')
            ax[row, j].set_xlabel(f'acc {"xyz"[j]} (m s$^{{-2}}$)')
        ax[row, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{sr.OUT}/align_axes.png', dpi=150)
