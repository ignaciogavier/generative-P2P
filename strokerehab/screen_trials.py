"""screen every StrokeRehab trial for an exact quaternion <-> segment-angle relation (gravity recoverable).

For each trial with segment angles, and for the hand and forearm sensors of the labelled arm, fit constant rotations A, B
with R_angles(t) = A q(t)^-1 B (q = released vector part, w = +sqrt(1-|v|^2); angles = Euler 'xyz' of [course, pitch, roll],
validated as exact on s01). Fit on 400 evenly spaced samples, then evaluate on every 5th sample of the whole trial.
A trial passes for a sensor if the median residual < 1 deg AND p99 < 3 deg.
Output: <STROKEREHAB_OUT>/screen_trials.csv (one row per trial x sensor), including A and v = B x_hat for later gravity reconstruction.
"""
import os, sys, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strokerehab as sr


def quat_w_pos(v):
    return Rr.from_quat(np.c_[v, np.sqrt(np.clip(1 - (v ** 2).sum(1), 0, None))])


def fit(Rq, Ra, restarts=6):
    def res(p):
        return ((Rr.from_rotvec(p[:3]) * Rq * Rr.from_rotvec(p[3:])).inv() * Ra).as_rotvec().ravel()
    best = None
    for s in range(restarts):
        r = least_squares(res, np.random.default_rng(s).normal(0, 1.5, 6), loss='soft_l1', f_scale=0.1)
        if best is None or r.cost < best.cost: best = r
        if np.median(np.linalg.norm(res(best.x).reshape(-1, 3), axis=1)) < np.deg2rad(0.5): break
    return best.x


def screen(f):
    out = []
    try:
        d = pd.read_csv(f, low_memory=False); cols = {sr._norm(c): c for c in d.columns}
        if not any(k.startswith('forearmcourse') for k in cols):
            return [dict(file=f, subject=sr.subject_of(f), task=f.split('/')[-2], has_angles=False)]
        t = sr.load_trial(f); s = t['side'].lower()
        n = len(d); idx = np.linspace(0, n - 1, min(400, n)).astype(int); ev = np.arange(0, n, 5)
        for seg in ('hand', 'forearm'):
            v = d[[cols[f'{s}{seg}rot{x}'] for x in 'xyz']].values.astype(float)
            Rq = quat_w_pos(v).inv()
            Ra = Rr.from_euler('xyz', d[[cols[f'{seg}{k}{s}deg'] for k in ('course', 'pitch', 'roll')]].values, degrees=True)
            p = fit(Rq[idx], Ra[idx])
            A, B = Rr.from_rotvec(p[:3]), Rr.from_rotvec(p[3:])
            e = np.rad2deg(np.linalg.norm(((A * Rq[ev] * B).inv() * Ra[ev]).as_rotvec(), axis=1))
            vv = B.apply([1, 0, 0])
            out.append(dict(file=f, subject=sr.subject_of(f), task=f.split('/')[-2], has_angles=True, side=s, sensor=seg,
                            n=n, label_frac_p2p=float(np.mean(t['y'] == 1)), max_abs_v=float(np.linalg.norm(v, axis=1).max()),
                            med_err=float(np.median(e)), p99_err=float(np.percentile(e, 99)),
                            passed=bool(np.median(e) < 1 and np.percentile(e, 99) < 3),
                            A_rotvec=' '.join(f'{x:.6f}' for x in p[:3]), B_rotvec=' '.join(f'{x:.6f}' for x in p[3:]),
                            vx=vv[0], vy=vv[1], vz=vv[2]))
    except Exception as ex:
        out.append(dict(file=f, subject=sr.subject_of(f), task=f.split('/')[-2], error=repr(ex)))
    return out


if __name__ == '__main__':
    files = sr.trial_files('study30')
    if len(sys.argv) > 1: files = files[:int(sys.argv[1])]   # smoke test
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    with Pool(nproc) as pool:
        rows = [r for rs in pool.imap_unordered(screen, files, chunksize=4) for r in rs]
    df = pd.DataFrame(rows)
    out = os.path.join(sr.OUT, 'screen_trials.csv' if len(sys.argv) == 1 else 'screen_trials_smoke.csv')
    df.to_csv(out, index=False)
    a = df[df.get('has_angles', False) == True]
    print(f'{len(files)} trials, {int(a.file.nunique())} with angles, errors: {int(df.get("error", pd.Series(dtype=str)).notna().sum())}')
    if len(a):
        print(a.groupby('sensor').passed.agg(['sum', 'count']))
        print(a[a.passed].groupby('sensor').subject.nunique().rename('subjects passing'))
        print(a.groupby(['subject', 'sensor']).passed.mean().unstack().round(2).to_string())
