"""cache the StrokeRehab trials whose HAND quaternion is exact (screen_trials.csv, sensor=hand, passed).

Per trial (100 Hz, Noraxon hand-sensor frame):
  acc_lin [n,3] m/s^2  vendor linear acceleration (gravity removed; mG * 9.80665e-3)
  g_dir   [n,3]        unit vertical in the sensor frame, SIGN UNKNOWN: g_dir = q(t)^-1 (B x_hat), equivalently
                       A^-1 R_angles(t) x_hat (checked below on every trial)
  y       [n]          {-1,0,1} from strokerehab.label_map;  lab_id + labels: raw label strings
Written to <workspace>/cache_hand/<subject>__<task>__<file stem>.npz.
"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strokerehab as sr

OUT = f'{sr.ROOT}/cache_hand'
G0 = 9.80665


def one(row):
    f = row['file']
    t = sr.load_trial(f, sensor='Hand')
    assert t['side'].lower() == row['side'], (f, t['side'], row['side'])
    q = Rr.from_quat(t['quat'])
    v = np.array([row['vx'], row['vy'], row['vz']])
    g = q.inv().apply(v)
    # cross-check with the angle route on every 10th sample
    d = pd.read_csv(f, low_memory=False); cols = {sr._norm(c): c for c in d.columns}; s = row['side']
    Ra = Rr.from_euler('xyz', d[[cols[f'hand{k}{s}deg'] for k in ('course', 'pitch', 'roll')]].values[::10], degrees=True)
    A = Rr.from_rotvec(np.array(row['A_rotvec'].split(), float))
    g2 = A.inv().apply(Ra.apply([1, 0, 0]))
    err = np.rad2deg(np.arccos(np.clip((g[::10] * g2).sum(1), -1, 1)))
    labels, lab_id = np.unique(t['raw_labels'], return_inverse=True)
    name = f"{row['subject']}__{row['task']}__{os.path.splitext(os.path.basename(f))[0]}".replace(' ', '_')
    np.savez_compressed(f'{OUT}/{name}.npz', acc_lin=(t['acc_lin'] * G0 * 1e-3).astype(np.float32),
                        g_dir=g.astype(np.float32), y=t['y'].astype(np.int8), lab_id=lab_id.astype(np.int16),
                        labels=labels, side=s, subject=row['subject'], task=row['task'], src=f)
    return dict(name=name, subject=row['subject'], task=row['task'], side=s, n=len(g),
                g_route_err_med=float(np.median(err)), g_route_err_p99=float(np.percentile(err, 99)),
                frac_p2p=float(np.mean(t['y'] == 1)), frac_excl=float(np.mean(t['y'] == -1)))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(f'{sr.OUT}/screen_trials.csv')
    d = d[(d.sensor == 'hand') & (d.passed.astype(str) == 'True')]
    rows = d.to_dict('records')
    if len(sys.argv) > 1: rows = rows[:int(sys.argv[1])]
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 1))) as p:
        res = pd.DataFrame(p.map(one, rows, chunksize=2))
    res.to_csv(f'{sr.OUT}/cache_hand_index.csv', index=False)
    print(len(res), 'trials cached;', res.subject.nunique(), 'subjects;', res.n.sum() / 100 / 3600, 'h')
    print('g route agreement (deg): median of medians', res.g_route_err_med.median(), 'max p99', res.g_route_err_p99.max())
