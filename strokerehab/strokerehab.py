"""StrokeRehab / PrimSeq loader (study30 = stroke, study36 = healthy controls).

Per trial CSV (100 Hz): segment angles, 9 x 3 accelerometer columns (mG, gravity REMOVED by the vendor software),
9 x 3 'Rot' columns (vector part of the unit orientation quaternion, w implied >= 0), label column 'MarkerNames'
(per sample, labelled arm only, prefix r_/l_). Header spelling differs between files (with/without underscores),
so columns are looked up by normalized name.

Label mapping (agreed with the first author, 2026-11-16):
  P2P (1):   reach, transport, retract (= reposition in the Schambra taxonomy), reposition
             EXCEPT the cyclic and failed variants
  non-P2P (0): stabilize, rest, idle, cyclic / failed reach-transport-reposition
  excluded (-1): ignore, empty / unknown labels
Sensor: HAND IMU (dorsum) of the labelled arm - the only sensor whose quaternion matches the vendor angles exactly
(forearm passes in 6 trials only), so gravity can be recovered for it. See screen_trials.py.
"""
import glob, os, re
import numpy as np
import pandas as pd

ROOT = os.environ.get('STROKEREHAB_DIR', 'strokerehab_data')   # raw study30/study36 + cache_hand/ + windows_hand/
OUT = os.environ.get('STROKEREHAB_OUT', 'results/strokerehab')   # result tables
os.makedirs(OUT, exist_ok=True)
FS_IN = 100


def _norm(c):
    return re.sub(r'[^a-z0-9]', '', c.lower())


def label_map(lab):
    l = str(lab).lower()
    if l in ('', 'nan') or 'ignore' in l:
        return -1
    if any(k in l for k in ('reach', 'transport', 'retract', 'reposition')):
        return 0 if ('cyclic' in l or 'fail' in l) else 1
    if any(k in l for k in ('stabilize', 'rest', 'idle')):
        return 0
    return -1


def trial_files(study='study30'):
    if study == 'study30':
        return sorted(f for f in glob.glob(f'{ROOT}/study30/files/s[0-9]*/*/*.csv'))
    return sorted(glob.glob(f'{ROOT}/study36/files/healthy controls/*/C*/*/*.csv'))


def subject_of(path):
    m = re.search(r'/(s\d+|C\d+)/', path)
    return m.group(1)


def load_trial(path, sensor='Forearm'):
    """Returns dict(acc_lin [n,3] mG, quat [n,4] scalar-last, y [n] in {-1,0,1}, side 'RT'/'LT', raw_labels)."""
    d = pd.read_csv(path, low_memory=False)
    cols = {_norm(c): c for c in d.columns}
    labcol = cols.get('markernames')
    raw = d[labcol].astype(str).values if labcol else np.array(['nan'] * len(d))
    sides = pd.Series([r[:2] for r in raw if r[:2] in ('r_', 'l_')]).value_counts()
    side = 'RT' if (len(sides) and sides.index[0] == 'r_') else 'LT'
    s = side.lower()
    acc = d[[cols[f'{sensor.lower()}accelsensor{x}{s}mg'] for x in 'xyz']].values.astype(float)
    v = d[[cols[f'{s}{sensor.lower()}rot{x}'] for x in 'xyz']].values.astype(float)
    y = np.array([label_map(r) for r in raw])
    return dict(acc_lin=acc, quat=quat_w_pos(v), y=y, side=side, raw_labels=raw, n_vec_gt1=int((np.abs(v) > 1).sum()))


def quat_w_pos(v):
    """Scalar-last quaternion with w = +sqrt(1 - |v|^2). Verified convention: with it, R_angles = A q^-1 B holds to <1 deg
    on 474 hand trials (screen_trials.py). The continuity-based sign choice below made exact trials worse (0.05 -> 9 deg)."""
    return np.c_[v, np.sqrt(np.clip(1 - (v ** 2).sum(1), 0, None))]


def complete_quat(v):  # DEPRECATED (wrong convention, kept for the record)
    """Only the vector part (x, y, z) is released; |w| = sqrt(1 - |v|^2) but its SIGN is lost. Assuming w >= 0 produces
    100+ deg jumps between consecutive 10 ms samples. Choose the sign per sample that keeps the rotation continuous
    (max |<q_t, q_{t-1}>|, sign-invariant since q and -q are the same rotation). Returns scalar-last [n, 4]."""
    w = np.sqrt(np.clip(1 - (v ** 2).sum(1), 0, None))
    q = np.c_[v, w]
    for t in range(1, len(q)):
        alt = np.r_[v[t], -w[t]]
        if abs(alt @ q[t - 1]) > abs(q[t] @ q[t - 1]):
            q[t] = alt
    return q
