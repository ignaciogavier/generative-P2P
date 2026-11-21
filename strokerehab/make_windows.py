"""StrokeRehab hand-sensor windows in our input format (accelerometer, specific force in m s^-2, our wrist axes).

From the cache (build_cache.py) and the label-free axis alignment (align_axes.py -> align_choice.json):
  a_tot = s M a_lin + G0 M g_dir   (s = det M)
then 10 Hz low-pass (4th-order Butterworth, zero phase) at 100 Hz, decimate to 50 Hz, 3 s windows with 2 s stride.
Label (paper rule): positive if the window contains >= 60% of the duration of a P2P movement
(a P2P movement = a maximal run of one P2P label; cyclic/failed variants are non-P2P). Windows that overlap any excluded
sample (ignore / unknown label) are dropped and counted.
Only right-arm trials (the alignment reference is the right wrist).
Output: <workspace>/windows_hand/<subject>.npz with X [n,150,3] float32, y [n], trial [n], t0 [n] (s); summary CSV in <STROKEREHAB_OUT>/.
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strokerehab as sr

G0 = 9.80665
W, STRIDE, FS_OUT, TH = 150, 100, 50, 0.6
B, A_ = butter(4, 10, fs=100)


def runs(lab_id, y):
    """(start, end) at 100 Hz of maximal runs of one P2P label."""
    out, i, n = [], 0, len(y)
    while i < n:
        j = i
        while j + 1 < n and lab_id[j + 1] == lab_id[i]: j += 1
        if y[i] == 1: out.append((i, j + 1))
        i = j + 1
    return out


def windows(z, M):
    s = round(np.linalg.det(M))
    a = s * z['acc_lin'] @ M.T + G0 * z['g_dir'] @ M.T
    a = filtfilt(B, A_, a, axis=0)[::2]
    y100, lab = z['y'], z['lab_id']
    p2p = runs(lab, y100)
    X, Y, T0, n_drop = [], [], [], 0
    for k in range(0, len(a) - W + 1, STRIDE):
        lo, hi = 2 * k, 2 * (k + W)                           # window in 100 Hz samples
        if (y100[lo:hi] == -1).any(): n_drop += 1; continue
        pos = any(max(0, min(hi, e) - max(lo, b)) >= TH * (e - b) for b, e in p2p)
        X.append(a[k:k + W]); Y.append(int(pos)); T0.append(k / FS_OUT)
    return X, Y, T0, n_drop, len(p2p)


if __name__ == '__main__':
    ch = json.load(open(f'{sr.OUT}/align_choice.json')); M = np.array(ch['M'], float)
    K = os.environ.get('STROKEREHAB_ALIGN_K')                          # sensitivity: alternative candidate from align_candidates.csv
    if K is not None:
        c = pd.read_csv(f'{sr.OUT}/align_candidates.csv').set_index('k').loc[int(K)]
        M = np.array(c.M.split(), float).reshape(3, 3); print('ALIGN_K', K, 'M =', M.tolist(), flush=True)
    sfx = '' if K is None else f'_k{K}'
    idx = pd.read_csv(f'{sr.OUT}/cache_hand_index.csv'); idx = idx[idx.side == 'rt']
    out = f'{sr.ROOT}/windows_hand{sfx}'; os.makedirs(out, exist_ok=True)
    rows = []
    for sj, g in idx.groupby('subject'):
        X, Y, TR, T0 = [], [], [], []
        nd = nm = 0
        for name in g.name:
            x, y, t0, d, m = windows(np.load(f'{sr.ROOT}/cache_hand/{name}.npz'), M)
            X += x; Y += y; T0 += t0; TR += [name] * len(y); nd += d; nm += m
        X = np.array(X, np.float32); Y = np.array(Y, np.int8)
        np.savez_compressed(f'{out}/{sj}.npz', X=X, y=Y, trial=np.array(TR), t0=np.array(T0))
        rows.append(dict(subject=sj, trials=len(g), windows=len(Y), pos=int(Y.sum()), pos_rate=float(Y.mean()),
                         dropped_excluded=nd, p2p_movements=nm, acc_norm_median=float(np.median(np.linalg.norm(X, axis=2)))))
    df = pd.DataFrame(rows); df.to_csv(f'{sr.OUT}/windows_hand_summary{sfx}.csv', index=False)
    print(df.to_string(index=False)); print('total windows', df.windows.sum(), 'positives', df.pos.sum())
