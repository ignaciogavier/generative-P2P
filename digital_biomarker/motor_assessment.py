"""Motor assessment (Fig. 5): nested leave-one-subject-out Ridge regression of the FMA-UE score from digital biomarkers.

Run from digital_biomarker/ after feature_extraction.py. Usage: python motor_assessment.py <timestamp> <p2p_true|p2p_pred>
(p2p_true = manual segmentation, p2p_pred = detector segmentation).
Outer LOSO over the 17 stroke subjects. Inside each outer fold, an inner LOSO over the 16 training subjects scores every
candidate (ICC reliability threshold 0.75-0.80 x Ridge alpha 0.1-10 x, for rule 'inner', the segmentation padding delta
0-1 s) on its pooled inner out-of-fold predictions (Spearman with FMA-UE). The best candidate is refitted on the 16
subjects and predicts the held-out subject, so the test subject never influences any choice. Features: ICC(2,1) between
the two halves of each recording, computed on the training subjects only; StandardScaler + Ridge.
Other rules, for comparison: 'orig' = delta chosen once from the |corr| of the best reliable feature with the clinical
scores of all subjects (the submitted analysis; leaks the test subjects' outcomes), 'd0' = no padding.
Output: output/motor_assessment/<timestamp>_<seg>.csv (r, RMSE, MAE, chosen hyper-parameters per rule) and _<seg>_pred.csv
(out-of-fold FMA-UE prediction per subject; average over detector seeds with average_seeds.py).
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from scripts.shimmer_data_loader import load_stroke_demographics  # noqa: E402
from scripts.features import icc  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

TS, SEG = sys.argv[1], sys.argv[2]
OUT = os.path.join('output', 'motor_assessment'); os.makedirs(OUT, exist_ok=True)
NJ = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
ICC_THRESH = 0.75

THRS = np.linspace(0.75, 0.8, 6)
ALPHAS = np.logspace(-1, 1, 5, base=10)


def iccs(f1, f2):
    F = np.stack([f1, f2], 2)
    return np.array([icc(F[:, f], '(2, 1)')[0] for f in range(F.shape[1])])


def fit_predict(Xtr, ytr, Xte, mask, alpha):
    sc = StandardScaler().fit(Xtr[:, mask])
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr[:, mask]), ytr)
    return m.predict(sc.transform(Xte[:, mask]))


def spearman(y, p):
    c = stats.spearmanr(y, p)[0]
    return 0.0 if np.isnan(c) else c


def inner_scores(X, f1, f2, y):
    """Pooled inner-LOSO out-of-fold Spearman for every (threshold, alpha); NaN if invalid."""
    n = len(y); P = np.full((len(THRS), len(ALPHAS), n), np.nan)
    for k in range(n):
        tr = np.arange(n) != k
        ic = iccs(f1[tr], f2[tr])
        for i, t in enumerate(THRS):
            mask = ic > t
            if not mask.any(): continue
            for j, a in enumerate(ALPHAS):
                P[i, j, k] = fit_predict(X[tr], y[tr], X[~tr], mask, a)[0]
    S = np.full((len(THRS), len(ALPHAS)), np.nan)
    for i in range(len(THRS)):
        for j in range(len(ALPHAS)):
            if np.isfinite(P[i, j]).all(): S[i, j] = spearman(y, P[i, j])
    return S


def outer_fold(D, y, tr_s, dls):
    n = len(y); tr = np.arange(n) != tr_s; best = None
    for dl in dls:
        e = D[dl]; X, f1, f2 = e['features'], e['features_1st'], e['features_2nd']
        S = inner_scores(X[tr], f1[tr], f2[tr], y[tr])
        if not np.isfinite(S).any(): continue
        i, j = np.unravel_index(np.nanargmax(S), S.shape)
        if best is None or S[i, j] > best[0]: best = (S[i, j], dl, i, j)
    if best is None: return None
    _, dl, i, j = best
    e = D[dl]; X = e['features']
    mask = iccs(e['features_1st'][tr], e['features_2nd'][tr]) > THRS[i]
    return fit_predict(X[tr], y[tr], X[~tr], mask, ALPHAS[j])[0], dl, THRS[i], ALPHAS[j], best[0]


def load(ts, seg):
    d = np.load(f'output/spearmanr_all/feature_dict_alldelta_{ts}_mahony_ICC_(2, 1)_0.75_wnd_sz_3_overlap_1_3'
                '_split_method_1_1.5.npz', allow_pickle=True)['data'].item()
    return {k[2]: v for k, v in d.items() if k[1] == seg}


def orig_delta(D):
    best, best_c = None, None
    for dl in sorted(D):
        df = D[dl]['dataframe']; ok = df[df['icc'] >= ICC_THRESH].sort_values(by=['corr'], key=np.abs, ascending=False)
        c = ok['corr'].iloc[0] if len(ok) else 0.0
        if best is None or abs(c) > abs(best_c): best, best_c = dl, c
    return best


if __name__ == '__main__':
    demo = load_stroke_demographics(os.path.join('input', 'stroke_subject_info.xlsx'))
    subjects = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
    y = np.array([demo[s].FM for s in subjects], float)
    D = load(TS, SEG)
    assert all(list(D[k]['subjects']) == subjects for k in D)
    rows, P = [], {'subject': subjects, 'FM': y}
    RULES = os.environ.get('RULES', 'orig d0 inner').split(); SFX = os.environ.get('OUTSFX', '')
    for name, rule in [(n, r) for n, r in [('orig', orig_delta(D)), ('d0', 0.0), ('inner', 'inner')] if n in RULES]:
        if rule != 'inner' and rule not in D: print(f'{name}: delta {rule} unavailable'); continue
        dls = sorted(D) if rule == 'inner' else [rule]
        res = Parallel(n_jobs=NJ)(delayed(outer_fold)(D, y, k, dls) for k in range(len(y)))
        if any(r is None for r in res):
            print(f'{name}: no usable candidate in some outer fold', flush=True)
            rows.append(dict(set=TS, seg=SEG, rule=name, r=np.nan, rmse=np.nan, mae=np.nan, deltas='unusable')); continue
        pr = np.array([r[0] for r in res])
        rows.append(dict(set=TS, seg=SEG, rule=name, r=stats.pearsonr(y, pr)[0], rmse=np.sqrt(((pr - y) ** 2).mean()),
                         mae=np.abs(pr - y).mean(), deltas=';'.join(f'{c:g}' for c in sorted({r[1] for r in res})),
                         thr=';'.join(f'{c:g}' for c in sorted({r[2] for r in res})),
                         alpha=';'.join(f'{c:g}' for c in sorted({r[3] for r in res}))))
        P[name] = pr; print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/{TS}_{SEG}{SFX}.csv', index=False)
    pd.DataFrame(P).to_csv(f'{OUT}/{TS}_{SEG}{SFX}_pred.csv', index=False)
