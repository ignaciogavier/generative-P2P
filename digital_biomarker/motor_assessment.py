"""Motor assessment (Fig. 5): nested leave-one-subject-out Ridge regression of the FMA-UE score from digital biomarkers.

Run from digital_biomarker/ after feature_extraction.py. Usage: python motor_assessment.py <timestamp> <p2p_true|p2p_pred>
(p2p_true = manual segmentation, p2p_pred = detector segmentation).
Outer LOSO over the 17 stroke subjects; inner LOSO grid search over the ICC reliability threshold (0.75-0.80), the Ridge
alpha (0.1-10) and, for rule 'inner', the segmentation padding delta (0-1 s): the delta is chosen inside each outer fold
by the best inner score, so the test subject's score never influences any choice.
Other rules, for comparison: 'orig' = delta chosen once from the |corr| of the best reliable feature with the clinical
scores of all subjects (the submitted analysis; leaks the test subjects' outcomes), 'd0' = no padding.
Output: output/motor_assessment/<timestamp>_<seg>.csv (r, RMSE, chosen deltas per rule) and _<seg>_pred.csv
(out-of-fold FMA-UE prediction per subject; average over detector seeds with average_seeds.py).
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
from scripts.shimmer_data_loader import load_stroke_demographics  # noqa: E402
from scripts.features import icc  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

TS, SEG = sys.argv[1], sys.argv[2]
OUT = os.path.join('output', 'motor_assessment'); os.makedirs(OUT, exist_ok=True)
NJ = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
ICC_THRESH = 0.75


def spearman_scorer(y_true, y_pred):
    corrs = []
    for i in range(y_true.shape[1]):
        corr, _ = stats.spearmanr(y_true[:, i], y_pred[:, i])
        corrs.append(corr if not np.isnan(corr) else 0)
    return np.mean(corrs)


class score(object):
    def __init__(self, X_full, y_full):
        self.X_full, self.y_full = X_full, y_full

    def __call__(self, estimator, X, y=None):
        return spearman_scorer(self.y_full, estimator.predict(self.X_full))


class FeatureSelector(BaseEstimator, RegressorMixin):
    def __init__(self, X_full, feat1, feat2, threshold=0.75):
        self.X_full, self.feat1, self.feat2, self.threshold = X_full, feat1, feat2, threshold

    def fit(self, X, y=None):
        idx = np.zeros((self.X_full.shape[0],), dtype=bool)
        for i in range(len(idx)):
            j = sum(idx[:i + 1])
            if j == X.shape[0]:
                break
            if np.allclose(self.X_full[i], X[j]):
                idx[i] = True
        self.iccs = np.array([icc(np.stack([self.feat1, self.feat2], 2)[idx, f], '(2, 1)')[0]
                              for f in range(self.X_full.shape[1])])
        self.mask = self.iccs > self.threshold
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return X[:, self.mask]


def inner_search(X, f1, f2, y, groups):
    pipe = Pipeline([('feature_selector', FeatureSelector(X, f1, f2)),
                     ('scaler', StandardScaler()),
                     ('regressor', MultiOutputRegressor(Ridge()))])
    grid = {'feature_selector__threshold': np.linspace(0.75, 0.8, 6),
            'regressor__estimator__alpha': np.logspace(-1, 1, 5, base=10)}
    gs = GridSearchCV(pipe, grid, cv=LeaveOneGroupOut().split(X, y, groups), scoring=score(X, y), n_jobs=NJ)
    gs.fit(X, y)
    return gs


def load(ts, seg):
    d = np.load(f'output/spearmanr_all/feature_dict_alldelta_{ts}_mahony_ICC_(2, 1)_0.75_wnd_sz_3_overlap_1_3'
                '_split_method_1_1.5.npz', allow_pickle=True)['data'].item()
    return {k[2]: v for k, v in d.items() if k[1] == seg}


def orig_delta(D):
    """The published rule: first delta kept, replaced only by a strictly higher |corr| of the best ICC-passing feature."""
    best, best_c = None, None
    for dl in sorted(D):
        df = D[dl]['dataframe']; ok = df[df['icc'] >= ICC_THRESH].sort_values(by=['corr'], key=np.abs, ascending=False)
        c = ok['corr'].iloc[0] if len(ok) else 0.0
        if best is None or abs(c) > abs(best_c): best, best_c = dl, c
    return best


def nested(D, y, groups, rule):
    """Outer LOSO; delta fixed (rule = float) or chosen in the inner LOSO (rule = 'inner')."""
    preds, chosen = [], []
    for tr_s in range(len(groups)):
        best = None
        for dl in (sorted(D) if rule == 'inner' else [rule]):
            e = D[dl]; X, f1, f2 = e['features'], e['features_1st'], e['features_2nd']
            tr = np.arange(len(groups)) != tr_s
            try:
                gs = inner_search(X[tr], f1[tr], f2[tr], y[tr], groups[tr])
            except ValueError as err:          # no feature passes the ICC filter for this delta -> delta unusable here
                print(f'  fold {groups[tr_s]} delta {dl}: skipped ({str(err)[:60]})', flush=True); continue
            if rule == 'inner' and not np.isfinite(gs.best_score_): continue   # fixed rules: as fig5_stats.py (refit even if all inner scores NaN)
            if best is None or gs.best_score_ > best[0]: best = (gs.best_score_, dl, gs, X[~tr])
        if best is None: raise RuntimeError(f'no usable delta for outer fold {groups[tr_s]}')
        chosen.append(best[1]); preds.append(best[2].best_estimator_.predict(best[3]))
    return np.concatenate(preds, 0).ravel(), chosen


demo = load_stroke_demographics(os.path.join('input', 'stroke_subject_info.xlsx'))
subjects = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
y = np.array([[demo[s].FM] for s in subjects], float)
D = load(TS, SEG)
assert all(list(D[k]['subjects']) == subjects for k in D)
groups = np.array(subjects)
rows, P = [], {'subject': subjects, 'FM': y.ravel()}
RULES = os.environ.get('RULES', 'orig d0 inner').split(); SFX = os.environ.get('OUTSFX', '')
for name, rule in [(n, r) for n, r in [('orig', orig_delta(D)), ('d0', 0.0), ('inner', 'inner')] if n in RULES]:
    if rule != 'inner' and rule not in D: print(f'{name}: delta {rule} unavailable (skipped by feature_extraction)'); continue
    try:
        pr, ch = nested(D, y, groups, rule)
    except RuntimeError as err:
        print(f'{name}: {err}', flush=True); rows.append(dict(set=TS, seg=SEG, rule=name, r=np.nan, rmse=np.nan, deltas='unusable', n_deltas_available=len(D))); continue
    rows.append(dict(set=TS, seg=SEG, rule=name, r=stats.pearsonr(y.ravel(), pr)[0], rmse=np.sqrt(((pr - y.ravel()) ** 2).mean()),
                     deltas=';'.join(f'{c:g}' for c in sorted(set(ch))), n_deltas_available=len(D)))
    P[name] = pr; print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/{TS}_{SEG}{SFX}.csv', index=False)
pd.DataFrame(P).to_csv(f'{OUT}/{TS}_{SEG}{SFX}_pred.csv', index=False)
