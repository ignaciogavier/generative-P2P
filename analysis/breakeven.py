"""Break-even points behind the annotation-saving arrows of Figs. 3 and 4.

Break-even of comparator B against reference A at size n0:
  f(N) = median over LOSO folds of [B(N) - A(n0)]   (paired by fold),
  N* = first N where f crosses 0 from below, linear interpolation between N-1 and N.
  N* <= min(N) if f(min) >= 0; censored ('> max') if f never reaches 0.
95% CI: 10,000 bootstrap resamples of folds (same resampled folds for every N), percentile; censored draws count as
+inf, and the share of censored draws is reported.
Hours: healthy subject = 30 min recording x 30 min annotation per minute = 15 h; stroke subject = 50 min x 30 = 25 h.
Input: per_fold_all.csv (detection_stats.py). Output: <out_dir>/breakeven.csv
"""
import os
import numpy as np, pandas as pd

import sys
W = sys.argv[1] if len(sys.argv) > 1 else 'results'
D = pd.read_csv(f'{W}/per_fold_all.csv')
NB, SEED = 10_000, 20261121
H = {'fig3': 15.0, 'fig4': 25.0}


def table(fig, cond):
    return D[(D.figure == fig) & (D.condition == cond)].pivot(index='seed', columns='n', values='TeUnbAUPRC').sort_index()


def crossing(f, Ns):
    if f[0] >= 0: return Ns[0]
    for i in range(1, len(f)):
        if f[i] >= 0: return Ns[i - 1] + (-f[i - 1]) / (f[i] - f[i - 1]) * (Ns[i] - Ns[i - 1])
    return np.inf


def breakeven(fig, ref, n0, cmp):
    A, B = table(fig, ref), table(fig, cmp)
    folds = A.index.intersection(B.index); A, B = A.loc[folds], B.loc[folds]
    Ns = np.array(B.columns); a = A[n0].to_numpy(); b = B.to_numpy()
    est = crossing(np.median(b - a[:, None], 0), Ns)
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(folds), (NB, len(folds)))
    boots = np.array([crossing(np.median(b[i] - a[i, None], 0), Ns) for i in idx])
    lo, hi = np.quantile(boots, [0.025, 0.975], method='inverted_cdf')   # no interpolation: inf stays inf (= '> max_N')
    return dict(figure=fig, reference=f'{ref}({n0})', comparator=cmp, n_folds=len(folds), breakeven=est, ci_lo=lo, ci_hi=hi,
                share_censored=float(np.isinf(boots).mean()), max_N=Ns.max(),
                subjects_saved=est - n0 if np.isfinite(est) else np.nan,
                hours_saved=(est - n0) * H[fig] if np.isfinite(est) else np.nan)


rows = [breakeven('fig3', 'synth', 1, 'real'), breakeven('fig3', 'synth', 1, 'real_aug')]
for n0 in range(1, 8):
    for cmp in ['scratch', 'real10_ft', 'real14_ft', 'scratch_aug', 'realaug1_ft']:
        if cmp in set(D.condition): rows.append(breakeven('fig4', 'synth_ft', n0, cmp))
R = pd.DataFrame(rows)
R.to_csv(f'{W}/breakeven.csv', index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
print(R.round(2).to_string(index=False))
