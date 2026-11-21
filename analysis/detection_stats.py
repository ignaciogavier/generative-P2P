"""Detection statistics for Figs. 3 and 4 from per-fold results (build_per_fold.py).

Per (figure, condition, size): n folds, median, IQR, 95% bootstrap CI of the median (10,000 fold resamples).
Paired comparisons (same seed = same test subject) at each size: Wilcoxon signed-rank, two-sided; Holm correction
within each comparison family (one pair of conditions across all sizes). Condition names used by the families:
  fig3: synth, real, real_aug;  fig4: synth_ft, real10_ft, real14_ft, scratch, realaug1_ft, scratch_aug.
Pairs with a missing condition are skipped.
Usage: python detection_stats.py --curve fig3 synth results/fig3_synth_per_fold.csv --curve fig3 real results/fig3_real_per_fold.csv ...
Outputs (--out_dir): per_fold_all.csv, detection_summary.csv, detection_paired_tests.csv
"""
import os, argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument('--curve', nargs=3, action='append', metavar=('FIGURE', 'CONDITION', 'CSV'), required=True)
ap.add_argument('--out_dir', default='results')
a = ap.parse_args(); OUT = a.out_dir; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261115)
N_BOOT = 10_000
frames = []
for fig, cond, path in a.curve:
    d = pd.read_csv(path)
    n = d['n'] if 'n' in d else d['n_train']
    frames.append(pd.DataFrame(dict(figure=fig, condition=cond, n=n, seed=d.seed, TeUnbAUPRC=d.TeUnbAUPRC)))

D = pd.concat(frames, ignore_index=True)
D['x_total_subj'] = np.where(D.figure == 'fig4', D.n + 4, D.n)  # Fig. 4 x-axis = train + 3 val + 1 test
D.to_csv(f'{OUT}/per_fold_all.csv', index=False)

# ---- Summary ----
rows = []
for (fig, cond, n), g in D.groupby(['figure', 'condition', 'n']):
    v = g.TeUnbAUPRC.to_numpy()
    bs = np.median(v[rng.integers(0, len(v), (N_BOOT, len(v)))], 1)
    rows.append(dict(figure=fig, condition=cond, n_train=n, n_folds=len(v), median=np.median(v),
                     q25=np.quantile(v, .25), q75=np.quantile(v, .75),
                     ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975)))
S = pd.DataFrame(rows)
S.to_csv(f'{OUT}/detection_summary.csv', index=False)

# ---- Paired tests ----
families = {'fig3': [('synth', 'real'), ('synth', 'real_aug'), ('real_aug', 'real')],
            'fig4': [('synth_ft', 'scratch'), ('synth_ft', 'real10_ft'), ('synth_ft', 'real14_ft'), ('synth_ft', 'realaug1_ft'),
                     ('synth_ft', 'scratch_aug'), ('real10_ft', 'scratch'), ('scratch_aug', 'scratch'), ('realaug1_ft', 'scratch')]}
T = []
for fig, pairs in families.items():
    for a, b in pairs:
        A = D[(D.figure == fig) & (D.condition == a)]; B = D[(D.figure == fig) & (D.condition == b)]
        if A.empty or B.empty: continue
        fam = []
        for n in sorted(set(A.n) & set(B.n)):
            m = A[A.n == n].set_index('seed').TeUnbAUPRC.to_frame('a').join(
                B[B.n == n].set_index('seed').TeUnbAUPRC.to_frame('b'), how='inner')
            diff = m.a - m.b
            p = wilcoxon(m.a, m.b).pvalue if (diff != 0).any() else 1.0
            fam.append(dict(figure=fig, cond_a=a, cond_b=b, n_train=n, n_pairs=len(m),
                            median_a=m.a.median(), median_b=m.b.median(), median_diff=diff.median(),
                            frac_a_better=(diff > 0).mean(), p_raw=p))
        fam = pd.DataFrame(fam)
        # Holm within family
        order = np.argsort(fam.p_raw.to_numpy()); k = len(fam)
        adj = np.empty(k); running = 0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (k - rank) * fam.p_raw.iloc[idx])); adj[idx] = running
        fam['p_holm'] = adj
        T.append(fam)
T = pd.concat(T, ignore_index=True)
T.to_csv(f'{OUT}/detection_paired_tests.csv', index=False)

pd.set_option('display.width', 220)
print(S.round(3).to_string(index=False))
print(T.round(4).to_string(index=False))
