"""Analyse the StrokeRehab data-efficiency sweep (finetune_sweep.py, 180 tasks).
Epoch rule (paper): max VaUnbAUPRC, tie -> max TrUnbAUPRC; report TeUnbAUPRC at that epoch.
Paired tests on fold (same test subject, same train/val split, same seed across conditions): Wilcoxon signed-rank,
Holm within each comparison family (across the 4 values of N); median difference with 10,000-sample bootstrap 95% CI.
Outputs (<STROKEREHAB_OUT>): sweep_per_fold.csv, sweep_summary.csv, sweep_paired_tests.csv, strokerehab.{pdf,svg,png}
"""
import os, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sn

OUT = os.environ.get('STROKEREHAB_OUT', 'results/strokerehab')
E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{OUT}/sweep/*.csv')) if 'smoke' not in os.path.basename(f)])
assert len(E) == 180 * 30, len(E)
P = (E.sort_values(['VaUnbAUPRC', 'TrUnbAUPRC'], ascending=False).groupby(['cond', 'N', 'fold']).head(1)
     .sort_values(['cond', 'N', 'fold']).reset_index(drop=True))
P.to_csv(f'{OUT}/sweep_per_fold.csv', index=False)

S = P.groupby(['cond', 'N']).TeUnbAUPRC.agg(['median', 'mean', 'std', lambda x: x.quantile(.25), lambda x: x.quantile(.75), 'count'])
S.columns = ['median', 'mean', 'std', 'q25', 'q75', 'n_folds']; S = S.reset_index(); S.to_csv(f'{OUT}/sweep_summary.csv', index=False)
print(S.round(3).to_string(index=False))

rng = np.random.default_rng(0)
def boot(d, B=10000):
    m = np.median(d[rng.integers(0, len(d), (B, len(d)))], 1); return np.percentile(m, [2.5, 97.5])

W = P.pivot_table(index=['N', 'fold'], columns='cond', values='TeUnbAUPRC')
rows = []
for a, b in [('synth1', 'scratch'), ('real10', 'scratch'), ('synth1', 'real10')]:
    fam = []
    for N in sorted(P.N.unique()):
        d = (W.loc[N, a] - W.loc[N, b]).to_numpy(); lo, hi = boot(d)
        fam.append(dict(comparison=f'{a} - {b}', N=N, n=len(d), median_diff=np.median(d), ci_lo=lo, ci_hi=hi,
                        wins=int((d > 0).sum()), p=wilcoxon(d).pvalue))
    ps = np.array([r['p'] for r in fam]); o = np.argsort(ps); adj = np.empty_like(ps)
    adj[o] = np.minimum(1, np.maximum.accumulate(ps[o] * (len(ps) - np.arange(len(ps)))))
    for r, q in zip(fam, adj): r['p_holm'] = q
    rows += fam
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/sweep_paired_tests.csv', index=False)
print(T.round(4).to_string(index=False))

# ---- figure, paper style (as FIGS/make_figs.py) ----
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans'],
                     'mathtext.fontset': 'cm', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
TAB = sn.color_palette('tab10'); order = ['synth1', 'real10', 'scratch']; col = [TAB[0], TAB[1], TAB[2]]
fig, ax = plt.subplots(figsize=(5, 3)); fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.95)
d = P.assign(cond=pd.Categorical(P.cond, order))
sn.boxplot(d, x='N', y='TeUnbAUPRC', hue='cond', hue_order=order, palette=col, gap=0.1, ax=ax,
           flierprops=dict(marker='o', markerfacecolor='none'), legend=False)
ax.grid(); ax.set_axisbelow(True); ax.set_ylim(0.3, 1); ax.set_ylabel('AUPRC')
ax.set_xlabel(r'$\tilde{\mathbfit{S}}_{\mathbf{tr}}$(Total StrokeRehab subjects used for training)')
ax.legend([Patch(facecolor=c, edgecolor='0.25', lw=1.2) for c in col],
          ['PT on synthetic movements (one subject), FT on StrokeRehab', 'PT on real healthy movements (ten subjects), FT on StrokeRehab',
           'Train on StrokeRehab movements'], loc='lower right', fontsize=7)
for ext in ('pdf', 'svg', 'png'): fig.savefig(f'{OUT}/strokerehab.{ext}', dpi=300)
print('done')
