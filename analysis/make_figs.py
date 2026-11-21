"""Figs. 3 and 4 (seaborn boxplots per training-set size, paper style) with the annotation-saving arrows.

Input (results dir): per_fold_all.csv (detection_stats.py) and breakeven.csv (breakeven.py).
Fig. 3: synth, real, real_aug.  Fig. 4: (a) synth_ft, real10_ft, scratch; (b) synth_ft, realaug1_ft, scratch, scratch_aug.
Fig. 4 arrows start at ARROW_N0 = 4 stroke subjects. Augmentation baselines are hatched.
Usage: python make_figs.py [results_dir] -> <results_dir>/fig3.{pdf,svg,png}, fig4.{pdf,svg,png}
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sn

W = sys.argv[1] if len(sys.argv) > 1 else 'results'
D = pd.read_csv(f'{W}/per_fold_all.csv')
ARROW_N0 = 4
_BE = pd.read_csv(f'{W}/breakeven.csv').set_index(['reference', 'comparator'])


def _be(ref, cmp):
    r = _BE.loc[(ref, cmp)]
    assert np.isfinite(r.breakeven), f'{ref} vs {cmp}: no break-even within {r.max_N}'
    return r.breakeven, 5 * round(r.hours_saved / 5)


plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans'],
                     'mathtext.fontset': 'cm', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
TAB = sn.color_palette('tab10')
HATCH = '////'


def boxes(ax, d, order, colors, hatched, nmax):
    d = d.assign(cond=pd.Categorical(d.condition, order), x=pd.Categorical(d.n, range(1, nmax + 1)))
    sn.boxplot(d, x='x', y='TeUnbAUPRC', hue='cond', hue_order=order, palette=colors, gap=0.1, ax=ax,
               flierprops=dict(marker='o', markerfacecolor='none'), legend=False)
    # hatch the augmentation boxes (patches are drawn hue by hue)
    k = 0
    per_hue = [d[d.condition == c].n.nunique() for c in order]
    patches = [p for p in ax.patches if isinstance(p, matplotlib.patches.PathPatch)]
    for c, m in zip(order, per_hue):
        for p in patches[k:k + m]:
            if c in hatched: p.set_hatch(HATCH); p.set_edgecolor('0.25')
        k += m
    ax.grid(); ax.set_axisbelow(True); ax.set_ylim(0.5, 1)
    ax.set_xticks(range(0, nmax, 2)); ax.set_xticklabels(range(1, nmax + 1, 2)); ax.set_xlim(-0.5, nmax - 0.5)
    ax.set_ylabel('AUPRC')


def legend(ax, labels, colors, hatched_flags, **kw):
    h = [Patch(facecolor=c, edgecolor='0.25', hatch=HATCH if hf else None, lw=1.2) for c, hf in zip(colors, hatched_flags)]
    ax.legend(h, labels, loc='lower right', **kw)


def arrow(ax, x0, x1, y, text):
    ax.annotate('', xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))
    ax.plot([x1, x1], [y - .008, y + .008], color='#d62728', lw=1.2)
    ax.text((x0 + x1) / 2, y + .004, text, color='#d62728', ha='center', va='bottom', fontsize=9)


def save(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(f'{W}/{name}.{ext}', dpi=300)
    plt.close(fig)


# ---------------- Fig. 3 ----------------
f3 = D[D.figure == 'fig3']
order3 = ['synth', 'real', 'real_aug']; col3 = [TAB[0], TAB[1], TAB[4]]
fig, ax = plt.subplots(figsize=(5, 3)); fig.subplots_adjust(left=0.12, right=0.97, bottom=0.16, top=0.95)
boxes(ax, f3, order3, col3, {'real_aug'}, 14)
ax.set_xlabel(r'$\mathbfit{S}_{\mathbf{tr}}$(Total subjects used for training)')
legend(ax, ['Train on synthetic movements', 'Train on real movements', 'Train on real movements + augmentation'],
       col3, [0, 0, 1])
_n, _h = _be('synth(1)', 'real'); arrow(ax, 0, _n - 1, 0.96, f'~{_h:.0f} hours of annotation reduction')
save(fig, 'fig3')

# ---------------- Fig. 4: (a) pretraining comparators; (b) augmentation baselines ----------------
f4 = D[D.figure == 'fig4']
XL4 = r'$\tilde{\mathbfit{S}}_{\mathbf{tr}}$(Total stroke subjects used for training)'
fig, axs = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.975, hspace=0.08)
order_a = ['synth_ft', 'real10_ft', 'scratch']; col_a = [TAB[0], TAB[1], TAB[2]]
boxes(axs[0], f4, order_a, col_a, set(), 13)
legend(axs[0], ['PT on synthetic movements (one subject), FT on stroke', 'PT on real healthy movements (ten subjects), FT on stroke',
                'Train on real stroke movements'], col_a, [0, 0, 0])
_n, _h = _be(f'synth_ft({ARROW_N0})', 'scratch'); arrow(axs[0], ARROW_N0 - 1, _n - 1, 0.966, f'~{_h:.0f} hours of annotation reduction')
_n, _h = _be(f'synth_ft({ARROW_N0})', 'real10_ft'); arrow(axs[0], ARROW_N0 - 1, _n - 1, 0.928, f'~{_h:.0f} hours of annot. red.')
order_b = ['synth_ft', 'realaug1_ft', 'scratch', 'scratch_aug']; col_b = [TAB[0], TAB[4], TAB[2], TAB[5]]
boxes(axs[1], f4, order_b, col_b, {'realaug1_ft', 'scratch_aug'}, 13)
legend(axs[1], ['PT on synthetic movements (one subject), FT on stroke', 'PT on real healthy movements + aug. (one subject), FT on stroke',
                'Train on real stroke movements', 'Train on real stroke movements + augmentation'], col_b, [0, 1, 0, 1], fontsize=8)
axs[0].set_xlabel(''); axs[1].set_xlabel(XL4)
for a, t in zip(axs, ['(a)', '(b)']):
    fig.text(0.005, a.get_position().y1, t, ha='left', va='top', fontsize=10)
save(fig, 'fig4')

print('done')
