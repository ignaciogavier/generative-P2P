"""Ablation table: each variant vs the full pipeline, paired by LOSO fold.

Grids: healthy S_tr = 1, 3, 5 (Fig. 3 protocol) and stroke fine-tuning on n = 5 subjects (Fig. 4 protocol).
Per variant: median [IQR], median of the full pipeline, median paired difference, two-sided Wilcoxon signed-rank,
Holm over the 4 grids within each variant.
Inputs: per-fold CSVs (build_per_fold.py); <prefix>_per_fold.csv (healthy) and <prefix>_ft_per_fold.csv (stroke).
Usage: python ablation_table.py --full results/synth --variant A=results/A_no_resampling --variant B=results/B_coeff_kde \\
           --variant orig=results/std_H0p05 --variant C=results/fixed_sensor --out results/ablation_table.csv
"""
import argparse
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def get(prefix, ft, n):
    d = pd.read_csv(f'{prefix}{"_ft" if ft else ""}_per_fold.csv')
    return d[d.n == n].set_index('seed')['TeUnbAUPRC']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', required=True); ap.add_argument('--variant', action='append', required=True, help='name=prefix')
    ap.add_argument('--out', default='results/ablation_table.csv')
    a = ap.parse_args(); FULL = a.full; VAR = dict(v.split('=', 1) for v in a.variant)
    rows = []
    for v, pre in VAR.items():
        block = []
        for ft, n, label in [(False, 1, 'healthy S_tr=1'), (False, 3, 'healthy S_tr=3'), (False, 5, 'healthy S_tr=5'), (True, 5, 'stroke ft n=5')]:
            x, f = get(pre, ft, n), get(FULL, ft, n)
            idx = x.index.intersection(f.index); x, f = x.loc[idx], f.loc[idx]
            block.append(dict(variant=v, grid=label, n_folds=len(idx), median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75),
                              full_median=f.median(), median_diff=(x - f).median(), p=stats.wilcoxon(x, f).pvalue))
        ph = multipletests([b['p'] for b in block], method='holm')[1]
        for b, p in zip(block, ph): b['p_holm'] = p
        rows += block
    T = pd.DataFrame(rows); T.to_csv(a.out, index=False)
    pd.set_option('display.width', 200); print(T.round(4).to_string(index=False))
