"""Synthetic dataset size vs calibrated test AUPRC (healthy LOSO, S_tr = 1, 3, 5 source subjects).

Inputs: per-fold CSV of the size runs (build_per_fold.py --by_synth_size; synth_size 0.2-0.8 = 4k-16k trajectories) and
the full-size (20k) Fig. 3 per-fold CSV. Paired Wilcoxon vs 20k, Holm over sizes within each S_tr.
Usage: python dataset_size.py --sizes results/size_per_fold.csv --full results/synth_per_fold.csv --out_dir results
Outputs: dataset_size_per_fold.csv, dataset_size_summary.csv
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', required=True); ap.add_argument('--full', required=True); ap.add_argument('--out_dir', default='results')
    a = ap.parse_args()
    sel = pd.read_csv(a.sizes)
    full = pd.read_csv(a.full); full = full[full.n.isin([1, 3, 5])].assign(synth_size=1.0)
    P = pd.concat([sel, full])[['synth_size', 'n', 'seed', 'TeUnbAUPRC']]
    P['n_trajectories'] = (P.synth_size * 20000).round().astype(int)
    P.sort_values(['synth_size', 'n', 'seed']).to_csv(f'{a.out_dir}/dataset_size_per_fold.csv', index=False)
    rows = []
    for n in (1, 3, 5):
        ref = P[(P.n == n) & (P.synth_size == 1.0)].set_index('seed').TeUnbAUPRC
        block = []
        for sz in sorted(P.synth_size.unique()):
            x = P[(P.n == n) & (P.synth_size == sz)].set_index('seed').TeUnbAUPRC.loc[ref.index]
            block.append(dict(n_source=n, trajectories=int(sz * 20000), median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75),
                              diff_vs_20k=(x - ref).median(), p=np.nan if sz == 1.0 else stats.wilcoxon(x, ref).pvalue))
        ps = [b['p'] for b in block if not np.isnan(b['p'])]; ph = iter(multipletests(ps, method='holm')[1])
        for b in block: b['p_holm'] = np.nan if np.isnan(b['p']) else next(ph)
        rows += block
    S = pd.DataFrame(rows); S.to_csv(f'{a.out_dir}/dataset_size_summary.csv', index=False)
    pd.set_option('display.width', 200); print(S.round(4).to_string(index=False))
    print(S.pivot(index='trajectories', columns='n_source', values='median').round(3))
