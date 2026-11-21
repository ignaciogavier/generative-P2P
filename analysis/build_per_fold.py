"""Per-fold results of one curve from the pulled WandB runs (pull_wandb.py), with the paper's selection rule:
per (size, seed) keep the run with most epochs (>= 30, i.e. completed), take the epoch with the highest calibrated
validation AUPRC (tie: training AUPRC) and report the calibrated test AUPRC (TeUnbAUPRC) at that epoch.

--kind healthy  runs validated/tested on real healthy subjects (Fig. 3; size = synthetic or real training subjects)
--kind stroke   runs with 3 validation + 1 test stroke subjects (Fig. 4; size = stroke training subjects)
--by_synth_size adds synth_size to the grouping (dataset-size experiment).
Usage: python build_per_fold.py --runs_dir results/wandb --tag fig3_synth --kind healthy --out results/fig3_synth_per_fold.csv
Output columns: id, epoch, metrics..., n, seed[, synth_size], curve. Missing (size, seed) cells are printed.
"""
import argparse
import numpy as np, pandas as pd

HSEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 18, 22, 26, 28, 35, 44, 62]
SSEEDS = [22, 2, 8, 18, 26, 3, 1, 6, 5, 13, 62, 7, 35, 10, 4, 0, 28]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', default='results/wandb'); ap.add_argument('--tag', required=True)
    ap.add_argument('--kind', choices=['healthy', 'stroke'], required=True)
    ap.add_argument('--by_synth_size', action='store_true'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    runs = pd.read_csv(f'{a.runs_dir}/runs.csv'); ep = pd.read_csv(f'{a.runs_dir}/per_epoch.csv')
    for c in ['rSize', 'strSize', 'sSize']: runs[c] = runs[c].apply(eval)
    runs = runs[runs.tags.fillna('').str.split('|').apply(lambda t: a.tag in t)]
    if a.kind == 'stroke':
        runs = runs[runs.strSize.apply(lambda x: tuple(x[1:]) == (3, 1))]; runs['n'] = runs.strSize.apply(lambda x: x[0]); seeds = SSEEDS
    else:
        runs = runs[runs.rSize.apply(lambda x: tuple(x[1:]) == (3, 1)) & (runs.strSize.apply(sum) == 0)]
        runs['n'] = runs.apply(lambda r: r.sSize[0] or r.rSize[0], axis=1); seeds = HSEEDS
    keys = ['n', 'seed'] + (['synth_size'] if a.by_synth_size else [])
    runs = runs[runs.seed.isin(seeds)].merge(ep.groupby('id').epoch.count().rename('n_epochs'), left_on='id', right_index=True)
    runs = runs[runs.n_epochs >= 30].sort_values('n_epochs', ascending=False).groupby(keys).head(1)
    d = ep[ep.id.isin(runs.id)].merge(runs[['id'] + keys], on='id')
    sel = (d.sort_values(keys + ['VaUnbAUPRC', 'TrUnbAUPRC'], ascending=[True] * len(keys) + [False, False])
           .groupby(keys, as_index=False).head(1))
    sel.assign(curve=a.tag).to_csv(a.out, index=False)
    full = pd.MultiIndex.from_product([sorted(sel.n.unique()), seeds], names=['n', 'seed'])
    miss = full.difference(pd.MultiIndex.from_frame(sel[['n', 'seed']]))
    print(f'{a.tag}: {len(sel)} folds -> {a.out}' + (f'; missing (n, seed): {list(miss)}' if len(miss) and not a.by_synth_size else ''))
