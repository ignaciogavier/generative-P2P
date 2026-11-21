"""Download the training runs logged by p2p_classifier/P2P_classifier.py from Weights & Biases (read-only).

Writes to --out_dir: runs.csv (one row per run: id, state, seed, splits, pretrained model, synth_dir, synth_size, tags)
and per_epoch.csv (one row per run and epoch: calibrated / plain AUPRC and F1 on train, val, test). Resumes from an
existing per_epoch.csv. Per-fold results: build_per_fold.py.
Usage: python pull_wandb.py --entity <entity> --project P2P-classifier [--tags fig3_synth fig4_synth ...] [--since 2026-01-01]
"""
import os, time, argparse
import numpy as np, pandas as pd
os.environ['WANDB_SILENT'] = 'True'
import wandb

KEYS = ['train_auprc_unb', 'val_auprc_unb', 'test_auprc_unb', 'train_f1_unb', 'val_f1_unb', 'test_f1_unb',
        'train_auprc', 'val_auprc', 'test_auprc']
COLS = ['TrUnbAUPRC', 'VaUnbAUPRC', 'TeUnbAUPRC', 'TrUnbF1', 'VaUnbF1', 'TeUnbF1', 'TrAUPRC', 'VaAUPRC', 'TeAUPRC']


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--entity', required=True); ap.add_argument('--project', default='P2P-classifier')
    ap.add_argument('--tags', nargs='*', default=[], help='only runs carrying one of these tags')
    ap.add_argument('--since', default=None, help='only runs created after this date (YYYY-MM-DD)')
    ap.add_argument('--out_dir', default='results/wandb')
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    F = []
    if a.since: F.append({'created_at': {'$gt': a.since}})
    if a.tags: F.append({'tags': {'$in': a.tags}})
    runs = wandb.Api(timeout=300).runs(f'{a.entity}/{a.project}', filters={'$and': F} if F else {}, order='-created_at', per_page=500)
    meta = []
    for i, r in enumerate(runs):
        c = r.config
        if not all(k in c for k in ['real_splits', 'stroke_splits', 'synth_splits']): continue
        meta.append(dict(id=r.id, created=r.created_at, state=r.state, seed=c.get('random_seed'),
                         rSize=tuple(c['real_splits']), strSize=tuple(c['stroke_splits']), sSize=tuple(c['synth_splits']),
                         pretrained=c.get('pretrained'), ptm=tuple(c.get('pretrained_model') or (0, 0, 0)),
                         pretr_path=c.get('pretr_path'), aug=c.get('aug'), synth_dir=c.get('synth_dir'), synth_size=c.get('synth_size'),
                         steps=r.summary.get('_step', -1), tags='|'.join(r.tags), _run=r))
        if i % 200 == 0: log('listed', i)
    log('runs with split config:', len(meta))
    new = pd.DataFrame([{k: v for k, v in m.items() if k != '_run'} for m in meta])
    rf = f'{a.out_dir}/runs.csv'
    if os.path.exists(rf):                                        # keep runs pulled before, refresh the listed ones
        old = pd.read_csv(rf); new = pd.concat([old[~old.id.isin(new.id)], new])
    new.to_csv(rf, index=False)
    ep = f'{a.out_dir}/per_epoch.csv'
    done = set(pd.read_csv(ep).id) if os.path.exists(ep) else set()
    for j, m in enumerate(meta):
        if m['id'] in done or m['steps'] < 0 or m['state'] != 'finished': continue
        for attempt in range(3):
            try:
                h = m['_run'].history(keys=KEYS, samples=1000, pandas=True); break
            except Exception as e:
                log('retry', m['id'], e); time.sleep(10)
        else:
            log('FAILED', m['id']); continue
        if h is None or len(h) == 0: continue
        h = h.reset_index(drop=True)
        df = pd.DataFrame({'id': m['id'], 'epoch': np.arange(len(h))})
        for k, col in zip(KEYS, COLS): df[col] = h[k].values if k in h else np.nan
        df.to_csv(ep, mode='a', header=not os.path.exists(ep), index=False)
        if j % 50 == 0: log(f'history {j}/{len(meta)}')
    log('history done')
