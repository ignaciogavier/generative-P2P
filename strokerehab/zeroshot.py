"""zero-shot: detectors trained only on our HEALTHY data, applied unchanged to StrokeRehab (15 subjects, hand IMU,
gravity restored, axes aligned label-free). No StrokeRehab data is used for training, selection or thresholds.

Detector families (18 seeds each, saved models): synth1 (synthetic from 1 subject, = Fig. 4 synth pretraining),
real1, real10, real14 (real healthy), real1aug (real 1 subject + Um et al. 2017 augmentation).
Metric: calibrated AUPRC (paper formula) per (family, seed, subject), plus pooled over all subjects.
Output: <STROKEREHAB_OUT>/zeroshot_per_subject.csv, zeroshot_summary.csv
"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, OUT, SUBJECTS, load_subject, build_model, auprc_unb, predict

HSEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 18, 22, 26, 28, 35, 44, 62]
FAMILIES = {'synth1': 'synth1_seed{}.pth', 'real1': 'train_1real_seed{}.pth', 'real10': 'train_10real_seed{}.pth',
            'real14': 'train_14real_seed{}.pth', 'real1aug': 'train_1realaug_seed{}.pth'}   # in P2P_MODELS

if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = {s: load_subject(s) for s in SUBJECTS}
    Xall = np.concatenate([data[s][0] for s in SUBJECTS]); yall = np.concatenate([data[s][1] for s in SUBJECTS])
    sid = np.concatenate([[s] * len(data[s][1]) for s in SUBJECTS])
    seeds = HSEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else HSEEDS
    rows = []
    for fam, pat in FAMILIES.items():
        for seed in seeds:
            m = build_model(dev); m.load_state_dict(torch.load(f'{MODELS}/{pat.format(seed)}', map_location=dev))
            p = predict(m, Xall, dev)
            for s in SUBJECTS:
                k = sid == s
                rows.append(dict(family=fam, seed=seed, subject=s, n=int(k.sum()), pos_rate=float(yall[k].mean()),
                                 auprc_unb=auprc_unb(yall[k], p[k])))
            rows.append(dict(family=fam, seed=seed, subject='pooled', n=len(yall), pos_rate=float(yall.mean()),
                             auprc_unb=auprc_unb(yall, p)))
            print(fam, seed, round(rows[-1]['auprc_unb'], 3), flush=True)
    d = pd.DataFrame(rows)
    sfx = os.environ.get('STROKEREHAB_WINDOWS', 'windows_hand').replace('windows_hand', '')   # '' for the main analysis
    out = f'zeroshot_per_subject{sfx}.csv' if len(sys.argv) == 1 else f'zeroshot_smoke{sfx}.csv'
    d.to_csv(f'{OUT}/{out}', index=False)
    ps = d[d.subject != 'pooled'].groupby(['family', 'seed']).auprc_unb.median().rename('median_over_subjects')
    S = pd.concat([d[d.subject == 'pooled'].set_index(['family', 'seed']).auprc_unb.rename('pooled'), ps], axis=1).reset_index()
    S = S.groupby('family')[['pooled', 'median_over_subjects']].agg(['median', lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
    print(S.round(3).to_string())
    if len(sys.argv) == 1: S.to_csv(f'{OUT}/zeroshot_summary{sfx}.csv')
