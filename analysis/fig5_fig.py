"""Fig. 5: FMA-UE digital biomarker vs the five clinical scores; top row manual segmentation, bottom row P2P detectors.

Input: per-subject predictions (digital_biomarker/average_seeds.py) and the clinical scores.
The regression lines' 95% bands are bootstrapped by seaborn (seed fixed at 0).
--errbars adds the per-subject SD of the predicted FMA-UE across detector seeds to the bottom row.
Usage: python fig5_fig.py --pred_csv ../digital_biomarker/output/motor_assessment/local_H0p35_avg18_preds.csv --out results/fig5
       -> results/fig5.{svg,pdf,png}
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.stats import pearsonr

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'digital_biomarker')
sys.path.insert(0, DB)
from scripts.shimmer_data_loader import load_stroke_demographics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--pred_csv', required=True); ap.add_argument('--out', default='results/fig5')
ap.add_argument('--info', default=os.path.join(DB, 'input', 'stroke_subject_info.xlsx')); ap.add_argument('--errbars', action='store_true')
a = ap.parse_args(); ERR = a.errbars
plt.rcParams.update({'figure.dpi': 150, 'mathtext.fontset': 'stix', 'font.family': 'sans-serif',
                     'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
P = pd.read_csv(a.pred_csv).set_index('subject')
demo = load_stroke_demographics(a.info)
subjects = list(P.index)
SCORES = ['FM', 'WMFT_TR', 'WMFT_FAS', 'MAL_AoU', 'MAL_QoM']
score_df = pd.DataFrame({k: [getattr(demo[s], k) for s in subjects] for k in SCORES}, index=subjects)
y_pred_manual, y_pred_auto = P[['pred_manual']].to_numpy(), P[['pred_auto']].to_numpy()

# ---- figure (as in the paper) ----
fig, ax = plt.subplots(2, 5, figsize=(10, 4), sharey=True, sharex='col')
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.97, hspace=0.2)
for s, score in enumerate(SCORES):
    axi = ax[:, s]
    dFAll = pd.DataFrame(np.block([[score_df[[score]].to_numpy(), y_pred_manual, np.zeros_like(y_pred_manual)],
                                   [score_df[[score]].to_numpy(), y_pred_auto, np.ones_like(y_pred_manual)]]),
                         columns=[score, 'predictedFM', 'is_auto'])
    if ERR:
        axi[1].errorbar(score_df[score], P['pred_auto'], yerr=P['pred_auto_sd'], fmt='none',
                        ecolor=sn.color_palette('tab10')[0], alpha=0.35, lw=0.8, zorder=1)
    score = 'FMA_UE' if score == 'FM' else score
    dFAll = dFAll.rename(columns={'FM': score})
    legend_labels = []
    for group, data in dFAll.groupby('is_auto'):
        sn.scatterplot(data=data, x=score, y='predictedFM', color=sn.color_palette('tab10')[int(1 - group)], label=group, ax=axi[int(group)])
        sn.regplot(data=data, x=score, y='predictedFM', scatter=False, truncate=False, color=sn.color_palette('tab10')[int(1 - group)], ax=axi[int(group)], seed=0)
        r, _ = pearsonr(data[score], data['predictedFM'])
        legend_labels.append(f'r = {r:.3f}')
    axi[1].set_xlabel(score.replace('_', '-'))
    axi[0].grid(); axi[1].grid()
    print(score, *legend_labels)
    axi[0].legend(handles=axi[0].get_legend_handles_labels()[0], labels=legend_labels[:1])
    axi[1].legend(handles=axi[1].get_legend_handles_labels()[0], labels=legend_labels[1:])
ax[0, 0].set_ylabel('Digital Biomarker\nUsing Manual Annotations')
ax[1, 0].set_ylabel('Digital Biomarker\nUsing P2P Detectors')
ax[0, 0].set_ylim(25, 75)
out = a.out
for f in ['svg', 'pdf', 'png']:
    fig.savefig(f'{out}.{f}', dpi=150 if f != 'png' else 300)
print('saved', out)
