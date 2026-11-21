"""StrokeRehab zero-shot: synth1 vs the other detector families, per subject.
Per subject: median calibrated AUPRC over the 18 seeds; then paired over the 15 subjects: median difference, wins,
two-sided Wilcoxon. Input/output in <STROKEREHAB_OUT>: zeroshot_per_subject.csv -> zeroshot_compare.csv
"""
import os
import pandas as pd
from scipy.stats import wilcoxon

OUT = os.environ.get('STROKEREHAB_OUT', 'results/strokerehab'); FAM = 'synth1'
Z = pd.read_csv(f'{OUT}/zeroshot_per_subject.csv')
M = Z[Z.subject != 'pooled'].groupby(['family', 'subject']).auprc_unb.median().unstack('family')   # 'pooled' row = all subjects together
rows = []
for other in ['real1', 'real1aug', 'real10', 'real14']:
    if other not in M: continue
    d = M[FAM] - M[other]
    rows.append(dict(family=FAM, vs=other, n_subjects=len(d), median_diff=d.median(), wins=int((d > 0).sum()), p=wilcoxon(d).pvalue))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/zeroshot_compare.csv', index=False); print(T.round(4).to_string(index=False))
