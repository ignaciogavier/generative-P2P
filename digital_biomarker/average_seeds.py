"""Average the out-of-fold FMA-UE predictions of several detector sets (one per pretraining seed) - Fig. 5, automatic.

Run from digital_biomarker/ after motor_assessment.py. The manual reference uses the same delta rule.
Usage: python average_seeds.py --manual manual --auto local_H0p35_seed0 local_H0p35_seed1 ... --rule inner --name local_H0p35_avg18
Reads output/motor_assessment/<timestamp>_<seg>_pred.csv; writes output/motor_assessment/<name>_preds.csv with
subject, pred_manual, pred_auto (mean over sets), pred_auto_sd, n_seeds -> analysis/fig5_stats.py and analysis/fig5_fig.py.
"""
import os, argparse
import pandas as pd

D = os.path.join('output', 'motor_assessment')


def pred(ts, seg):
    return pd.read_csv(f'{D}/{ts}_{seg}_pred.csv').set_index('subject')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manual', required=True, help='timestamp of the manual (p2p_true) run')
    ap.add_argument('--auto', nargs='+', required=True, help='timestamps of the detector (p2p_pred) runs')
    ap.add_argument('--rule', default='inner', choices=['inner', 'orig', 'd0'])
    ap.add_argument('--name', required=True)
    a = ap.parse_args()
    P = pd.concat({ts: pred(ts, 'p2p_pred')[a.rule] for ts in a.auto}, axis=1)
    assert P.notna().all().all(), 'missing predictions'
    out = pd.DataFrame({'pred_manual': pred(a.manual, 'p2p_true')[a.rule], 'pred_auto': P.mean(1),
                        'pred_auto_sd': P.std(1, ddof=1) if P.shape[1] > 1 else float('nan'), 'n_seeds': P.shape[1]})
    out.index.name = 'subject'; out.to_csv(f'{D}/{a.name}_preds.csv'); print(out.round(2).to_string())
