"""Statistics of Fig. 5: clinical correlations of the FMA-UE digital biomarker, automatic vs manual segmentation.

Input: per-subject out-of-fold FMA-UE predictions (digital_biomarker/average_seeds.py: subject, pred_manual, pred_auto)
and the clinical scores (digital_biomarker/input/stroke_subject_info.xlsx). As in the paper, the Ridge model predicts
FMA-UE; WMFT-TR, WMFT-FAS, MAL-AoU and MAL-QoM are correlated with the predicted FMA-UE.
  - per score x {manual, auto}: Pearson r with a subject-bootstrap BCa 95% CI (10,000 resamples);
  - auto vs manual (dependent correlations sharing the clinical score): Williams' t, Steiger's Z, bootstrap CI of the
    difference in |r|;
  - FMA-UE estimate quality: RMSE, MAE, R^2 with bootstrap CIs.
Caveat: the bootstrap resamples subjects' out-of-fold predictions; the regression is not refitted per draw.
Usage: python fig5_stats.py --pred_csv ../digital_biomarker/output/motor_assessment/local_H0p35_avg18_preds.csv --out_dir results/fig5
Outputs: fig5_correlations_ci.csv, fig5_auto_vs_manual_tests.csv, fig5_fma_estimate_quality.csv, fig5_summary.json
(fig5_per_subject_predictions.csv holds clinical scores: keep it private).
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy import stats

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'digital_biomarker')
sys.path.insert(0, DB)
from scripts.shimmer_data_loader import load_stroke_demographics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--pred_csv', required=True); ap.add_argument('--out_dir', default='results/fig5')
ap.add_argument('--info', default=os.path.join(DB, 'input', 'stroke_subject_info.xlsx'))
a = ap.parse_args(); OUT = a.out_dir; os.makedirs(OUT, exist_ok=True)
N_BOOT = 10_000
RNG_SEED = 20261115
SCORES = ['FM', 'WMFT_TR', 'WMFT_FAS', 'MAL_AoU', 'MAL_QoM']
LABELS = {'FM': 'FMA-UE', 'WMFT_TR': 'WMFT-TR', 'WMFT_FAS': 'WMFT-FAS', 'MAL_AoU': 'MAL-AoU', 'MAL_QoM': 'MAL-QoM'}

demo = load_stroke_demographics(a.info)
subjects = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
score_df = pd.DataFrame({k: [getattr(demo[s], k) for s in subjects] for k in SCORES}, index=subjects)
y = score_df[['FM']].to_numpy()
_p = pd.read_csv(a.pred_csv).set_index('subject').loc[subjects]
pred = {'manual': _p['pred_manual'].to_numpy(float), 'auto': _p['pred_auto'].to_numpy(float)}
per_subj = score_df.copy()
per_subj['pred_manual'], per_subj['pred_auto'] = pred['manual'], pred['auto']
per_subj.index.name = 'subject'
per_subj.to_csv(f'{OUT}/fig5_per_subject_predictions.csv')

# ---------------- 2-4. Statistics ----------------
n = len(subjects)
rng = np.random.default_rng(RNG_SEED)
boot_idx = rng.integers(0, n, size=(N_BOOT, n))


def r_vec(a, b, idx):
    A, B = a[idx], b[idx]
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))


def jackknife(fn, n):
    return np.array([fn(np.delete(np.arange(n), i)[None, :])[0] for i in range(n)])


def bca_ci(theta_hat, boots, jack, alpha=0.05):
    boots = boots[np.isfinite(boots)]
    z0 = stats.norm.ppf(np.mean(boots < theta_hat) + 0.5 * np.mean(boots == theta_hat))
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5 + 1e-300)
    zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    q = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    return tuple(np.quantile(boots, q))


def stat_with_ci(fn):
    full = fn(np.arange(n)[None, :])[0]
    boots = fn(boot_idx)
    lo, hi = bca_ci(full, boots, jackknife(fn, n))
    plo, phi = np.quantile(boots[np.isfinite(boots)], [0.025, 0.975])
    return full, lo, hi, plo, phi


def williams_t(r12, r13, r23, n):
    """Williams (1959) T2 as in Steiger (1980) eq. 7; H0: rho12 = rho13. df = n - 3."""
    detR = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    rbar = (r12 + r13) / 2
    t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) /
                              (2 * (n - 1) / (n - 3) * detR + rbar ** 2 * (1 - r23) ** 3))
    return t, 2 * stats.t.sf(abs(t), n - 3)


def steiger_z(r12, r13, r23, n):
    """Steiger (1980) Z with pooled r (eq. 14 / Meng-style Fisher z)."""
    z12, z13 = np.arctanh(r12), np.arctanh(r13)
    rbar = (r12 + r13) / 2
    psi = r23 * (1 - 2 * rbar ** 2) - 0.5 * rbar ** 2 * (1 - 2 * rbar ** 2 - r23 ** 2)
    c = psi / (1 - rbar ** 2) ** 2
    Z = (z12 - z13) * np.sqrt(n - 3) / np.sqrt(2 - 2 * c)
    return Z, 2 * stats.norm.sf(abs(Z))


rows, comp = [], []
pa, pm = pred['auto'], pred['manual']
r_am = stats.pearsonr(pa, pm)[0]
for s in SCORES:
    x = score_df[s].to_numpy(dtype=float)
    for src in ['manual', 'auto']:
        p = pred[src]
        full, lo, hi, plo, phi = stat_with_ci(lambda idx, p=p: r_vec(x, p, idx))
        rows.append(dict(score=LABELS[s], source=src, n=n, r=full, ci_bca_lo=lo, ci_bca_hi=hi,
                         ci_pct_lo=plo, ci_pct_hi=phi, p_pearson=stats.pearsonr(x, p)[1]))
    r_a, r_m = stats.pearsonr(x, pa)[0], stats.pearsonr(x, pm)[0]
    # Compare |r| so that the negatively-correlated WMFT-TR is handled consistently
    sign = np.sign(r_a)
    t, pt = williams_t(sign * r_a, sign * r_m, r_am, n)
    Z, pz = steiger_z(sign * r_a, sign * r_m, r_am, n)
    full, lo, hi, plo, phi = stat_with_ci(
        lambda idx: sign * (r_vec(x, pa, idx) - r_vec(x, pm, idx)))
    comp.append(dict(score=LABELS[s], n=n, r_auto=r_a, r_manual=r_m, r_auto_manual=r_am,
                     gain_abs_r=full, gain_ci_bca_lo=lo, gain_ci_bca_hi=hi,
                     williams_t=t, williams_df=n - 3, williams_p=pt, steiger_z=Z, steiger_p=pz))

corr = pd.DataFrame(rows); comp = pd.DataFrame(comp)
corr.to_csv(f'{OUT}/fig5_correlations_ci.csv', index=False)
comp.to_csv(f'{OUT}/fig5_auto_vs_manual_tests.csv', index=False)

qual = []
yf = y.ravel().astype(float)
for src in ['manual', 'auto']:
    p = pred[src]
    fns = {'RMSE': lambda idx, p=p: np.sqrt(((yf[idx] - p[idx]) ** 2).mean(1)),
           'MAE': lambda idx, p=p: np.abs(yf[idx] - p[idx]).mean(1),
           'R2': lambda idx, p=p: 1 - ((yf[idx] - p[idx]) ** 2).sum(1) /
                                  ((yf[idx] - yf[idx].mean(1, keepdims=True)) ** 2).sum(1)}
    for name, fn in fns.items():
        full, lo, hi, plo, phi = stat_with_ci(fn)
        qual.append(dict(source=src, metric=name, n=n, value=full, ci_bca_lo=lo, ci_bca_hi=hi,
                         ci_pct_lo=plo, ci_pct_hi=phi))
qual = pd.DataFrame(qual)
qual.to_csv(f'{OUT}/fig5_fma_estimate_quality.csv', index=False)

gain = comp.gain_abs_r.to_numpy()
summary = dict(n_subjects=n, n_boot=N_BOOT, rng_seed=RNG_SEED,
               mean_gain_abs_r=float(gain.mean()), sd_gain_abs_r=float(gain.std(ddof=1)),
               r_auto_vs_manual_predictions=float(r_am),
               predictions=os.path.abspath(a.pred_csv))
json.dump(summary, open(f'{OUT}/fig5_summary.json', 'w'), indent=2)

pd.set_option('display.width', 200)
print(corr.round(3).to_string()); print(comp.round(3).to_string()); print(qual.round(3).to_string())
print(json.dumps(summary, indent=2))
