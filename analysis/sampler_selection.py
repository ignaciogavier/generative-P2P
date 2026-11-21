"""Choice of the Phase I sampler and bandwidth without any detector (test-free), per healthy subject, P2P movements:
  ll_per_dim    5-fold held-out log-likelihood per dimension (common standardized space, Jacobians included)
  novelty       median NN distance from samples to the subject's real movements / median leave-one-out NN distance among
                real movements (0 = copies of real movements, 1 = as different as real movements are from each other)
  plausibility  share of 10,000 samples per subject outside the [0.5, 99.5]th percentiles of the REAL movements pooled
                over subjects (duration, displacement, peak speed, peak acceleration, reach); real data: 4.0%
Selection rule (fixed before any detector was trained): highest novelty with plausibility violations <= 6%
(real + 2 points), ties -> held-out likelihood. Result: local KDE, h = 0.35.
Usage: python sampler_selection.py --feat_dir ../generative_pipeline/feature_extraction --out_dir results/sampler  (multi-core)
Outputs: plausibility_limits.csv, sampler_selection_subject.csv, sampler_selection.csv, sampler_frontier.png
"""
import os, sys, argparse
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.spatial import cKDTree
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'generative_pipeline'))
import kde_samplers as ks  # noqa: E402

MET = ['T', 'disp', 'vpeak', 'apeak', 'reach']
FEAT, LIM = None, None


def init(feat, lim):
    global FEAT, LIM
    FEAT, LIM = feat, lim


def limits(feat):
    real = pd.concat([ks.kinematics(ks.load_features(feat, s, 'p2p')) for s in ks.SUBJECTS])
    lim = real.quantile([0.005, 0.995]).T.rename(columns={0.005: 'lo', 0.995: 'hi'})
    lim.loc['T', 'lo'] = real['T'].quantile(0.005)
    return lim


def per_subject(s):
    Z = ks.load_features(FEAT, s, 'p2p'); mu, sd = Z.mean(0), Z.std(0); Zs = (Z - mu) / sd
    tree = cKDTree(Zs); dreal = np.median(tree.query(Zs, 2)[0][:, 1])
    rows, kin = [], []
    for tf in ks.TFS:
        for h in ks.HS:
            ll = []
            for tr, te in KFold(5, shuffle=True, random_state=0).split(Zs):
                m = ks.Model(tf, h); m.mu, m.sd = mu, sd; m.fit_s(Zs[tr]); ll.append(m.logpdf_s(Zs[te]))
            m = ks.Model(tf, h).fit(Z); rng = np.random.default_rng(1000 * s + 17)
            S = m.sample(10000, rng)
            K = ks.kinematics(S)
            V = pd.concat([(K[c] < LIM.loc[c, 'lo']) | (K[c] > LIM.loc[c, 'hi']) for c in MET], axis=1)
            r = dict(transform=tf, h=h, subject=s, ll_per_dim=float(np.mean(np.concatenate(ll)) / Z.shape[1]),
                     novelty=float(np.median(tree.query((S[:2000] - mu) / sd)[0]) / dreal), any=float(V.any(axis=1).mean()))
            r.update({c: float(V[c].mean()) for c in MET}); rows.append(r)
            kin.append(pd.DataFrame(dict(transform=tf, h=h, apeak=K.apeak.to_numpy()[:1000], vpeak=K.vpeak.to_numpy()[:1000])))
    print('subject', s, 'done', flush=True)
    return rows, pd.concat(kin)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--feat_dir', required=True); ap.add_argument('--out_dir', default='results/sampler')
    a = ap.parse_args(); HERE = a.out_dir; os.makedirs(HERE, exist_ok=True)
    lim = limits(a.feat_dir); lim.to_csv(f'{HERE}/plausibility_limits.csv')
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1)), initializer=init, initargs=(a.feat_dir, lim)) as p:
        out = p.map(per_subject, ks.SUBJECTS)
    R = pd.DataFrame(sum([o[0] for o in out], [])); R.to_csv(f'{HERE}/sampler_selection_subject.csv', index=False)
    K = pd.concat([o[1] for o in out])
    A = R.groupby(['transform', 'h'])[['ll_per_dim', 'novelty']].median().join(R.groupby(['transform', 'h'])[MET + ['any']].mean())
    A['apeak_p99'] = K.groupby(['transform', 'h']).apeak.quantile(.99); A['vpeak_p99'] = K.groupby(['transform', 'h']).vpeak.quantile(.99)
    A = A.reset_index(); ok = A[A['any'] <= 0.06].sort_values(['novelty', 'll_per_dim'], ascending=False)
    A['selected'] = False
    if len(ok): A.loc[ok.index[0], 'selected'] = True
    A.to_csv(f'{HERE}/sampler_selection.csv', index=False)
    pd.set_option('display.width', 250); print(A.round(3).to_string(index=False))
    print('best feasible per transform:'); print(ok.groupby('transform').head(1).round(3).to_string(index=False))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, tf in enumerate(ks.TFS):
        d = A[A['transform'] == tf]; ax.plot(d.novelty, 100 * d['any'], 'o-', color=f'C{i}', label=tf, ms=4, lw=1.5)
        for _, r in d.iterrows(): ax.annotate(f'{r.h:g}', (r.novelty, 100 * r['any']), fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax.axhline(4.0, color='0.4', ls=':', lw=1); ax.axhline(6.0, color='#d62728', ls='--', lw=1)
    ax.text(ax.get_xlim()[1], 6.3, 'selection limit (6%)', ha='right', fontsize=7, color='#d62728')
    ax.text(ax.get_xlim()[1], 3.75, 'real movements (4.0%)', ha='right', va='top', fontsize=7, color='0.4')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Novelty (sample-to-real NN distance / real-to-real NN distance)')
    ax.set_ylabel('Physically implausible samples (%)'); ax.grid(alpha=.4); ax.legend(title='transform (labels: h)', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{HERE}/sampler_frontier.png', dpi=200)
