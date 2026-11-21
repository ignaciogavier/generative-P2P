"""Phase I(b): sample new movements per healthy subject and write Phase II trajectory files.

Input : phase1a features, <feat_dir>/Subject{nnn}_MoCapTraj_feat_v2.npy (P2P and non-P2P movements of each subject).
Output: <out_dir>/<variant>/S{n}_p2p.npy and S{n}_neg.npy in skeleton format [log10 T, hand, 6 keyframes x 3 axes]
        (hand drawn at random), and <out_dir>/<variant>/tasks.txt for phase2_simulate_movement.py, one line per
        (subject, class): "<variant> <subject> <is_not_P2P> <traj_file> <n> <fixed_sensor>", Phase II seed = 777 + line index.
Samplers (--sampler):
  local, white, std, copula, shape   KDE with bandwidth --h in a transformed feature space (kde_samplers.Model), per
                                     subject. Paper (revised): local, h = 0.35. Submitted paper: std, h = 0.05.
  none                               ablation A: real movements redrawn with replacement, no KDE noise (h -> 0)
  coeff                              ablation B: standardized KDE (h = 0.05) on the quintic coefficients instead of PVDVVP
Seeds follow the runs reported in the paper, so the same inputs give the same files:
  KDE samplers: 1000*s + is_not_P2P + 500 + 37*index(sampler in TFS) + round(1000*h);  none: 1000*s + is_not_P2P;
  coeff: 1000*s + is_not_P2P + 100.
--n: trajectories per class and subject (each gives one left- and one right-wrist window); 0 = the reduced-grid counts
of the ablations (kde_samplers.NEED / 2). Revised paper: local --n 10000; ablations A/B: --n 0.

Example: python phase1b_synthesize_trajectories.py --feat_dir feature_extraction --out_dir trajectories --sampler local --h 0.35 --n 10000
"""
import os, argparse
import numpy as np
import kde_samplers as ks


def variant_name(sampler, h):
    if sampler in ks.TFS: return f"{sampler}_H{f'{h:g}'.replace('.', 'p')}"
    return {'none': 'A_no_resampling', 'coeff': 'B_coeff_kde'}[sampler]


def sample(sampler, h, Z, n, s, inp):
    if sampler in ks.TFS:
        rng = np.random.default_rng(1000 * s + inp + 500 + 37 * ks.TFS.index(sampler) + int(round(1000 * h)))
        hands = rng.integers(0, 2, size=(n, 1)).astype(float)
        return ks.skeleton_from_pvdvvp(ks.Model(sampler, h).fit(Z).sample(n, rng), hands)
    seed = 1000 * s + inp + {'none': 0, 'coeff': 100}[sampler]
    rng = np.random.default_rng(seed)
    hands = rng.integers(0, 2, size=(n, 1)).astype(float)
    if sampler == 'none':
        return ks.skeleton_from_pvdvvp(Z[rng.integers(0, len(Z), n)], hands)
    return ks.skeleton_from_coeff(ks.kde_sample_sklearn(ks.to_coeff_space(Z), 0.05, n, seed), hands)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--feat_dir', default='feature_extraction', help='phase1a output folder')
    ap.add_argument('--out_dir', default='trajectories')
    ap.add_argument('--sampler', default='local', choices=ks.TFS + ['none', 'coeff'])
    ap.add_argument('--h', type=float, default=0.35, help='KDE bandwidth (KDE samplers only)')
    ap.add_argument('--n', type=int, default=10000, help='trajectories per class and subject; 0 = reduced-grid counts')
    ap.add_argument('--subjects', type=int, nargs='*', default=ks.SUBJECTS)
    ap.add_argument('--fixed_sensor', type=int, default=0, help='written to tasks.txt (ablation C)')
    a = ap.parse_args()
    v = variant_name(a.sampler, a.h); d = os.path.join(a.out_dir, v)
    assert not os.path.exists(d), f'{d} exists - refusing to overwrite'
    os.makedirs(d)
    tasks = []
    for s in a.subjects:
        n = a.n or int(np.ceil(ks.NEED[s] / 2))
        for kd, inp in (('p2p', 0), ('neg', 1)):
            f = os.path.join(d, f'S{s}_{kd}.npy')
            np.save(f, sample(a.sampler, a.h, ks.load_features(a.feat_dir, s, kd), n, s, inp))
            tasks.append(f'{v} {s} {inp} {f} {n} {a.fixed_sensor}')
        print(v, 'subject', s, n, flush=True)
    open(os.path.join(d, 'tasks.txt'), 'w').write('\n'.join(tasks) + '\n')
    print(len(tasks), 'Phase II tasks ->', os.path.join(d, 'tasks.txt'))
