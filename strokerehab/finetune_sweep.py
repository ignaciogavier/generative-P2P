"""data-efficiency sweep on StrokeRehab (Fig. 4 protocol on an independent public dataset).

Fold f (0..14): test subject = SUBJECTS[f]; the other 14 are permuted with np.random.default_rng(SSEEDS[f]);
validation = last 3 of the permutation, training = first N (nested: the N=3 training set contains the N=1 set, etc.).
Conditions (Fig. 4 recipe, unchanged):
  scratch  random init, Adam lr 1e-4 on all parameters
  real10   init from <P2P_MODELS>/train_10real_seed{SSEEDS[f]}.pth (healthy real, 10 subjects)
  synth1   init from <P2P_MODELS>/synth1_seed{SSEEDS[f]}.pth (synthetic from 1 healthy subject)
  pretrained: layer-wise Adam lr 1e-4 x (1/6, 1/5, 1/4, 1/3, 1/2, 1) for conv1, layer1..4, fc (as p2p_classifier/P2P_classifier.py).
ResNet34, dropout 0.2, batch 64, 30 epochs, BCE weighted by inverse class frequency of the training set, no augmentation.
Model selection: epoch with max validation calibrated AUPRC (tie: train), report test calibrated AUPRC (paper rule).
Output: <STROKEREHAB_OUT>/sweep/<cond>_N<N>_f<f>.csv (one row per epoch). No model files are written.
Usage: python finetune_sweep.py <task id 0..179>  [--epochs E for smoke tests]
  task -> cond = CONDS[i // 60], N = NS[(i % 60) // 15], f = i % 15
"""
import os, sys, argparse, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, OUT, SUBJECTS, SSEEDS, load_subject, build_model, auprc_unb

CONDS = ['scratch', 'real10', 'synth1']
NS = [1, 3, 5, 10]
PRETR = {'real10': 'train_10real_seed{}.pth', 'synth1': 'synth1_seed{}.pth'}   # in P2P_MODELS (see p2p_classifier)


def split(f, N):
    others = [s for s in SUBJECTS if s != SUBJECTS[f]]
    perm = list(np.random.default_rng(SSEEDS[f]).permutation(others))
    return perm[:N], perm[-3:], [SUBJECTS[f]]


def stack(subjs):
    X, y = zip(*[load_subject(s) for s in subjs]); return np.concatenate(X), np.concatenate(y)


def epoch(model, loader, dev, w, opt=None):
    model.train(opt is not None); yt, yp = [], []
    with torch.set_grad_enabled(opt is not None):
        for X, y in loader:
            X, y = X.to(dev), y.to(dev)[:, None]
            out = model(X)
            if opt is not None:
                loss = nn.functional.binary_cross_entropy(out, y, weight=w[0] * (1 - y) + w[1] * y)
                opt.zero_grad(); loss.backward(); opt.step()
            yt.append(y.detach().cpu().numpy().ravel()); yp.append(out.detach().cpu().numpy().ravel())
    return auprc_unb(np.concatenate(yt), np.concatenate(yp))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('task', type=int); ap.add_argument('--epochs', type=int, default=30)
    a = ap.parse_args()
    cond, N, f = CONDS[a.task // 60], NS[(a.task % 60) // 15], a.task % 15
    seed = SSEEDS[f]; torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.enabled = False                       # as in P2P_classifier.py
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr, va, te = split(f, N)
    print(f'task {a.task}: cond={cond} N={N} fold={f} seed={seed} train={tr} val={va} test={te}', flush=True)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = stack(tr), stack(va), stack(te)
    p = ytr.mean(); w = [1 / (1 - p), 1 / p]
    print(f'windows train {len(ytr)} (pos {p:.2f}) val {len(yva)} test {len(yte)}', flush=True)
    mk = lambda X, y, sh: DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=64, shuffle=sh)
    Ltr, Lva, Lte = mk(Xtr, ytr, True), mk(Xva, yva, False), mk(Xte, yte, False)

    model = build_model(dev); lr = 1e-4
    if cond == 'scratch':
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        path = f'{MODELS}/{PRETR[cond].format(seed)}'
        model.load_state_dict(torch.load(path, map_location=dev)); print('loaded', path)
        opt = optim.Adam([{'params': model.conv1.parameters(), 'lr': lr / 6}, {'params': model.layer1.parameters(), 'lr': lr / 5},
                          {'params': model.layer2.parameters(), 'lr': lr / 4}, {'params': model.layer3.parameters(), 'lr': lr / 3},
                          {'params': model.layer4.parameters(), 'lr': lr / 2}, {'params': model.fc.parameters(), 'lr': lr}])
    rows = []
    for e in range(a.epochs):
        t0 = time.time()
        tr_m = epoch(model, Ltr, dev, w, opt)
        rows.append(dict(cond=cond, N=N, fold=f, seed=seed, test_subject=te[0], epoch=e, TrUnbAUPRC=tr_m,
                         VaUnbAUPRC=epoch(model, Lva, dev, w), TeUnbAUPRC=epoch(model, Lte, dev, w)))
        print(f"epoch {e:02d} tr {rows[-1]['TrUnbAUPRC']:.3f} va {rows[-1]['VaUnbAUPRC']:.3f} te {rows[-1]['TeUnbAUPRC']:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(f'{OUT}/sweep', exist_ok=True)
    out = f'{OUT}/sweep/{cond}_N{N}_f{f}.csv' if a.epochs == 30 else f'{OUT}/sweep/smoke_{cond}_N{N}_f{f}.csv'
    pd.DataFrame(rows).to_csv(out, index=False); print('wrote', out)
