"""Shared helpers: StrokeRehab windows, the paper's detector, and the paper's calibrated AUPRC (verbatim formula from
p2p_classifier/P2P_classifier.run_epoch: precision recalibrated to a 50% prevalence, AP = -sum(diff(recall) * precision))."""
import os, sys
import numpy as np
import torch
from sklearn.metrics._ranking import _binary_clf_curve

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'p2p_classifier'))
from utils_architectures import ResNet, BasicBlock  # noqa: E402

WS = os.environ.get('STROKEREHAB_DIR', 'strokerehab_data')
MODELS = os.environ.get('P2P_MODELS', 'models')
OUT = os.environ.get('STROKEREHAB_OUT', 'results/strokerehab')
SUBJECTS = ['s01', 's29', 's31', 's32', 's34', 's35', 's36', 's37', 's39', 's41', 's45', 's46', 's47', 's48', 's49']
SSEEDS = [22, 2, 8, 18, 26, 3, 1, 6, 5, 13, 62, 7, 35, 10, 4, 0, 28]  # Fig. 4 seeds; fold f uses SSEEDS[f]


def load_subject(s):
    z = np.load(f"{WS}/{os.environ.get('STROKEREHAB_WINDOWS', 'windows_hand')}/{s}.npz")
    return z['X'].transpose(0, 2, 1).astype(np.float32), z['y'].astype(np.float32)  # (n, 3, 150) like CustomDataset


def build_model(device):
    # ResNet34, dropout 0.2, accelerometer only (paper configuration)
    return ResNet(BasicBlock, [3, 4, 6, 3], 0.2, with_gyro=False, with_acc=True).to(device)


def auprc_unb(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2: return np.nan
    ratio = y_true.sum() / (len(y_true) - y_true.sum())
    fps, tps, thrs = _binary_clf_curve(y_true, y_pred, pos_label=1)
    fps, tps = fps[::-1], tps[::-1]
    prec_unbs = np.hstack([np.divide(tps, tps + ratio * fps, out=np.zeros(len(tps)), where=(tps + ratio * fps != 0)), 1])
    recs = np.hstack([tps / tps[0], 0])
    return float(-np.sum(np.diff(recs) * prec_unbs[:-1]))


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    return np.concatenate([model(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy().ravel() for i in range(0, len(X), bs)])
