"""Window scores of the Fig. 5 detectors on the stroke recordings.

Each stroke subject is scored ONLY by the detector for which it was the held-out test subject (17 detectors, one per
subject, trained with p2p_classifier/P2P_classifier.py --stroke_splits 3 3 1 --stroke_split_legacy).
Run from digital_biomarker/. Reads input/wnd_sz_3_overlap_1_3/<sss>_stroke_windows.npz and the weights
<model_pattern> (with {s} = subject); writes input/<out_prefix>_wnd_sz_3_overlap_1_3/<ss>.npz with out = [1 - p, p],
the scores read by feature_extraction.py through PRED_PREFIX=<out_prefix>.
Example: python predict.py --model_pattern ../models/fig5_local_H0p35_seed1_S{s}.pth --out_prefix predictions_local_H0p35_seed1
"""
import os, sys, argparse
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'p2p_classifier'))
from utils_architectures import ResNet, BasicBlock  # noqa: E402

SUBJECTS = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_pattern', required=True, help='e.g. ../models/fig5_<name>_S{s}.pth')
    ap.add_argument('--out_prefix', required=True)
    a = ap.parse_args()
    out_dir = f'input/{a.out_prefix}_wnd_sz_3_overlap_1_3'; os.makedirs(out_dir, exist_ok=True)
    model = ResNet(BasicBlock, [3, 4, 6, 3], 0.2, with_gyro=False, with_acc=True)
    for s in SUBJECTS:
        w = np.load(f'input/wnd_sz_3_overlap_1_3/{s:03d}_stroke_windows.npz', allow_pickle=True)['windows']
        model.load_state_dict(torch.load(a.model_pattern.format(s=s), map_location='cpu')); model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(np.stack([x.local.acc.T for x in w]))).numpy().flatten()
        np.savez(f'{out_dir}/{s:02d}.npz', out=np.stack([1 - p, p], 1)); print(s, len(p), round(float(p.mean()), 3), flush=True)
