# Biomechanics-informed generative P2P

Code for the experiments of the paper: synthetic wrist-IMU data of point-to-point (P2P) arm movements, generated from the
motion capture of healthy subjects, used to train P2P movement detectors for stroke survivors, and the digital
biomarkers of upper-limb impairment computed from the detected movements.

The recordings are not included. The in-house healthy (MoCap + IMU) and stroke (IMU + clinical scores) datasets are not
public; StrokeRehab (external validation) is a public dataset. Nothing in this repository contains subject data.

## Pipeline

```
generative_pipeline/   Phase I  real MoCap movements -> minimum-jerk features -> KDE resampling -> new trajectories
                       Phase II trajectories -> Blender avatar -> synthetic wrist IMU windows (3 s, 50 Hz)
p2p_classifier/        ResNet-34 P2P detector: training, pretraining on synthetic data, fine-tuning on stroke data
digital_biomarker/     detector scores on the stroke recordings -> movement segmentation -> features -> FMA-UE
analysis/              per-fold results from WandB -> statistics, break-even points, ablations, Table II, figures
strokerehab/           external validation on StrokeRehab (fine-tuning sweep and zero-shot)
```

| Paper item | Produced by |
|---|---|
| Synthetic data (adopted: local KDE, h = 0.35) | `generative_pipeline/phase1b_synthesize_trajectories.py`, `phase2_simulate_movement.py` |
| Sampler and bandwidth choice, Table II | `analysis/sampler_selection.py`, `analysis/table2_metrics.py` |
| Fig. 3 (healthy LOSO), Fig. 4 (stroke LOSO) | `p2p_classifier/P2P_classifier.py` -> `analysis/build_per_fold.py`, `detection_stats.py`, `breakeven.py`, `make_figs.py` |
| Ablations, dataset size | `phase1b --sampler none / coeff / std`, `phase2 --fixed_sensor 1` -> `analysis/ablation_table.py`, `dataset_size.py` |
| Fig. 5 (clinical scores) | `digital_biomarker/predict.py`, `feature_extraction.py`, `motor_assessment.py`, `average_seeds.py` -> `analysis/fig5_stats.py`, `fig5_fig.py` |
| StrokeRehab | `strokerehab/` |

## Setup

Python 3.11: `pip install -r requirements.txt` (`bpy` is only needed for Phase II). Training logs to
[Weights & Biases](https://wandb.ai) (`wandb login`).

Paths are set with environment variables (defaults are relative to the directory you run from):

| Variable | Default | Contents |
|---|---|---|
| `P2P_DATA` | `data` | real windows: `data_real_th0.4_singleP2P_man/S{n}/`, `data_stroke_th0.4_singleP2P_man/S{n}/` |
| `P2P_SYNTH` | `synthetic_dataset/local_H0p35` | Phase II output, `S{n}_3s50Hz/` |
| `P2P_MODELS` | `models` | saved detectors (`synth1_seed<seed>.pth`, `train_10real_seed<seed>.pth`, ...) |
| `STROKE_KINEMATICS` | `input` | orientation-estimated stroke recordings `stroke_data_<ahrs>.npz` (digital_biomarker) |
| `STROKEREHAB_DIR` / `STROKEREHAB_OUT` | `strokerehab_data` / `results/strokerehab` | StrokeRehab raw data, cache, windows / result tables |

A window is a pair of files `<labels>_<k>_<l|r>{acc,gyr}.npy` (T x 3). Real windows: `{p|n}left{p|n}right` gives the
P2P label of each wrist; synthetic windows: `{p|n}{left|right}`.

## Reproducing the experiments

Seeds: each LOSO fold is a `--random_seed` (healthy: 0 1 2 3 4 5 6 7 8 10 13 18 22 26 28 35 44 62; stroke: 22 2 8 18 26 3
1 6 5 13 62 7 35 10 4 0 28). Network weights are not seeded (cuDNN is disabled), so retraining reproduces the results up
to run-to-run noise (SD of the calibrated AUPRC about 0.007); Phase I files and all statistics are deterministic.

**1. Synthetic data** (`generative_pipeline/`)
```bash
python phase1a_extracting_skeleton_mocap_and_create_features.py          # -> feature_extraction/Subject{nnn}_MoCapTraj_feat_v2.npy
python phase1b_synthesize_trajectories.py --sampler local --h 0.35 --n 10000   # -> trajectories/local_H0p35/{S*.npy, tasks.txt}
# one Phase II call per line of tasks.txt ("<variant> <subject> <is_not_P2P> <traj_file> <n> <fixed_sensor>"), seed = 777 + line index
python phase2_simulate_movement.py --subject_target 1 --is_not_P2P 0 --traj_file trajectories/local_H0p35/S1_p2p.npy \
    --out_dir synthetic_dataset/local_H0p35 --count 10000 --fixed_sensor 0 --seed 777
```
Ablations: `--sampler none --n 0` (no resampling), `--sampler coeff --n 0` (KDE on coefficients), `--sampler std --h 0.05`
(submitted sampler), and Phase II with `--fixed_sensor 1` on the adopted trajectories (fixed sensor placement).

**2. Detectors** (`p2p_classifier/`, one run per fold and training size; the docstring lists the exact arguments)
```bash
python P2P_classifier.py --random_seed 0 --synth_splits 1 0 0 --real_splits 0 3 1 --save_name synth1_seed0 --tag fig3_synth  # Fig. 3
python P2P_classifier.py --random_seed 0 --real_splits 1 3 1 --tag fig3_real                                                 # Fig. 3
python P2P_classifier.py --random_seed 22 --stroke_splits 5 3 1 --pretr_path models/synth1_seed22.pth --tag fig4_synth       # Fig. 4
python P2P_classifier.py --random_seed 22 --stroke_splits 5 3 1 --tag fig4_scratch                                           # Fig. 4
```

**3. Results and figures** (`analysis/`)
```bash
python pull_wandb.py --entity <your entity> --project P2P-classifier
python build_per_fold.py --tag fig3_synth --kind healthy --out results/fig3_synth_per_fold.csv     # one call per curve
python detection_stats.py --curve fig3 synth results/fig3_synth_per_fold.csv --curve fig3 real results/fig3_real_per_fold.csv \
    --curve fig4 synth_ft results/fig4_synth_per_fold.csv --curve fig4 scratch results/fig4_scratch_per_fold.csv ...
python breakeven.py results && python make_figs.py results
python ablation_table.py --full results/synth --variant A=results/A --variant B=results/B --variant orig=results/orig --variant C=results/C
python sampler_selection.py --feat_dir ../generative_pipeline/feature_extraction && python table2_metrics.py --feat_dir ../generative_pipeline/feature_extraction
```

**4. Digital biomarkers, Fig. 5** (run from `digital_biomarker/`, inputs in `digital_biomarker/input/`)

Each stroke subject is scored by the detector for which it was the held-out test subject: 17 detectors fine-tuned from
one synthetic model (`--stroke_splits 3 3 1 --stroke_split_legacy --pretr_path models/synth1_seed<p>.pth`, seeds 41 40 19 28
20 38 39 42 43 33 37 22 3 34 36 25 29 for test subjects 1 2 3 4 5 7 9 10 11 12 13 14 16 17 18 19 20), repeated for 18
pretraining seeds `<p>`.
```bash
python predict.py --model_pattern ../models/fig5_seed<p>_S{s}.pth --out_prefix predictions_seed<p>
PRED_PREFIX=predictions_seed<p> META_PREFIX=predictions python feature_extraction.py --ahrs_filter_type mahony \
    --rfunc spearmanr --timestamp seed<p> --corr_target FM --input_wnd_data wnd_sz_3_overlap_1_3
python motor_assessment.py seed<p> p2p_pred           # and once: <manual timestamp> p2p_true
python average_seeds.py --manual <manual timestamp> --auto seed0 seed1 ... --rule inner --name avg18
python ../analysis/fig5_stats.py --pred_csv output/motor_assessment/avg18_preds.csv && python ../analysis/fig5_fig.py --pred_csv output/motor_assessment/avg18_preds.csv
```
The segmentation padding is chosen inside the nested leave-one-subject-out loop (`motor_assessment.py`, rule `inner`),
for manual and automatic segmentation alike.

**5. StrokeRehab** (`strokerehab/`): `screen_trials.py` -> `build_cache.py` -> `align_axes.py` (label-free axis alignment
to our wrist frame) -> `make_windows.py` -> `finetune_sweep.py <task 0-179>` and `zeroshot.py` -> `analyze_sweep.py`,
`zeroshot_compare.py`.

## License

MIT (see LICENSE).
