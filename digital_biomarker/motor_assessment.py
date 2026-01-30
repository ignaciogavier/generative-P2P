#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, GroupKFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
import seaborn as sn
plt.rcParams['figure.dpi'] = 150
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Nimbus Sans']
plt.rcParams['font.size'] = 8

from scripts.shimmer_data_loader import load_stroke_demographics
from scripts.features import icc


# In[2]:


# Load scores
filename = os.path.join('input', 'stroke_subject_info.xlsx')
demographics = load_stroke_demographics(filename)

subject_list = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
score_keys = ['FM', 'MAL_AoU', 'MAL_QoM', 'WMFT_TR', 'WMFT_FAS']
score_dict = dict()
for score_key in score_keys: 
    score_dict[score_key] = np.array([getattr(demographics[s], score_key) for s in subject_list])    
score_df = pd.DataFrame(score_dict, index=subject_list)
score_df


# In[3]:


# Get data
data = np.load(f'output/spearmanr_all/feature_dict_1540_02_04_2025_mahony_ICC_(2, 1)_0.75' +
               '_wnd_sz_3_overlap_1_3_split_method_1_1.5.npz', allow_pickle=True)['data'].item()

data_manual = data[('mahony', 'p2p_true')]
data_auto = data[('mahony', 'p2p_pred')]

X_manual = data_manual['features'] # (17, 88)
feat1_manual = data_manual['features_1st']
feat2_manual = data_manual['features_2nd']
X_auto = data_auto['features'] # (17, 88)
feat1_auto = data_auto['features_1st']
feat2_auto = data_auto['features_2nd']

y = score_df[['FM']].to_numpy()
groups = data[('mahony', 'p2p_true')]['subjects']


# In[4]:


# Custom scoring function for mean Pearson R
def pearson_scorer(y_true, y_pred):
    corrs = []
    for i in range(y_true.shape[1]):
        corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
        corrs.append(corr if not np.isnan(corr) else 0)
    return np.mean(corrs)

# Custom scoring function for mean Spearman R
def spearman_scorer(y_true, y_pred):
    corrs = []
    for i in range(y_true.shape[1]):
        corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
        corrs.append(corr if not np.isnan(corr) else 0)
    return np.mean(corrs)

class score(object):
    def __init__(self, X_full, y_full):
        self.X_full = X_full
        self.y_full = y_full
    def __call__(self, estimator, X, y=None):
        return spearman_scorer(self.y_full, estimator.predict(self.X_full))

class FeatureSelector(BaseEstimator, RegressorMixin):

    def __init__(self, X_full, feat1, feat2, threshold=0.75):
        self.X_full = X_full
        self.feat1 = feat1
        self.feat2 = feat2
        self.threshold = threshold

    def fit(self, X, y=None):
        idx = np.zeros((self.X_full.shape[0],), dtype=bool)
        for i in range(len(idx)):
            j = sum(idx[:i+1])
            if j == X.shape[0]: break
            if np.allclose(self.X_full[i], X[j]): idx[i] = True

        self.iccs = np.array([icc(np.stack([self.feat1, self.feat2], 2)[idx,f], '(2, 1)')[0]
                              for f in range(self.X_full.shape[1])])
        self.mask = self.iccs > self.threshold

        self.is_fitted_ = True
        return self

    def transform(self, X):
        return X[:,self.mask]


# In[5]:


# Outer loop: Leave-One-Subject-Out CV
logo_outer = LeaveOneGroupOut()
y_preds = []
results_manual = defaultdict(list)

for train_idx, test_idx in logo_outer.split(X_manual, y, groups):
    X_train, X_test = X_manual[train_idx], X_manual[test_idx]
    feat1_train, feat1_test = feat1_manual[train_idx], feat1_manual[test_idx]
    feat2_train, feat2_test = feat2_manual[train_idx], feat2_manual[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    # Inner loop: LOSO for hyperparameter tuning
    logo_inner = LeaveOneGroupOut()
    # lpgo_inner = LeavePGroupsOut(n_groups=4)
    # gkf_inner = GroupKFold(n_splits=4)

    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(X_train, feat1_train, feat2_train)),
        ('scaler', StandardScaler()),
        ('regressor', MultiOutputRegressor(Ridge()))
    ])

    param_grid = {
        'feature_selector__threshold': np.linspace(0.75, 0.8, 6),
        'regressor__estimator__alpha': np.logspace(-1, 1, 5, base=10)
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=logo_inner.split(X_train, y_train, groups_train),
        scoring=score(X_train, y_train),
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    y_preds.append(y_pred)
    results_manual['test_subj'].append(groups[test_idx[0]])
    results_manual['FM_gt'].append(y_test.item())
    results_manual['FM_pred'].append(y_pred.item())
    results_manual['best_score'].append(grid_search.best_score_)
    results_manual['mean_icc'].append(best_model['feature_selector'].iccs
                                      [best_model['feature_selector'].mask].mean())
    results_manual['rel_feat'].append(best_model['feature_selector'].mask.sum())
    results_manual['threshold'].append(best_model['feature_selector'].threshold.item())
    results_manual['alpha'].append(best_model['regressor'].estimator.alpha.item())

    print(f"Subj: {groups[test_idx].item()}, GT: {y_test.item()}, Pred: {y_pred.item():.2f}")

y_pred_manual = np.concatenate(y_preds, 0)
results_manual = pd.DataFrame(results_manual)

print(f"\nAverage LOSO Pearson: {pearson_scorer(y, y_pred_manual):.4f}")
print(f"\nAverage LOSO Spearman: {spearman_scorer(y, y_pred_manual):.4f}")

results_manual


# In[6]:


# Outer loop: Leave-One-Subject-Out CV
logo_outer = LeaveOneGroupOut()
y_preds = []
results_auto = defaultdict(list)

for train_idx, test_idx in logo_outer.split(X_auto, y, groups):
    X_train, X_test = X_auto[train_idx], X_auto[test_idx]
    feat1_train, feat1_test = feat1_auto[train_idx], feat1_auto[test_idx]
    feat2_train, feat2_test = feat2_auto[train_idx], feat2_auto[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    # Inner loop: LOSO for hyperparameter tuning
    logo_inner = LeaveOneGroupOut()
    # lpgo_inner = LeavePGroupsOut(n_groups=4)
    # gkf_inner = GroupKFold(n_splits=4)

    pipeline = Pipeline([
        ('feature_selector', FeatureSelector(X_train, feat1_train, feat2_train)),
        ('scaler', StandardScaler()),
        ('regressor', MultiOutputRegressor(Ridge()))
    ])

    param_grid = {
        'feature_selector__threshold': np.linspace(0.75, 0.8, 6),
        'regressor__estimator__alpha': np.logspace(-1, 1, 5, base=10)
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=logo_inner.split(X_train, y_train, groups_train),
        scoring=score(X_train, y_train),
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    y_preds.append(y_pred)

    results_auto['test_subj'].append(groups[test_idx[0]])
    results_auto['FM_gt'].append(y_test.item())
    results_auto['FM_pred'].append(y_pred.item())
    results_auto['best_score'].append(grid_search.best_score_)
    results_auto['mean_icc'].append(best_model['feature_selector'].iccs
                                    [best_model['feature_selector'].mask].mean())
    results_auto['rel_feat'].append(best_model['feature_selector'].mask.sum())
    results_auto['threshold'].append(best_model['feature_selector'].threshold.item())
    results_auto['alpha'].append(best_model['regressor'].estimator.alpha.item())



    print(f"Subj: {groups[test_idx].item()}, GT: {y_test.item()}, Pred: {y_pred.item():.2f}")

y_pred_auto = np.concatenate(y_preds, 0)
results_auto = pd.DataFrame(results_auto)

print(f"\nAverage LOSO Pearson: {pearson_scorer(y, y_pred_auto):.4f}")
print(f"\nAverage LOSO Spearman: {spearman_scorer(y, y_pred_auto):.4f}")

results_auto


# In[12]:


dFAll = pd.DataFrame(np.block([[y, y_pred_manual, np.zeros_like(y)],
                               [y, y_pred_auto, np.ones_like(y)]]),
                     columns=['clinicalFM','predictedFM','is_auto'])

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.97)
ax.plot([25,75], [25,75], '--k')
ax.set_xlim(25,75)
ax.set_ylim(25,75)
ax.grid()

labels, legend_labels = {True:'Automatically predicted', False:'Manually annotated'}, []
for group, data in dFAll.groupby('is_auto'):
    sn.scatterplot(data=data, x='clinicalFM', y='predictedFM', color=sn.color_palette('tab10')[int(1-group)], label=group, ax=ax)
    sn.regplot(data=data, x='clinicalFM', y='predictedFM', scatter=False, truncate=False, color=sn.color_palette('tab10')[int(1-group)], ax=ax)
    r, _ = pearsonr(data['clinicalFM'], data['predictedFM'])
    legend_labels.append(f'{labels[group]}; r = {r:.3f}')

ax.set_xlabel('Assessed FM')
ax.set_ylabel('Digital Biomarker')
ax.legend(loc='lower right', handles=ax.get_legend_handles_labels()[0][::-1], labels=legend_labels[::-1])

[fig.savefig(f'./figures/assessment_fm.{f}', dpi=150) for f in ['svg','pdf']]


# Correlation of biomarker with other scores

# In[8]:


# WMFT_TR
print(f"LOSO Pearson w/WMFT_TR (manual): {pearson_scorer(score_df[['WMFT_TR']].to_numpy(), y_pred_manual):.4f}")
print(f"LOSO Spearman w/WMFT_TR (manual): {spearman_scorer(score_df[['WMFT_TR']].to_numpy(), y_pred_manual):.4f}")
print(f"LOSO Pearson w/WMFT_TR (auto): {pearson_scorer(score_df[['WMFT_TR']].to_numpy(), y_pred_auto):.4f}")
print(f"LOSO Spearman w/WMFT_TR (auto): {spearman_scorer(score_df[['WMFT_TR']].to_numpy(), y_pred_auto):.4f}")

# WMFT_FAS
print(f"LOSO Pearson w/WMFT_FAS (manual): {pearson_scorer(score_df[['WMFT_FAS']].to_numpy(), y_pred_manual):.4f}")
print(f"LOSO Spearman w/WMFT_FAS (manual): {spearman_scorer(score_df[['WMFT_FAS']].to_numpy(), y_pred_manual):.4f}")
print(f"LOSO Pearson w/WMFT_FAS (auto): {pearson_scorer(score_df[['WMFT_FAS']].to_numpy(), y_pred_auto):.4f}")
print(f"LOSO Spearman w/WMFT_FAS (auto): {spearman_scorer(score_df[['WMFT_FAS']].to_numpy(), y_pred_auto):.4f}")

# MAL_AoU
print(f"LOSO Pearson w/MAL_AoU (manual): {pearson_scorer(score_df[['MAL_AoU']].to_numpy(), y_pred_manual):.4f}")
print(f"LOSO Spearman w/MAL_AoU (manual): {spearman_scorer(score_df[['MAL_AoU']].to_numpy(), y_pred_manual):.4f}")
print(f"LOSO Pearson w/MAL_AoU (auto): {pearson_scorer(score_df[['MAL_AoU']].to_numpy(), y_pred_auto):.4f}")
print(f"LOSO Spearman w/MAL_AoU (auto): {spearman_scorer(score_df[['MAL_AoU']].to_numpy(), y_pred_auto):.4f}")

# MAL_QoM
print(f"LOSO Pearson w/MAL_QoM (manual): {pearson_scorer(score_df[['MAL_QoM']].to_numpy(), y_pred_manual):.4f}")
print(f"LOSO Spearman w/MAL_QoM (manual): {spearman_scorer(score_df[['MAL_QoM']].to_numpy(), y_pred_manual):.4f}")
print(f"LOSO Pearson w/MAL_QoM (auto): {pearson_scorer(score_df[['MAL_QoM']].to_numpy(), y_pred_auto):.4f}")
print(f"LOSO Spearman w/MAL_QoM (auto): {spearman_scorer(score_df[['MAL_QoM']].to_numpy(), y_pred_auto):.4f}")


fig, ax = plt.subplots(2, 2, figsize=(4, 4), sharey=True)
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.97, hspace=0.4)

for s, score in enumerate(['WMFT_TR','WMFT_FAS','MAL_AoU','MAL_QoM']):
    axi = ax[s//2,s%2]
    dFAll = pd.DataFrame(np.block([[score_df[[score]].to_numpy(), y_pred_manual, np.zeros_like(y_pred_manual)],
                                   [score_df[[score]].to_numpy(), y_pred_auto, np.ones_like(y_pred_manual)]]),
                         columns=[score,'predictedFM','is_auto'])

    labels, legend_labels = {True:'Automatically predicted', False:'Manually annotated'}, []
    for group, data in dFAll.groupby('is_auto'):
        sn.scatterplot(data=data, x=score, y='predictedFM', color=sn.color_palette('tab10')[int(1-group)], label=group, ax=axi)
        sn.regplot(data=data, x=score, y='predictedFM', scatter=False, truncate=False, color=sn.color_palette('tab10')[int(1-group)], ax=axi)
        r, _ = pearsonr(data[score], data['predictedFM'])
        legend_labels.append(f'r = {r:.3f}')
    axi.set_xlabel(score.replace('_', '-'))
    axi.set_ylabel('Predicted FM')
    axi.grid()
    axi.legend(handles=axi.get_legend_handles_labels()[0][::-1], labels=legend_labels[::-1])

axi.set_ylim(25,75)

[fig.savefig(f'./figures/assessment_scores.{f}', dpi=150) for f in ['svg','pdf']]


# In[30]:


fig, ax = plt.subplots(2, 5, figsize=(10, 4), sharey=True, sharex='col')
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.97, hspace=0.2)

for s, score in enumerate(['FM','WMFT_TR','WMFT_FAS','MAL_AoU','MAL_QoM']):
    axi = ax[:,s]
    dFAll = pd.DataFrame(np.block([[score_df[[score]].to_numpy(), y_pred_manual, np.zeros_like(y_pred_manual)],
                                   [score_df[[score]].to_numpy(), y_pred_auto, np.ones_like(y_pred_manual)]]),
                         columns=[score,'predictedFM','is_auto'])
    score = 'FMA_UE' if score == 'FM' else score
    dFAll = dFAll.rename(columns={'FM':score})

    labels, legend_labels = {True:'Automatically predicted', False:'Manually annotated'}, []
    for group, data in dFAll.groupby('is_auto'):
        sn.scatterplot(data=data, x=score, y='predictedFM', color=sn.color_palette('tab10')[int(1-group)], label=group, ax=axi[int(group)])
        sn.regplot(data=data, x=score, y='predictedFM', scatter=False, truncate=False, color=sn.color_palette('tab10')[int(1-group)], ax=axi[int(group)])
        r, _ = pearsonr(data[score], data['predictedFM'])
        legend_labels.append(f'r = {r:.3f}')
    axi[1].set_xlabel(score.replace('_', '-'))
    axi[0].grid()
    axi[1].grid()
    print(legend_labels[0], legend_labels[1])
    axi[0].legend(handles=axi[0].get_legend_handles_labels()[0], labels=legend_labels[:1])
    axi[1].legend(handles=axi[1].get_legend_handles_labels()[0], labels=legend_labels[1:])

ax[0,0].set_ylabel('Digital Biomarker\nUsing Manual Annotations')
ax[1,0].set_ylabel('Digital Biomarker\nUsing P2P Detectors')
ax[0,0].set_ylim(25,75)

[fig.savefig(f'./figures/assessment_scores_and_FM.{f}', dpi=150) for f in ['svg','pdf']]


# In[ ]:




