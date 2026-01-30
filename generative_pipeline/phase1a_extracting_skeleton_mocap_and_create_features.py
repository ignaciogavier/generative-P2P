# %%
import csv
import glob
import pandas as pd
import math
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import torch
import matplotlib.pyplot as plt
import matplotlib.collections as cln
import mpl_toolkits.mplot3d.art3d as a3d
import matplotlib.colors as clr
import matplotlib as mpl
import seaborn as sns
from scipy.interpolate import interp1d, lagrange
from scipy.signal import savgol_filter, butter, filtfilt, convolve, cwt, morlet2, upfirdn
from scipy.integrate import cumtrapz, solve_ivp
from scipy.fft import fft, fftfreq
from scipy.ndimage import maximum_filter
from scipy.stats import gaussian_kde
from sklearn.linear_model import HuberRegressor, LinearRegression, Lasso
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KernelDensity
from umap import UMAP

# %matplotlib inline

# %% [markdown]
# Define subject ID

# %%
for SUBJID in range(1,21):
    if SUBJID == 9: continue

    # %% [markdown]
    # Offsets needed to align time vectors

    # %%
    timeOffsets = {1: 10.9, 2: 2.1, 3: 3.4, 4: 3.0, 5: 12.4, 6: 4.45, 7:-26.95, 8:19.5, 10: 8.4, 11: 7.4,
                12: 24.4, 13: 11.5, 14: 39.5, 15: 8.3, 16: 14.2, 17: 2.3, 18: 6.3, 19: 7.6, 20: 3.0}

    # %% [markdown]
    # Labels P2P

    # %%
    MoCapFeat = np.load(f'feature_extraction/Subject{SUBJID:03d}_MoCap_feat.npy')
    MoCapFeatAut = np.load(f'feature_extraction/Subject{SUBJID:03d}_MoCap_featAut.npy')

    np.random.shuffle(MoCapFeat)
    np.random.shuffle(MoCapFeatAut)

    # %% [markdown]
    # Read MoCap files

    # %%
    mocapAll = pd.read_csv(f'Subject{SUBJID:03d}/mocap/Subject{SUBJID:03d}.tsv', header=11, sep='\t', index_col=0)
    mocapAll = mocapAll[['Time'] + [f'{s}_{m} {a}' for s in ['left','right'] for m in ['finger','wrist'] for a in ['X','Y','Z']]]

    # %% [markdown]
    # Extract MoCap data

    # %%
    timeMoCap = (mocapAll['Time'] - timeOffsets[SUBJID]).to_numpy()
    leftWristMoCap = (mocapAll[[f'left_wrist {a}' for a in ['X','Y','Z']]] / 1000).to_numpy()
    rightWristMoCap = (mocapAll[[f'right_wrist {a}' for a in ['X','Y','Z']]] / 1000).to_numpy()
    leftFingerMoCap = (mocapAll[[f'left_finger {a}' for a in ['X','Y','Z']]] / 1000).to_numpy()
    rightFingerMoCap = (mocapAll[[f'right_finger {a}' for a in ['X','Y','Z']]] / 1000).to_numpy()
    dTMoCap = np.diff(timeMoCap).mean()

    # %%
    b, a = butter(3, 3 * 2 * dTMoCap, 'lowpass')
    func = lambda x: interp1d(timeMoCap[~np.isnan(x)], x[~np.isnan(x)], fill_value='extrapolate')(timeMoCap)
    leftWristMoCap = filtfilt(b, a, np.apply_along_axis(func, 0, leftWristMoCap), axis=0)
    rightWristMoCap = filtfilt(b, a, np.apply_along_axis(func, 0, rightWristMoCap), axis=0)
    leftFingerMoCap = filtfilt(b, a, np.apply_along_axis(func, 0, leftFingerMoCap), axis=0)
    rightFingerMoCap = filtfilt(b, a, np.apply_along_axis(func, 0, rightFingerMoCap), axis=0)

    # # Upsample all
    # timeMoCap = upfirdn([.5, 1, .5], timeMoCap, 2, axis=0)
    # leftWristMoCap = upfirdn([.5, 1, .5], leftWristMoCap, 2, axis=0)
    # rightWristMoCap = upfirdn([.5, 1, .5], rightWristMoCap, 2, axis=0)
    # leftFingerMoCap = upfirdn([.5, 1, .5], leftFingerMoCap, 2, axis=0)
    # rightFingerMoCap = upfirdn([.5, 1, .5], rightFingerMoCap, 2, axis=0)
    # dTMoCap = np.diff(timeMoCap).mean()

    b, a = butter(3, .5 * 2 * dTMoCap, 'lowpass')
    leftElbowMoCap = filtfilt(b, a, leftWristMoCap - 1 * (leftFingerMoCap - leftWristMoCap), axis=0)
    rightElbowMoCap = filtfilt(b, a, rightWristMoCap - 1 * (rightFingerMoCap - rightWristMoCap), axis=0)
    torsoMoCap = (leftElbowMoCap + rightElbowMoCap) / 2

    frontMoCap = np.cross(np.array([[0,0,1]]), rightElbowMoCap - leftElbowMoCap, 1, 1)
    frontMoCap /= np.linalg.norm(frontMoCap, axis=1, keepdims=True)
    lateralMoCap = np.cross(frontMoCap, np.array([[0,0,1]]), 1, 1)
    zMoCap = np.cross(lateralMoCap, frontMoCap)

    lims = np.array([(min(leftWristMoCap[:,ax].min(), leftFingerMoCap[:,ax].min(), leftElbowMoCap[:,ax].min(),
                        rightWristMoCap[:,ax].min(), rightFingerMoCap[:,ax].min(), rightElbowMoCap[:,ax].min(),
                        torsoMoCap[:,ax].min()),
                    max(leftWristMoCap[:,ax].max(), leftFingerMoCap[:,ax].max(), leftElbowMoCap[:,ax].max(),
                        rightWristMoCap[:,ax].max(), rightFingerMoCap[:,ax].max(), rightElbowMoCap[:,ax].max(),
                        torsoMoCap[:,ax].max())) for ax in range(3)])

    leftWristMoCap -= lims[None,:,0]
    rightWristMoCap -= lims[None,:,0]
    leftFingerMoCap -= lims[None,:,0]
    rightFingerMoCap -= lims[None,:,0]
    leftElbowMoCap -= lims[None,:,0]
    rightElbowMoCap -= lims[None,:,0]
    torsoMoCap -= lims[None,:,0]

    # %%
    # fig, ax = plt.subplots(2, 2, figsize=(16,15))

    # tMin, tMax = 1114, 1117
    # tMask = (tMin < timeMoCap) * (timeMoCap < tMax)
    # normT = (timeMoCap[tMask] - timeMoCap[tMask][0]) / (timeMoCap[tMask][-1] - timeMoCap[tMask][0])

    # # Back view
    # ax[0,1].add_collection(cln.LineCollection(np.stack([leftWristMoCap[tMask][:-1,[0,2]], leftWristMoCap[tMask][1:,[0,2]]], 1), lw=2*(0+leftWristMoCap[tMask][:-1,1]), alpha=normT[:-1], color='C0'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([rightWristMoCap[tMask][:-1,[0,2]], rightWristMoCap[tMask][1:,[0,2]]], 1), lw=2*(0+rightWristMoCap[tMask][:-1,1]), alpha=normT[:-1], color='C3'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([leftFingerMoCap[tMask][:-1,[0,2]], leftFingerMoCap[tMask][1:,[0,2]]], 1), lw=2*(0+leftFingerMoCap[tMask][:-1,1]), alpha=normT[:-1], color='C2'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([rightFingerMoCap[tMask][:-1,[0,2]], rightFingerMoCap[tMask][1:,[0,2]]], 1), lw=2*(0+rightFingerMoCap[tMask][:-1,1]), alpha=normT[:-1], color='C1'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([leftElbowMoCap[tMask][:-1,[0,2]], leftElbowMoCap[tMask][1:,[0,2]]], 1), lw=2*(0+leftElbowMoCap[tMask][:-1,1]), alpha=normT[:-1], color='C4'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([rightElbowMoCap[tMask][:-1,[0,2]], rightElbowMoCap[tMask][1:,[0,2]]], 1), lw=2*(0+rightElbowMoCap[tMask][:-1,1]), alpha=normT[:-1], color='C5'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([torsoMoCap[tMask][:-1,[0,2]], torsoMoCap[tMask][1:,[0,2]]], 1), lw=2*(0+torsoMoCap[tMask][:-1,1]), alpha=normT[:-1], color='black'))

    # # Side view
    # ax[0,0].add_collection(cln.LineCollection(np.stack([leftWristMoCap[tMask][:-1,1:], leftWristMoCap[tMask][1:,1:]], 1), lw=2*(0+leftWristMoCap[tMask][:-1,0]), alpha=normT[:-1], color='C0'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([rightWristMoCap[tMask][:-1,1:], rightWristMoCap[tMask][1:,1:]], 1), lw=2*(0+rightWristMoCap[tMask][:-1,0]), alpha=normT[:-1], color='C3'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([leftFingerMoCap[tMask][:-1,1:], leftFingerMoCap[tMask][1:,1:]], 1), lw=2*(0+leftFingerMoCap[tMask][:-1,0]), alpha=normT[:-1], color='C2'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([rightFingerMoCap[tMask][:-1,1:], rightFingerMoCap[tMask][1:,1:]], 1), lw=2*(0+rightFingerMoCap[tMask][:-1,0]), alpha=normT[:-1], color='C1'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([leftElbowMoCap[tMask][:-1,1:], leftElbowMoCap[tMask][1:,1:]], 1), lw=2*(0+leftElbowMoCap[tMask][:-1,0]), alpha=normT[:-1], color='C4'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([rightElbowMoCap[tMask][:-1,1:], rightElbowMoCap[tMask][1:,1:]], 1), lw=2*(0+rightElbowMoCap[tMask][:-1,0]), alpha=normT[:-1], color='C5'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([torsoMoCap[tMask][:-1,1:], torsoMoCap[tMask][1:,1:]], 1), lw=2*(0+torsoMoCap[tMask][:-1,0]), alpha=normT[:-1], color='black'))

    # # Top view
    # ax[1,1].add_collection(cln.LineCollection(np.stack([leftWristMoCap[tMask][:-1,:2], leftWristMoCap[tMask][1:,:2]], 1), lw=2*(.5+leftWristMoCap[tMask][:-1,2]), alpha=normT[:-1], color='C0'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([rightWristMoCap[tMask][:-1,:2], rightWristMoCap[tMask][1:,:2]], 1), lw=2*(.5+rightWristMoCap[tMask][:-1,2]), alpha=normT[:-1], color='C3'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([leftFingerMoCap[tMask][:-1,:2], leftFingerMoCap[tMask][1:,:2]], 1), lw=2*(.5+leftFingerMoCap[tMask][:-1,2]), alpha=normT[:-1], color='C2'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([rightFingerMoCap[tMask][:-1,:2], rightFingerMoCap[tMask][1:,:2]], 1), lw=2*(.5+rightFingerMoCap[tMask][:-1,2]), alpha=normT[:-1], color='C1'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([leftElbowMoCap[tMask][:-1,:2], leftElbowMoCap[tMask][1:,:2]], 1), lw=2*(.5+leftElbowMoCap[tMask][:-1,2]), alpha=normT[:-1], color='C4'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([rightElbowMoCap[tMask][:-1,:2], rightElbowMoCap[tMask][1:,:2]], 1), lw=2*(.5+rightElbowMoCap[tMask][:-1,2]), alpha=normT[:-1], color='C5'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([torsoMoCap[tMask][:-1,:2], torsoMoCap[tMask][1:,:2]], 1), lw=2*(.5+torsoMoCap[tMask][:-1,2]), alpha=normT[:-1], color='black'))

    # ax[0,1].set_xlim(0.5,1.5)
    # ax[0,1].set_ylim(0,1)
    # ax[0,0].set_xlim(0.5,1.5)
    # ax[0,0].set_ylim(0,1)
    # ax[1,1].set_xlim(0.5,1.5)
    # ax[1,1].set_ylim(0.5,1.5)
    # ax[0,1].grid()
    # ax[0,0].grid()
    # ax[1,1].grid()
    # ax[1,0].set_axis_off()


    # %%
    # fig, ax = plt.subplots(1, 3, figsize=(18,6), subplot_kw={'projection':'3d'})
    # for i, axi in enumerate(ax): axi.view_init(vertical_axis='z', azim=[-90,30,150][i])

    # tMin, tMax = 1114, 1117
    # tMask = (tMin < timeMoCap) * (timeMoCap < tMax)
    # normT = (timeMoCap[tMask] - timeMoCap[tMask][0]) / (timeMoCap[tMask][-1] - timeMoCap[tMask][0])

    # for axi in ax:
    #     axi.add_collection(a3d.Line3DCollection(np.stack([leftWristMoCap[tMask][:-1], leftWristMoCap[tMask][1:]], 1), alpha=normT[:-1], color='C0'))
    #     axi.add_collection(a3d.Line3DCollection(np.stack([rightWristMoCap[tMask][:-1], rightWristMoCap[tMask][1:]], 1), alpha=normT[:-1], color='C3'))
    #     axi.add_collection(a3d.Line3DCollection(np.stack([leftFingerMoCap[tMask][:-1], leftFingerMoCap[tMask][1:]], 1), alpha=normT[:-1], color='C2'))
    #     axi.add_collection(a3d.Line3DCollection(np.stack([rightFingerMoCap[tMask][:-1], rightFingerMoCap[tMask][1:]], 1), alpha=normT[:-1], color='C1'))

    #     axi.set_xlim(.5,1.5)
    #     axi.set_ylim(.5,1.5)
    #     axi.set_zlim(0,1)
    #     axi.set_xlabel('X')
    #     axi.set_ylabel('Y')
    #     axi.set_zlabel('Z')
    #     axi.grid()

    # %%
    leftWristPosRel = np.einsum('nij,nj->ni', np.stack([lateralMoCap, frontMoCap, zMoCap], 1), leftWristMoCap - torsoMoCap * np.array([[1,1,1]])) + np.array([[0,0.3,0]])
    rightWristPosRel = np.einsum('nij,nj->ni', np.stack([lateralMoCap, frontMoCap, zMoCap], 1), rightWristMoCap - torsoMoCap * np.array([[1,1,1]])) + np.array([[0,0.3,0]])
    leftFingerPosRel = np.einsum('nij,nj->ni', np.stack([lateralMoCap, frontMoCap, zMoCap], 1), leftFingerMoCap - torsoMoCap * np.array([[1,1,1]])) + np.array([[0,0.3,0]])
    rightFingerPosRel = np.einsum('nij,nj->ni', np.stack([lateralMoCap, frontMoCap, zMoCap], 1), rightFingerMoCap - torsoMoCap * np.array([[1,1,1]])) + np.array([[0,0.3,0]])

    # %%
    # fig, ax = plt.subplots(2, 2, figsize=(16,15))

    # tMin, tMax = 1114, 1117
    # tMask = (tMin < timeMoCap) * (timeMoCap < tMax)
    # normT = (timeMoCap[tMask] - timeMoCap[tMask][0]) / (timeMoCap[tMask][-1] - timeMoCap[tMask][0])

    # # Back view
    # ax[0,1].add_collection(cln.LineCollection(np.stack([leftWristPosRel[tMask][:-1,[0,2]], leftWristPosRel[tMask][1:,[0,2]]], 1), lw=-2*(1+leftWristPosRel[tMask][:-1,1]), alpha=normT[:-1], color='C0'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([rightWristPosRel[tMask][:-1,[0,2]], rightWristPosRel[tMask][1:,[0,2]]], 1), lw=-2*(1+rightWristPosRel[tMask][:-1,1]), alpha=normT[:-1], color='C3'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([leftFingerPosRel[tMask][:-1,[0,2]], leftFingerPosRel[tMask][1:,[0,2]]], 1), lw=-2*(1+leftFingerPosRel[tMask][:-1,1]), alpha=normT[:-1], color='C2'))
    # ax[0,1].add_collection(cln.LineCollection(np.stack([rightFingerPosRel[tMask][:-1,[0,2]], rightFingerPosRel[tMask][1:,[0,2]]], 1), lw=-2*(1+rightFingerPosRel[tMask][:-1,1]), alpha=normT[:-1], color='C1'))

    # # Side view
    # ax[0,0].add_collection(cln.LineCollection(np.stack([leftWristPosRel[tMask][:-1,1:], leftWristPosRel[tMask][1:,1:]], 1), lw=-2*(1+leftWristPosRel[tMask][:-1,1]), alpha=normT[:-1], color='C0'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([rightWristPosRel[tMask][:-1,1:], rightWristPosRel[tMask][1:,1:]], 1), lw=-2*(1+rightWristPosRel[tMask][:-1,1]), alpha=normT[:-1], color='C3'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([leftFingerPosRel[tMask][:-1,1:], leftFingerPosRel[tMask][1:,1:]], 1), lw=-2*(1+leftFingerPosRel[tMask][:-1,1]), alpha=normT[:-1], color='C2'))
    # ax[0,0].add_collection(cln.LineCollection(np.stack([rightFingerPosRel[tMask][:-1,1:], rightFingerPosRel[tMask][1:,1:]], 1), lw=-2*(1+rightFingerPosRel[tMask][:-1,1]), alpha=normT[:-1], color='C1'))

    # # Top view
    # ax[1,1].add_collection(cln.LineCollection(np.stack([leftWristPosRel[tMask][:-1,:2], leftWristPosRel[tMask][1:,:2]], 1), lw=2*(1+leftWristPosRel[tMask][:-1,2]), alpha=normT[:-1], color='C0'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([rightWristPosRel[tMask][:-1,:2], rightWristPosRel[tMask][1:,:2]], 1), lw=2*(1+rightWristPosRel[tMask][:-1,2]), alpha=normT[:-1], color='C3'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([leftFingerPosRel[tMask][:-1,:2], leftFingerPosRel[tMask][1:,:2]], 1), lw=2*(1+leftFingerPosRel[tMask][:-1,2]), alpha=normT[:-1], color='C2'))
    # ax[1,1].add_collection(cln.LineCollection(np.stack([rightFingerPosRel[tMask][:-1,:2], rightFingerPosRel[tMask][1:,:2]], 1), lw=2*(1+rightFingerPosRel[tMask][:-1,2]), alpha=normT[:-1], color='C1'))

    # ax[0,1].set_xlim(-.5,.5)
    # ax[0,0].set_xlim(-.5,.5)
    # ax[1,1].set_xlim(-.5,.5)
    # ax[0,1].set_ylim(-.5,.5)
    # ax[0,0].set_ylim(-.5,.5)
    # ax[1,1].set_ylim(-.5,.5)
    # ax[0,1].set_ylabel('Vertical')
    # ax[0,0].set_ylabel('Vertical')
    # ax[1,1].set_ylabel('Frontal')
    # ax[0,1].set_xlabel('Lateral')
    # ax[0,0].set_xlabel('Frontal')
    # ax[1,1].set_xlabel('Lateral')
    # ax[0,1].grid()
    # ax[0,0].grid()
    # ax[1,1].grid()
    # ax[1,0].set_axis_off()


    # %%
    # fig, ax = plt.subplots(1, 3, figsize=(18,6), subplot_kw={'projection':'3d'})
    # for i, axi in enumerate(ax): axi.view_init(vertical_axis='z', azim=[-90,30,150][i])

    # tMin, tMax = 1114, 1117
    # tMask = (tMin < timeMoCap) * (timeMoCap < tMax)
    # normT = (timeMoCap[tMask] - timeMoCap[tMask][0]) / (timeMoCap[tMask][-1] - timeMoCap[tMask][0])

    # for axi in ax:
    #     axi.add_collection(a3d.Line3DCollection(np.stack([leftWristPosRel[tMask][:-1], leftWristPosRel[tMask][1:]], 1), alpha=normT[:-1], color='C0'))
    #     axi.add_collection(a3d.Line3DCollection(np.stack([rightWristPosRel[tMask][:-1], rightWristPosRel[tMask][1:]], 1), alpha=normT[:-1], color='C3'))
    #     axi.add_collection(a3d.Line3DCollection(np.stack([leftFingerPosRel[tMask][:-1], leftFingerPosRel[tMask][1:]], 1), alpha=normT[:-1], color='C2'))
    #     axi.add_collection(a3d.Line3DCollection(np.stack([rightFingerPosRel[tMask][:-1], rightFingerPosRel[tMask][1:]], 1), alpha=normT[:-1], color='C1'))

    #     axi.set_xlim(-.5,.5)
    #     axi.set_ylim(-.5,.5)
    #     axi.set_zlim(-.5,.5)
    #     axi.set_xlabel('Lateral')
    #     axi.set_ylabel('Frontal')
    #     axi.set_zlabel('Vertical')
    #     axi.grid()


    # %%
    # fig, ax = plt.subplots(1, 3, figsize=(18,6), subplot_kw={'projection':'3d'})
    # for i, axi in enumerate(ax): axi.view_init(vertical_axis='z', azim=[-90,30,150][i])

    # pointsAll = []
    # for fi in MoCapFeat:
    #     if fi[3] > 2: continue
    #     tMin, tMax = fi[0], fi[1]
    #     tMask = (tMin < timeMoCap) * (timeMoCap < tMax)
    #     if tMask.sum() < 2: continue
    #     normT = (timeMoCap[tMask] - timeMoCap[tMask][0]) / (timeMoCap[tMask][-1] - timeMoCap[tMask][0])

    #     for axi in ax:
    #         if fi[6] == 0:
    #             axi.add_collection(a3d.Line3DCollection(np.stack([leftWristPosRel[tMask][:-1], leftWristPosRel[tMask][1:]], 1), alpha=normT[:-1], color='C0'))
    #             pointsAll += list(leftWristPosRel[tMask][[0,-1]]*np.array([[-1,1,1]]))
    #         if fi[6] == 1:
    #             axi.add_collection(a3d.Line3DCollection(np.stack([rightWristPosRel[tMask][:-1], rightWristPosRel[tMask][1:]], 1), alpha=normT[:-1], color='C3'))
    #             pointsAll += list(rightWristPosRel[tMask][[0,-1]])

    #         axi.set_xlim(-.5,.5)
    #         axi.set_ylim(0,1)
    #         axi.set_zlim(-.5,.5)
    #         axi.set_xlabel('Lateral')
    #         axi.set_ylabel('Frontal')
    #         axi.set_zlabel('Vertical')
    #         axi.grid()

    # %%
    # fig, ax = plt.subplots(1, 3, figsize=(18,6), subplot_kw={'projection':'3d'})
    # for i, axi in enumerate(ax): axi.view_init(vertical_axis='z', azim=[-90,30,150][i])

    # pointsAll = []
    # for fi in MoCapFeat:
    #     if fi[3] != 3: continue
    #     tMin, tMax = fi[0], fi[1]
    #     tMask = (tMin < timeMoCap) * (timeMoCap < tMax)
    #     if tMask.sum() < 2: continue
    #     normT = (timeMoCap[tMask] - timeMoCap[tMask][0]) / (timeMoCap[tMask][-1] - timeMoCap[tMask][0])

    #     for axi in ax:
    #         if fi[6] == 0:
    #             axi.add_collection(a3d.Line3DCollection(np.stack([leftWristPosRel[tMask][:-1], leftWristPosRel[tMask][1:]], 1), alpha=normT[:-1], color='C0'))
    #             pointsAll += list(leftWristPosRel[tMask][[0,-1]]*np.array([[-1,1,1]]))
    #         if fi[6] == 1:
    #             axi.add_collection(a3d.Line3DCollection(np.stack([rightWristPosRel[tMask][:-1], rightWristPosRel[tMask][1:]], 1), alpha=normT[:-1], color='C3'))
    #             pointsAll += list(rightWristPosRel[tMask][[0,-1]])

    #         axi.set_xlim(-.5,.5)
    #         axi.set_ylim(0,1)
    #         axi.set_zlim(-.5,.5)
    #         axi.set_xlabel('Lateral')
    #         axi.set_ylabel('Frontal')
    #         axi.set_zlabel('Vertical')
    #         axi.grid()

    # %%
    # fig, ax = plt.subplots(1, 3, figsize=(18,6), subplot_kw={'projection':'3d'})
    # for i, axi in enumerate(ax): axi.view_init(vertical_axis='z', azim=[-90,30,150][i])

    # pointsAll = np.array(pointsAll)
    # edgesPoints = np.arange(-.5,.52,.02), np.arange(0,1.02,.02), np.arange(-.5,.52,.02)
    # histPoints, _ = np.histogramdd(pointsAll, bins=edgesPoints, density=True)
    # for i, axi in enumerate(ax):
    #     norm = clr.Normalize()
    #     mgrid = np.mgrid[-.5:.52:.02,0:1.02:.02,-.5:.52:.02]
    #     axi.plot_surface(*mgrid[:,[0,0,-1][i]], facecolors=plt.get_cmap('Blues')(norm(histPoints.sum(0))), zorder=0)
    #     axi.plot_surface(*mgrid[:,:,[-1,0,0][i]], facecolors=plt.get_cmap('Blues')(norm(histPoints.sum(1))), zorder=0)
    #     axi.plot_surface(*mgrid[...,0], facecolors=plt.get_cmap('Blues')(norm(histPoints.sum(2))), zorder=0)
    #     axi.set_xlim(-.5,.5)
    #     axi.set_ylim(0,1)
    #     axi.set_zlim(-.5,.5)
    #     axi.set_xlabel('Lateral')
    #     axi.set_ylabel('Frontal')
    #     axi.set_zlabel('Vertical')
    #     axi.grid()


    # %%
    # winDataFrame = pd.read_csv('/Users/igavier/Documents/PhD-nonsync/feature_extraction/data_real_singleP2P_MoCapfeaturesAut.csv')
    # classificationDataFrame = pd.DataFrame(data=np.load('/Users/igavier/Downloads/classification.npy'), columns=['p2p_score','p2p_gt','winID'])

    # %%
    # p2pDataFrame = pd.DataFrame(columns=['subject','hand','lateral_i','frontal_i','vertical_i',
    #                                      'lateral_f','frontal_f','vertical_f',
    #                                      'lateral_delta','frontal_delta','vertical_delta',
    #                                      'log_duration','log_length','log_energy','activity','misclassified'])

    # for i, fi in enumerate(MoCapFeat):
    #     if fi[3] > 2: continue
    #     tMin, tMax = fi[0], fi[1]
    #     tMask = (tMin < timeMoCap) * (timeMoCap < tMax)
    #     tm = sum(tMask) // 2

    #     misclassified = False
    #     condition = (winDataFrame['startT'] <= tMin) * (winDataFrame['startT'] > tMax - 3) * (winDataFrame['subjID'] == SUBJID)
    #     if np.any(condition):
    #         winIDs = winDataFrame.loc[condition]['# winID'].to_list()
    #         filtered = classificationDataFrame.loc[classificationDataFrame['winID'].isin(winIDs)]
    #         if np.any(filtered['p2p_score'] < 0.5): misclassified = True

    #     if fi[6] == 0: p2pDataFrame.loc[i] = [SUBJID, 0, *leftWristPosRel[tMask][0], *leftWristPosRel[tMask][-1],
    #                                           *(leftWristPosRel[tMask][-1]-leftWristPosRel[tMask][0]),
    #                                           np.log10(fi[2]), np.log10(fi[4]), np.log10(fi[8]),
    #                                           ['reposition','reaching','transport'][int(fi[3])], misclassified]
    #     if fi[6] == 1: p2pDataFrame.loc[i] = [SUBJID, 1, *rightWristPosRel[tMask][0], *rightWristPosRel[tMask][-1],
    #                                           *(rightWristPosRel[tMask][-1]-rightWristPosRel[tMask][0]),
    #                                           np.log10(fi[2]), np.log10(fi[4]), np.log10(fi[8]),
    #                                           ['reposition','reaching','transport'][int(fi[3])], misclassified]

    # %%
    # lw = plt.rcParams["contour.linewidth"]
    # plt.rcParams["contour.linewidth"] = 0
    # gp = sns.pairplot(p2pDataFrame, vars=['lateral_delta', 'frontal_delta', 'vertical_delta'],
    #                   hue='hand', kind='kde')
    # gp.map_offdiag(sns.kdeplot, fill=True, alpha=0.5, common_norm=False)
    # plt.rcParams["contour.linewidth"] = lw

    # %%
    # lw = plt.rcParams["contour.linewidth"]
    # plt.rcParams["contour.linewidth"] = 0
    # gp = sns.pairplot(p2pDataFrame, vars=['log_duration','log_length','log_energy'],
    #                   hue='hand', kind='kde')
    # gp.map_offdiag(sns.kdeplot, fill=True, alpha=0.5, common_norm=False)
    # plt.rcParams["contour.linewidth"] = lw

    # %%
    degree = 5
    polyCoeffToP = np.array([[1,    0  ,    0    ,  0     , 0      ,0           ],
                            [1,    1/5,    1/25 ,  1/125 , 1/625  ,1/3125      ],
                            [1,    2/5,    4/25 ,  8/125 , 16/625 ,32/3125     ],
                            [1,    3/5,    9/25 ,  27/125, 81/625 ,243/3125    ],
                            [1,    4/5,    16/25,  64/125, 256/625,1024/3125   ],
                            [1,    1  ,    1    ,  1     , 1      ,1           ]])
    polyPToCoeff = np.linalg.inv(polyCoeffToP)
    polyCoeffToPVDVVP = np.array([[1,    1/2,    1/4,    1/8,    1/16,   1/32    ],
                                [0,    1  ,    1  ,    3/4,    1/2 ,   5/16    ],
                                [0,    1  ,    1  ,    1  ,    1   ,   1       ],
                                [0,    1  ,    0  ,    0  ,    0   ,   0       ],
                                [0,    1  ,    2  ,    3  ,    4   ,   5       ],
                                [1,    1/2,    1/2,    1/2,    1/2 ,   1/2     ]])
    polyPVDVVPToCoeff = np.linalg.inv(polyCoeffToPVDVVP)

    movementsFeatures = []
    polyFeat = PolynomialFeatures(degree=degree, include_bias=False)
    linReg = LinearRegression()
    tMaskP2P = np.zeros_like(timeMoCap)

    for fi in MoCapFeat:
        # if fi[3] > 2: continue
        tMin, tMax = fi[0], fi[1]
        tMask = (tMin < timeMoCap) * (timeMoCap < tMax)
        if tMask.sum() < 2: continue
        tMaskP2P = np.logical_or(tMaskP2P, tMask)
        timePoly = polyFeat.fit_transform((timeMoCap[tMask,None] - timeMoCap[tMask][0]) /
                                        (timeMoCap[tMask][-1] - timeMoCap[tMask][0]))
        timeRedPoly = np.linspace(0, 1, polyFeat.degree + 1)[:,None]

        features = [np.log10(fi[2]), fi[6], fi[3]]
        for xyz in range(3):
            if fi[6] == 0:
                linReg.fit(timePoly, leftWristPosRel[tMask,xyz])
                features += np.einsum('ij,j->i', polyCoeffToPVDVVP, np.insert(linReg.coef_, 0, linReg.intercept_)).tolist()
                # features += linReg.predict(polyFeat.fit_transform(timeRedPoly)).tolist()
            if fi[6] == 1:
                linReg.fit(timePoly, rightWristPosRel[tMask,xyz])
                features += np.einsum('ij,j->i', polyCoeffToPVDVVP, np.insert(linReg.coef_, 0, linReg.intercept_)).tolist()
                # features += linReg.predict(polyFeat.fit_transform(timeRedPoly)).tolist()
        
        movementsFeatures += [features]

    movementsFeatures = np.array(movementsFeatures)
    np.save(f'./feature_extraction/Subject{SUBJID:03d}_MoCapTraj_feat_v2.npy', movementsFeatures)


