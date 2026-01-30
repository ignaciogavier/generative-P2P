#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import chi, lognorm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as mtransforms
import matplotlib.colors as cls
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Nimbus Sans']
plt.rcParams['font.size'] = 8
import seaborn as sn
import mpl_toolkits.mplot3d.art3d as a3d
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import KernelDensity
from umap import UMAP


# In[3]:


dTMoCap = 1./150
degree = 5
subjectTarget = 20


# In[4]:


# Converts quintic polynomial coefficients to keyframe positions at 0, 0.2, ..., 1
polyCoeffToP = np.array([[1,    0  ,    0    ,  0     , 0      ,0           ],
                         [1,    1/5,    1/25 ,  1/125 , 1/625  ,1/3125      ],
                         [1,    2/5,    4/25 ,  8/125 , 16/625 ,32/3125     ],
                         [1,    3/5,    9/25 ,  27/125, 81/625 ,243/3125    ],
                         [1,    4/5,    16/25,  64/125, 256/625,1024/3125   ],
                         [1,    1  ,    1    ,  1     , 1      ,1           ]])
polyPToCoeff = np.linalg.inv(polyCoeffToP)

# Converts quintic polynomial coefficients to kinematic features
# (pos and vel at 0.5, delta pos 1-0, vel at 0, vel at 1, mean pos at 0 and 1)
polyCoeffToPVDVVP = np.array([[1,    1/2,    1/4,    1/8,    1/16,   1/32    ],
                              [0,    1  ,    1  ,    3/4,    1/2 ,   5/16    ],
                              [0,    1  ,    1  ,    1  ,    1   ,   1       ],
                              [0,    1  ,    0  ,    0  ,    0   ,   0       ],
                              [0,    1  ,    2  ,    3  ,    4   ,   5       ],
                              [1,    1/2,    1/2,    1/2,    1/2 ,   1/2     ]])
polyPVDVVPToCoeff = np.linalg.inv(polyCoeffToPVDVVP)


# In[5]:


allFeatures = []
subjects = []
labels = []
falsenegatives = []
truepositives = []

for SUBJID in range(1,21):
    if SUBJID in [7, 9]: continue

    mFT = np.load(f'./mocap-traj-feat/Subject{SUBJID:03d}_MoCapTraj_feat_v2.npy') # expressed in PVDVVP
    # mFT[:,2:20] = np.einsum('ij,ndj->ndi', polyCoeffToP @ polyPVDVVPToCoeff, mFT[:,2:20].reshape((-1,3,6))).reshape((-1,18))
    allFeatures += [np.hstack([mFT[:,:2], mFT[:,3:]])]
    subjects += [SUBJID] * len(mFT)
    labels += list(mFT[:,2])

allFeatures = np.concatenate(allFeatures, 0)
subjects = np.array(subjects)
labels = np.array(labels)

allGRABFeatures = np.load(f'/home/igavier_umass_edu/Documents/synthetic-pipeline/grab-traj-feat/GRAB_MoCapTraj_feat.npy') # expressed in PVDVVP
labelsGRAB = allGRABFeatures[:,2]
allGRABFeatures = np.hstack([allGRABFeatures[:,:2], allGRABFeatures[:,3:]])


# In[6]:


umap = UMAP(n_neighbors=20, n_components=2, random_state=0, min_dist=1)

allEmbeddings = umap.fit_transform(np.concatenate([allFeatures[:,0:1], allFeatures[:,2:8] * (2 * allFeatures[:,1:2] - 1), allFeatures[:,8:]], 1), y=labels)
allGRABEmbeddings = umap.transform(np.concatenate([allGRABFeatures[:,0:1], allGRABFeatures[:,2:8] * (2 * allGRABFeatures[:,1:2] - 1), allGRABFeatures[:,8:]], 1))


# In[7]:


allP2PFeatures = np.concatenate([allFeatures[:,0:1], allFeatures[:,2:8] * (2 * allFeatures[:,1:2] - 1), allFeatures[:,8:]], 1)[(labels<3)]
allNonP2PFeatures = np.concatenate([allFeatures[:,0:1], allFeatures[:,2:8] * (2 * allFeatures[:,1:2] - 1), allFeatures[:,8:]], 1)[(labels>=3)]
allGRABP2PFeatures = np.concatenate([allGRABFeatures[:,0:1], allGRABFeatures[:,2:8] * (2 * allGRABFeatures[:,1:2] - 1), allGRABFeatures[:,8:]], 1)[labelsGRAB==0]
allGRABNonP2PFeatures = np.concatenate([allGRABFeatures[:,0:1], allGRABFeatures[:,2:8] * (2 * allGRABFeatures[:,1:2] - 1), allGRABFeatures[:,8:]], 1)[labelsGRAB==3]

kde = KernelDensity(bandwidth=5e-2).fit((allP2PFeatures - allP2PFeatures.mean(0)[None,:]) / allP2PFeatures.std(0)[None,:])
kde_neg = KernelDensity(bandwidth=1e-2).fit((allNonP2PFeatures - allNonP2PFeatures.mean(0)[None,:]) / allNonP2PFeatures.std(0)[None,:])
kdeGRAB = KernelDensity(bandwidth=5e-2).fit((allGRABP2PFeatures - allGRABP2PFeatures.mean(0)[None,:]) / allGRABP2PFeatures.std(0)[None,:])
kdeGRAB_neg = KernelDensity(bandwidth=1e-2).fit((allGRABNonP2PFeatures - allGRABNonP2PFeatures.mean(0)[None,:]) / allGRABNonP2PFeatures.std(0)[None,:])
# kde_target = KernelDensity(bandwidth=.01).fit(allP2PFeatures, sample_weight=(1+10*falsenegatives[labels==1])/np.exp(kde.score_samples(allP2PFeatures)))


# In[8]:


allP2PFeatures.shape


# In[9]:


numGenerated = 10000

# Generate using all P2P
# newEmbeddings = kde.sample(numGenerated)
# newMovements = umap.inverse_transform(newEmbeddings)

newMovements = kde.sample(numGenerated)
newMovements = newMovements * allP2PFeatures.std(0)[None,:] + allP2PFeatures.mean(0)[None,:]
hands = np.random.randint(2, size=len(newMovements))[:,None]
newEmbeddings = umap.transform(newMovements)
newMovementsNeg = kde_neg.sample(numGenerated) # multiply by 8 if using 12.5% P2P
newMovementsNeg = newMovementsNeg * allNonP2PFeatures.std(0)[None,:] + allNonP2PFeatures.mean(0)[None,:]
handsNeg = np.random.randint(2, size=len(newMovementsNeg))[:,None]
newEmbeddingsNeg = umap.transform(newMovementsNeg)

# newMovements = kdeGRAB.sample(numGenerated)
# newMovements = newMovements * allGRABP2PFeatures.std(0)[None,:] + allGRABP2PFeatures.mean(0)[None,:]
# hands = np.random.randint(2, size=len(newMovements))[:,None]
# newEmbeddings = umap.transform(newMovements)
# newMovementsNeg = kdeGRAB_neg.sample(numGenerated) # multiply by 8 if using 12.5% P2P
# newMovementsNeg = newMovementsNeg * allGRABNonP2PFeatures.std(0)[None,:] + allGRABNonP2PFeatures.mean(0)[None,:]
# handsNeg = np.random.randint(2, size=len(newMovementsNeg))[:,None]
# newEmbeddingsNeg = umap.transform(newMovementsNeg)

# newMovementsTarget = kde.sample(numGenerated)
# handsTarget = np.random.randint(2, size=len(newMovementsTarget))[:,None]


# In[12]:


fig, ax = plt.subplots(1, 3, figsize=(18,6), subplot_kw={'projection':'3d'})
for i, axi in enumerate(ax): axi.view_init(vertical_axis='z', azim=[-90,30,150][i])

pointsAll = []
# perm = np.random.permutation(len(allFeatures))[:100]
# print(allFeatures[perm].shape)
for nm, h in zip(newMovements[:100], hands.flatten()[:100]):
# for nm, h in zip(allFeatures[perm][:,[0]+list(range(2,20))],
#                  allFeatures[perm][:,1]):
    # h = int(h)
    for axi in ax:
        # keysT = np.linspace(0, 1, polyFeat.degree + 1)
        # coeffs = [lagrange(keysT, nm[2:].reshape(3,-1)[xyz]).coef[::-1] for xyz in range(3)]
        coeffs = [np.einsum('ij,j->i', polyPVDVVPToCoeff, nm[1:].reshape(3,-1)[xyz]) for xyz in range(3)]
        # coeffs = [np.einsum('ij,j->i', polyPToCoeff, nm[1:].reshape(3,-1)[xyz]) for xyz in range(3)]

        time = np.arange(int(10**nm[0] / dTMoCap)) * dTMoCap
        pos = np.stack([Polynomial(c)(time / time[-1]) for c in coeffs], 1) * np.array([[2*h-1,1,1]])
        if np.linalg.norm(pos[-1] - pos[0]) < 0.1: continue

        normT = (time - time[0]) / (time[-1] - time[0])

        axi.add_collection(a3d.Line3DCollection(np.stack([pos[:-1], pos[1:]], 1), alpha=normT[:-1], color=f'C{3*h}'))

        axi.set_xlim(-.5,.5)
        axi.set_ylim(0,1)
        axi.set_zlim(-.5,.5)
        axi.set_xlabel('Lateral')
        axi.set_ylabel('Frontal')
        axi.set_zlabel('Vertical')
        axi.grid()

fig.savefig('./figures/3d_synth_movements.pdf', dpi=150)


# In[10]:


fig, ax = plt.subplots(1, 3, figsize=(18,6), subplot_kw={'projection':'3d'})
for i, axi in enumerate(ax): axi.view_init(vertical_axis='z', azim=[-90,30,150][i])

pointsAll = []
for nm, h in zip(newMovementsNeg[:100], handsNeg.flatten()[:100]):
    for axi in ax:
        # keysT = np.linspace(0, 1, polyFeat.degree + 1)
        # coeffs = [lagrange(keysT, nm[2:].reshape(3,-1)[xyz]).coef[::-1] for xyz in range(3)]
        coeffs = [np.einsum('ij,j->i', polyPVDVVPToCoeff, nm[1:].reshape(3,-1)[xyz]) for xyz in range(3)]
        # coeffs = [np.einsum('ij,j->i', polyPToCoeff, nm[1:].reshape(3,-1)[xyz]) for xyz in range(3)]

        time = np.arange(int(10**nm[0] / dTMoCap)) * dTMoCap
        pos = np.stack([Polynomial(c)(time / time[-1]) for c in coeffs], 1) * np.array([[2*h-1,1,1]])

        normT = (time - time[0]) / (time[-1] - time[0])

        axi.add_collection(a3d.Line3DCollection(np.stack([pos[:-1], pos[1:]], 1), alpha=normT[:-1], color=f'C{3*h}'))

        axi.set_xlim(-.5,.5)
        axi.set_ylim(0,1)
        axi.set_zlim(-.5,.5)
        axi.set_xlabel('Lateral')
        axi.set_ylabel('Frontal')
        axi.set_zlabel('Vertical')
        axi.grid()


# In[11]:


# transformMat = np.eye(6) # 
transformMat = polyCoeffToP @ polyPVDVVPToCoeff

# Correct new movements to include hands information
newMovements = np.concatenate([newMovements[:,0:1], hands,
                               np.einsum('ij,nj->ni', transformMat, (2 * hands - 1) * newMovements[:,1:7]),
                               np.einsum('ij,nj->ni', transformMat, newMovements[:,7:13]),
                               np.einsum('ij,nj->ni', transformMat, newMovements[:,13:20])], 1)

# newMovementsTarget = np.concatenate([newMovementsTarget[:,0:1], handsTarget,
#                                      np.einsum('ij,nj->ni', transformMat, (2 * handsTarget - 1) * newMovementsTarget[:,1:7]),
#                                      np.einsum('ij,nj->ni', transformMat, newMovementsTarget[:,7:13]),
#                                      np.einsum('ij,nj->ni', transformMat, newMovementsTarget[:,13:20])], 1)

newMovementsNeg = np.concatenate([newMovementsNeg[:,0:1], handsNeg,
                                  np.einsum('ij,nj->ni', transformMat, (2 * handsNeg - 1) * newMovementsNeg[:,1:7]),
                                  np.einsum('ij,nj->ni', transformMat, newMovementsNeg[:,7:13]),
                                  np.einsum('ij,nj->ni', transformMat, newMovementsNeg[:,13:20])], 1)

# np.save(f'mocap-synthtraj-feat/skeletons_from_mocapS{subjectTarget}_10000trajectory_v2.npy', newMovements)
# np.save(f'mocap-synthtraj-feat/skeletons_from_mocapS{subjectTarget}_10000trajectory_neg_v2.npy', newMovementsNeg)
# np.save(f'point-to-point_generation/GRAB/skeletons_from_GRAB_10000trajectory.npy', newMovements)
# np.save(f'point-to-point_generation/GRAB/skeletons_from_GRAB_10000trajectory_neg1.npy', newMovementsNeg)


# In[12]:


fig, ax = plt.subplots(1, 1, figsize=(8,6))

maskLabel = (labels < 3)
maskSubj = (subjects == 10)
mask = maskSubj
sc = ax.scatter(allEmbeddings[mask,0], allEmbeddings[mask,1], c=labels[mask])#np.linalg.norm(allFeatures[mask,4::6], axis=1))
xlim, ylim = ax.get_xlim(), ax.get_ylim()

fig.colorbar(sc, label='Label')


fig, ax = plt.subplots(2, 10, figsize=(24,6), sharex=True, sharey=True)

for i in range(10):
    nnm = (labels != 3)
    maskLabel = (labels < 3)
    maskSubj = (subjects == (i + 1 + (i >= 6) + (i >= 7)))
    ax[0,i].set_title(f'P2P Subject {i + 1 + (i >= 6) + (i >= 7)}')
    ax[1,i].set_title(f'Non P2P Subject {i + 1 + (i >= 6) + (i >= 7)}')
    sc = ax[0,i].hist2d(allEmbeddings[maskLabel*nnm * maskSubj,0], allEmbeddings[maskLabel*nnm * maskSubj,1], bins=50, range=(xlim, ylim), density=True, cmap='turbo', vmin=0, vmax=1/30)
    sc = ax[1,i].hist2d(allEmbeddings[~maskLabel*nnm * maskSubj,0], allEmbeddings[~maskLabel*nnm * maskSubj,1], bins=50, range=(xlim, ylim), density=True, cmap='turbo', vmin=0, vmax=1/10)


# In[13]:


fig, ax = plt.subplots(2, 3, figsize=(14,6))

ax[0,0].set_title(f'P2P all subj (real)')
ax[0,1].set_title(f'P2P Subject 10 (synth)')
ax[0,2].set_title(f'P2P (GRAB)')
ax[1,0].set_title(f'Non all subj (real)')
ax[1,1].set_title(f'Non P2P Subject 10 (synth)')
ax[1,2].set_title(f'Non P2P (GRAB)')

allGRABP2PEmbeddings = allGRABEmbeddings[labelsGRAB == 0]
allGRABNonP2PEmbeddings = allGRABEmbeddings[labelsGRAB == 3]

sc = ax[0,2].hist2d(allGRABP2PEmbeddings[:,0], allGRABP2PEmbeddings[:,1], bins=50, range=(xlim, ylim), density=True, cmap='turbo', vmin=0, vmax=1/30)
sc = ax[1,2].hist2d(allGRABNonP2PEmbeddings[:,0], allGRABNonP2PEmbeddings[:,1], bins=50, range=(xlim, ylim), density=True, cmap='turbo', vmin=0, vmax=1/30)

sc = ax[0,1].hist2d(newEmbeddings[:,0], newEmbeddings[:,1], bins=50, range=(xlim, ylim), density=True, cmap='turbo', vmin=0, vmax=1/30)
sc = ax[1,1].hist2d(newEmbeddingsNeg[:,0], newEmbeddingsNeg[:,1], bins=50, range=(xlim, ylim), density=True, cmap='turbo', vmin=0, vmax=1/30)

maskSubj = (subjects != 0)
sc = ax[0,0].hist2d(allEmbeddings[maskLabel*maskSubj,0], allEmbeddings[maskLabel*maskSubj,1], bins=50, range=(xlim, ylim), density=True, cmap='turbo', vmin=0, vmax=1/30)
sc = ax[1,0].hist2d(allEmbeddings[~maskLabel*maskSubj,0], allEmbeddings[~maskLabel*maskSubj,1], bins=50, range=(xlim, ylim), density=True, cmap='turbo', vmin=0, vmax=1/30)


# In[14]:


def plot_chi(data, ax, fit=False, label=None):
    lims_len = np.nanpercentile(data, [5, 95]) * np.array([.1, 10])
    n, b, _ = ax.hist(data, bins=np.logspace(*np.log10(lims_len), 51, base=10), label=label)

    if fit:
        df, loc, scale = chi.fit(data, floc=0)
        print(df, loc, scale)
        ax.plot(np.sqrt(b[1:] * b[:-1]),
                   n.sum() * (b[1:] - b[:-1]) * chi.pdf(np.sqrt(b[1:] * b[:-1]), df, loc=loc, scale=scale),
                   label=fr'Xi($k=${df:.1f},$a=${scale:.3f})')


fig, ax = plt.subplots(2, 2, figsize=(8,6), sharex='col', sharey='col')

plot_chi(np.linalg.norm(allFeatures[(labels<3),4::6], axis=1), ax[0,0], True, 'Original P2P')
plot_chi(np.linalg.norm(allFeatures[(labels>=3),4::6], axis=1), ax[0,1], False, 'Original NonP2P')
plot_chi(np.linalg.norm(newMovements[:,2::6]-newMovements[:,7::6], axis=1), ax[1,0], True, 'Generated P2P')
plot_chi(np.linalg.norm(newMovementsNeg[:,2::6]-newMovementsNeg[:,7::6], axis=1), ax[1,1], False, 'Generated NonP2P')

ax[0,0].set_ylabel('Count (normalized)')
ax[1,0].set_ylabel('Count (normalized)')
ax[1,0].set_xlabel('Distance [m]')
ax[1,1].set_xlabel('Distance [m]')
ax[1,0].set_xscale('log')
ax[1,1].set_xscale('log')
ax[1,0].set_xlim(1e-3,1)
ax[1,1].set_xlim(1e-5,1)
ax[0,0].legend(loc='lower left')
ax[0,1].legend(loc='lower left')
ax[1,0].legend(loc='lower left')
ax[1,1].legend(loc='lower left')


# In[15]:


def plot_lognorm(data, ax, fit=False, label=None):
    lims_len = np.nanpercentile(data, [5, 95]) * np.array([.1, 10])
    n, b, _ = ax.hist(data, bins=np.logspace(*np.log10(lims_len), 51, base=10), label=label)

    if fit:
        shape, loc, scale = lognorm.fit(data, floc=0)
        print(shape, loc, scale)
        ax.plot(np.sqrt(b[1:] * b[:-1]),
                   n.sum() * (b[1:] - b[:-1]) * lognorm.pdf(np.sqrt(b[1:] * b[:-1]), shape, loc=loc, scale=scale),
                   label=fr'LogNorm($\mu=${np.log(scale):.2f},$\sigma=${shape:.2f})')


fig, ax = plt.subplots(2, 2, figsize=(8,6), sharex='col', sharey='col')

plot_lognorm(10**allFeatures[(labels<3),0], ax[0,0], True, 'Original P2P')
plot_lognorm(10**allFeatures[(labels>=3),0], ax[0,1], False, 'Original NonP2P')
plot_lognorm(10**newMovements[:,0], ax[1,0], True, 'Generated P2P')
plot_lognorm(10**newMovementsNeg[:,0], ax[1,1], False, 'Generated NonP2P')

ax[0,0].set_ylabel('Count (normalized)')
ax[1,0].set_ylabel('Count (normalized)')
ax[1,0].set_xlabel('Duration [s]')
ax[1,1].set_xlabel('Duration [s]')
ax[1,0].set_xscale('log')
ax[1,1].set_xscale('log')
ax[1,0].set_xlim(1e-2,10)
ax[1,1].set_xlim(1e-2,50)
ax[0,0].legend(loc='lower left')
ax[0,1].legend(loc='lower left')
ax[1,0].legend(loc='lower left')
ax[1,1].legend(loc='lower left')


# In[16]:


N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1, plt.get_cmap('tab10')(0)[0], N)
vals[:, 1] = np.linspace(1, plt.get_cmap('tab10')(0)[1], N)
vals[:, 2] = np.linspace(1, plt.get_cmap('tab10')(0)[2], N)
tab0cmp = cls.ListedColormap(vals)
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(1, plt.get_cmap('tab10')(1)[0], N)
vals[:, 1] = np.linspace(1, plt.get_cmap('tab10')(1)[1], N)
vals[:, 2] = np.linspace(1, plt.get_cmap('tab10')(1)[2], N)
tab1cmp = cls.ListedColormap(vals)


# In[17]:


from scipy.special import rel_entr


# In[18]:


def plot_joint(distance, duration, fit=False, lims_dis=None, lims_dur=None, label=None, color='r', g=None):
    if not g: g = sn.JointGrid()
    g.figure.set_size_inches(3, 3, forward=True)
    cmap = {'b':tab0cmp,'r':tab1cmp}[color]

    bdis = np.logspace(*np.log10(lims_dis), 51, base=10)
    ndis, _, _ = g.ax_marg_x.hist(distance, bins=bdis,
                               color=cmap(.8), alpha=1.0)
    # if fit:
    #     df, loc, scale = chi.fit(distance, floc=0)
    #     g.ax_marg_x.plot(np.sqrt(bdis[1:] * bdis[:-1]),
    #                      ndis.sum() * (bdis[1:] - bdis[:-1]) * chi.pdf(np.sqrt(bdis[1:] * bdis[:-1]), df, loc=loc, scale=scale),
    #                      label='Xi', lw=1, color='k')
    #     print(fr'k={df:.3f}', fr'a={scale:.3f}')
    #     g.ax_marg_x.legend(loc='upper left', fontsize=9)

    bdur = np.logspace(*np.log10(lims_dur), 51, base=10)
    ndur, _, _ = g.ax_marg_y.hist(duration, bins=bdur, orientation='horizontal',
                               color=cmap(.8), alpha=1.0)
    # if fit:
    #     shape, loc, scale = lognorm.fit(duration, floc=0)
    #     g.ax_marg_y.plot(ndur.sum() * (bdur[1:] - bdur[:-1]) * lognorm.pdf(np.sqrt(bdur[1:] * bdur[:-1]), shape, loc=loc, scale=scale),
    #                      np.sqrt(bdur[1:] * bdur[:-1]), label='LogNorm', lw=1, color='k', alpha=1.0)
    #     print(fr'\mu={np.log(scale):.3f}', fr'\sigma={shape:.3f}')
    #     g.ax_marg_y.legend(loc='upper right', fontsize=9)

    ndisdur, _, _, _ = g.ax_joint.hist2d(distance, duration, bins=(bdis, bdur), cmap=cmap)
    if fit:
        log_data = np.log10(np.stack([10*np.ones_like(distance), distance, duration], 1))
        coeffs, _, _, _ = np.linalg.lstsq(log_data[:,0:2], log_data[:,2], rcond=None)
        g.ax_joint.plot(bdis[len(bdis)//5:4*len(bdis)//5], 2*bdis[len(bdis)//5:4*len(bdis)//5]**(1/3), '--k',
                        label=fr'Scalability rule', lw=1)
        g.ax_joint.text(0.05, 1, r'$\frac{1}{3}$', fontsize=11)
        g.ax_joint.plot(bdis, 10**coeffs[0]*bdis**coeffs[1], color='k',
                        label=f'Loglog regression\n(slope = {coeffs[1]:.3f})', lw=1)
        print(fr'slope={{{coeffs[1]:.3f}}}')
        g.ax_joint.legend(loc='lower left', fontsize=9)

    g.ax_joint.set_xscale('log')
    g.ax_joint.set_yscale('log')
    g.ax_joint.grid()
    g.ax_joint.set_xlim(*lims_dis)
    g.ax_joint.set_ylim(*lims_dur)
    g.ax_joint.set_xlabel('Distance [$m$]', size=11)
    g.ax_joint.set_ylabel('Duration [$s$]', size=11)
    g.ax_marg_x.set_xscale('log')
    g.ax_marg_y.set_yscale('log')
    g.ax_marg_x.set_xlim(*lims_dis)
    g.ax_marg_y.set_ylim(*lims_dur)
    g.figure.subplots_adjust(left=0.19, bottom=0.14)

    plt.savefig(f'./figures/{label}.pdf', dpi=150)
    plt.savefig(f'./figures/{label}.svg', dpi=150)

    return ndisdur, ndis, ndur

distance_real_p2p, duration_real_p2p = np.linalg.norm(allFeatures[(labels<3),4::6], axis=1), 10**allFeatures[(labels<3),0]
distance_real_nonp2p, duration_real_nonp2p = np.linalg.norm(allFeatures[(labels>=3),4::6], axis=1), 10**allFeatures[(labels>=3),0]
distance_synth_p2p, duration_synth_p2p = np.linalg.norm(newMovements[:,2::6]-newMovements[:,7::6], axis=1), 10**newMovements[:,0]
distance_synth_nonp2p, duration_synth_nonp2p = np.linalg.norm(newMovementsNeg[:,2::6]-newMovementsNeg[:,7::6], axis=1), 10**newMovementsNeg[:,0]

lims_dis_p2p = np.nanpercentile(np.concatenate([distance_real_p2p, distance_synth_p2p]), [5, 95]) * np.array([.25, 4])
lims_dur_p2p = np.nanpercentile(np.concatenate([duration_real_p2p, duration_synth_p2p]), [5, 95]) * np.array([.25, 4])
lims_dis_nonp2p = np.nanpercentile(np.concatenate([distance_real_nonp2p, distance_synth_nonp2p]), [5, 95]) * np.array([.25, 4])
lims_dur_nonp2p = np.nanpercentile(np.concatenate([duration_real_nonp2p, duration_synth_nonp2p]), [5, 95]) * np.array([.25, 4])

plot_joint(distance_real_p2p, duration_real_p2p, True, lims_dis_p2p, lims_dur_p2p, 'joint_durdist_real_p2p', color='r')
plot_joint(distance_real_nonp2p, duration_real_nonp2p, False, lims_dis_nonp2p, lims_dur_nonp2p, 'joint_durdist_real_nonp2p', color='r')
plot_joint(distance_synth_p2p, duration_synth_p2p, True, lims_dis_p2p, lims_dur_p2p, 'joint_durdist_synth_p2p', color='b')
plot_joint(distance_synth_nonp2p, duration_synth_nonp2p, False, lims_dis_nonp2p, lims_dur_nonp2p, 'joint_durdist_synth_nonp2p', color='b')


# In[19]:


def plot_joint(curvature, energy, fit=False, lims_cur=None, lims_ene=None, label=None, color='r'):
    g = sn.JointGrid()
    g.figure.set_size_inches(3, 3, forward=True)
    cmap = {'b':tab0cmp,'r':tab1cmp}[color]

    bcur = np.logspace(*np.log10(lims_cur), 51, base=10)
    ncur, _, _ = g.ax_marg_x.hist(curvature, bins=bcur,
                                  color=cmap(.8))

    bene = np.logspace(*np.log10(lims_ene), 51, base=10)
    nene, _, _ = g.ax_marg_y.hist(energy, bins=bene, orientation='horizontal',
                                  color=cmap(.8))

    ncurene, _, _, _ = g.ax_joint.hist2d(curvature, energy, bins=(bcur, bene), cmap=cmap)
    if fit:
        log_data = np.log10(np.stack([10*np.ones_like(curvature), curvature, energy], 1))
        coeffs, _, _, _ = np.linalg.lstsq(log_data[:,0:2], log_data[:,2], rcond=None)
        g.ax_joint.plot(bcur[len(bcur)//5:4*len(bcur)//5], 2*bcur[len(bcur)//5:4*len(bcur)//5]**(1/3), '--k',
                        label=fr'Control rule', lw=1)
        g.ax_joint.text(0.05, 1, r'$\frac{1}{3}$', fontsize=11)
        g.ax_joint.plot(bcur, 10**coeffs[0]*bcur**coeffs[1], color='k',
                        label=f'Loglog regression\n(slope = {coeffs[1]:.3f})', lw=1)
        print(fr'slope={{{coeffs[1]:.3f}}}')
        g.ax_joint.legend(loc='lower left', fontsize=9)

    g.ax_joint.set_xscale('log')
    g.ax_joint.set_yscale('log')
    g.ax_joint.grid()
    g.ax_joint.set_xlim(*lims_cur)
    g.ax_joint.set_ylim(*lims_ene)
    g.ax_joint.set_xlabel('Radius of curvature [$m$]', size=11)
    g.ax_joint.set_ylabel('Instantaneous velocity [$m s^{-1}$]', size=11)
    g.ax_marg_x.set_xscale('log')
    g.ax_marg_y.set_yscale('log')
    g.ax_marg_x.set_xlim(*lims_cur)
    g.ax_marg_y.set_ylim(*lims_ene)
    g.figure.subplots_adjust(left=0.19, bottom=0.14)

    plt.savefig(f'./figures/{label}.pdf', dpi=150)
    plt.savefig(f'./figures/{label}.svg', dpi=150)

    return ncurene, ncur, nene

def energy_curvature_polynomial(poly, is_pdvpp=True):
    transformMat = polyPVDVVPToCoeff if is_pdvpp else polyPToCoeff
    poly = np.concatenate([poly[:,:2],
                           np.einsum('ij,nj->ni', transformMat, poly[:,2:8]),
                           np.einsum('ij,nj->ni', transformMat, poly[:,8:14]),
                           np.einsum('ij,nj->ni', transformMat, poly[:,14:20])], 1)

    curvature, energy = [], []
    for i in range(len(poly)):
        duration = 10 ** poly[i,0]
        h = poly[i,1]
        coeffs = poly[i,2:20].reshape(3,-1)
        time = np.arange(int(duration / dTMoCap)) * dTMoCap
        pos = np.stack([Polynomial(c)(time / time[-1]) for c in coeffs], 1) * np.array([[2*h-1,1,1]])
        vel = np.gradient(pos, dTMoCap, axis=0)
        acc = np.gradient(vel, dTMoCap, axis=0)
        curvature += [(np.linalg.norm(vel, axis=1) ** 3 /
                       np.linalg.norm(np.cross(vel, acc, axisa=1, axisb=1), axis=1))]
        energy += [(np.linalg.norm(vel, axis=1))]

    return np.concatenate(curvature, 0), np.concatenate(energy, 0)


curvature_real_p2p, energy_real_p2p = energy_curvature_polynomial(allFeatures[labels<3])
curvature_real_nonp2p, energy_real_nonp2p = energy_curvature_polynomial(allFeatures[labels>=3])
curvature_synth_p2p, energy_synth_p2p = energy_curvature_polynomial(newMovements, False)
curvature_synth_nonp2p, energy_synth_nonp2p = energy_curvature_polynomial(newMovementsNeg, False)

lims_cur_p2p = np.nanpercentile(np.concatenate([curvature_real_p2p, curvature_synth_p2p]), [5, 95]) * np.array([.25, 4])
lims_ene_p2p = np.nanpercentile(np.concatenate([energy_real_p2p, energy_synth_p2p]), [5, 95]) * np.array([.25, 4])
lims_cur_nonp2p = np.nanpercentile(np.concatenate([curvature_real_nonp2p, curvature_synth_nonp2p]), [5, 95]) * np.array([.25, 4])
lims_ene_nonp2p = np.nanpercentile(np.concatenate([energy_real_nonp2p, energy_synth_nonp2p]), [5, 95]) * np.array([.25, 4])

plot_joint(curvature_real_p2p, energy_real_p2p, True, lims_cur_p2p, lims_ene_p2p, 'joint_curvel_real_p2p', color='r')
plot_joint(curvature_real_nonp2p, energy_real_nonp2p, False, lims_cur_nonp2p, lims_ene_nonp2p, 'joint_curvel_real_nonp2p', color='r')
plot_joint(curvature_synth_p2p, energy_synth_p2p, True, lims_cur_p2p, lims_ene_p2p, 'joint_curvel_synth_p2p', color='b')
plot_joint(curvature_synth_nonp2p, energy_synth_nonp2p, False, lims_cur_nonp2p, lims_ene_nonp2p, 'joint_curvel_synth_nonp2p', color='b')


# In[23]:


def kl_hist_1d(x1, x2, bins=50, eps=1e-10):
    # shared bin edges so both distributions align
    hist_range = (min(x1.min(), x2.min()), max(x1.max(), x2.max()))
    p_hist, edges = np.histogram(x1, bins=bins, range=hist_range, density=True)
    q_hist, _ = np.histogram(x2, bins=bins, range=hist_range, density=True)

    # normalize to make them probabilities
    p = p_hist / np.sum(p_hist)
    q = q_hist / np.sum(q_hist)

    # add epsilon to avoid log(0)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    return np.sum(p * np.log(p / q))


# In[24]:


print(f'KL distdur = {kl_hist_1d(np.stack([distance_real_p2p, duration_real_p2p], 1), np.stack([distance_synth_p2p, duration_synth_p2p], 1)):.5f}')
print(f'KL distance = {kl_hist_1d(distance_real_p2p, distance_synth_p2p):.5f}')
print(f'KL duration = {kl_hist_1d(duration_real_p2p, duration_synth_p2p):.5f}')


# In[25]:


print(f'KL curvene = {kl_hist_1d(np.stack([curvature_real_p2p, energy_real_p2p], 1), np.stack([curvature_synth_p2p, energy_synth_p2p], 1)):.5f}')
print(f'KL curvature = {kl_hist_1d(curvature_real_p2p, curvature_synth_p2p):.5f}')
print(f'KL energy = {kl_hist_1d(energy_real_p2p, energy_synth_p2p):.5f}')


# In[ ]:




