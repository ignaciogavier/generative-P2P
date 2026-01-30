import os

import numpy as np
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset, random_split

# 3 seconds window at fs = 50 Hz
T = 3 * 50 # 5 * 30 # 

# Labels for synthetic dataset
LBL = {('nleft', 'l'): 0,  ('nleft', 'r'): 0,
       ('pleft', 'l'): 1,  ('pleft', 'r'): 1, # if all P2P synthetic use double hand
       ('nright', 'l'): 0, ('nright', 'r'): 0,
       ('pright', 'l'): 1, ('pright', 'r'): 1} # if all P2P synthetic use double hand

# Labels for real dataset
LBL2 = {('nleftnright', 'l'): 0, ('nleftnright', 'r'): 0,
        ('pleftnright', 'l'): 1, ('pleftnright', 'r'): 0,
        ('nleftpright', 'l'): 0, ('nleftpright', 'r'): 1,
        ('pleftpright', 'l'): 1, ('pleftpright', 'r'): 1}

# Indices for subjects in real dataset
SUBJIDX = np.array([0, 519, 1103, 1748, 2113, 2554, 2951, 3442, 3823, 4288, 4917, 5482, 6163, 6760, 7159, 7895, 8458, 9148, 9526, 9923]) # use this line for all windows without discarding anything
# SUBJIDX = np.array([0, 456, 1001, 1533, 1851, 2233, 2566, 2996, 3343, 3760, 4295, 4767, 5359, 5863, 6239, 6624, 7048, 7624, 7953, 8276])
# SUBJIDX = np.array([0, 344, 742, 1104, 1300, 1560, 1780, 2087, 2317, 2618, 3002, 3372, 3817, 4161, 4432, 4668, 4982, 5391, 5583, 5773]) # use this line for single P2P
# SUBJIDXman = np.array([0, 341, 731, 1087, 1279, 1535, 1754, 2056, 2285, 2577, 2955, 3324, 3763, 4101, 4368, 4599, 4908, 5309, 5496, 5685]) # use this line for single P2P with manual labels
# SUBJIDXaut = np.array([0, 299, 639, 925, 1075, 1278, 1463, 1719, 1911, 2147, 2485, 2811, 3184, 3446, 3683, 3883, 4165, 4513, 4677, 4847]) # use this line for single P2P with automatic+manual labels
SUBJIDX = SUBJIDX ############################################
SUBJIDXstroke = np.array([0, 959, 1790, 3039, 3424, 4201, 4984, 5407, 6119, 6720, 7316, 7871, 9003, 9463, 9711, 10250, 10737, 11212])

# Features extracted from real dataset
# data_real_features = np.loadtxt('/home/igavier_umass_edu/Documents/P2P-classifier/data_real_features.csv',
#                                 skiprows=1, delimiter=',')
data_real_manfeatures = np.loadtxt('/home/igavier_umass_edu/Documents/P2P-classifier/data_real_singleP2P_man.csv',
                                skiprows=1, delimiter=',') # use this line for single P2P with manual features
data_real_autfeatures = np.loadtxt('/home/igavier_umass_edu/Documents/P2P-classifier/data_real_singleP2P_MoCapfeaturesAut.csv',
                                skiprows=1, delimiter=',') # use this line for single P2P
########################################################################################
########################################################################################
########################################################################################
data_real_features = data_real_manfeatures
########################################################################################
########################################################################################
########################################################################################

data_stroke_features = np.loadtxt('/home/igavier_umass_edu/Documents/P2P-classifier/data_stroke_singleP2P_man.csv',
                                skiprows=1, delimiter=',') # use this line for single P2P

data_synth_features_p2p = np.genfromtxt('/home/igavier_umass_edu/Documents/P2P-classifier/data_synth_features_p2p.csv',
                                 skip_header=1, delimiter=',')
data_synth_features_all = np.genfromtxt('/home/igavier_umass_edu/Documents/P2P-classifier/data_synth_features_all.csv',
                                 skip_header=1, delimiter=',')
data_synth_features_s10 = np.genfromtxt('/home/igavier_umass_edu/Documents/P2P-classifier/data_synth_features_s10.csv',
                                 skip_header=1, delimiter=',')
data_synth_features_s10_10000 = np.genfromtxt('/home/igavier_umass_edu/Documents/P2P-classifier/data_synth_features_s10_10000.csv',
                                 skip_header=1, delimiter=',')
########################################################################################
########################################################################################
########################################################################################
data_synth_features = data_synth_features_s10_10000
########################################################################################
########################################################################################
########################################################################################

# Interpolation function for synth dataset with low sampling rate
interp = lambda x: CubicSpline(np.arange(len(x)), x, bc_type='natural')(np.linspace(0, 36, T, endpoint=False))

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, file_names, is_synth=True, is_stroke=False, augment=False, classifier=True, with_gyro=True, with_acc=True):
        self.file_names = file_names
        self.is_synth = is_synth
        self.is_stroke = is_stroke
        self.augment = augment
        self.classifier = classifier
        self.with_gyro = with_gyro
        self.with_acc = with_acc

    def __len__(self):
        return len(self.file_names) // 2 # num of files divided by 2 (acc and gyr)

    def __getitem__(self, idx):
        files_idx = self.file_names[2*idx:2*idx+2]
        # assert set(map(lambda x: int(os.path.basename(x).split('_')[1]), files_idx)) == set([idx//2])
        
        if len(files_idx) < 2: print(idx, self.file_names[0]) # Couldn't find acc/gyr for that index
        data = np.stack([np.load(f) for f in files_idx], axis=1) # (T', 2, 3)
        if 'smooth2' in files_idx[0]: data = interp(data)
        
        data = data.reshape(data.shape[0], -1).T # (6, T')
        if data.shape[1] < T: data = np.pad(data, ((0, 0), (0, data.shape[0] - T)))
        elif data.shape[1] > T: data = np.stack([np.interp(np.linspace(0,3,T,False),
                                                           np.linspace(0,3,data.shape[1],False),
                                                           d) for d in data], 0) # (6, T)
        if not self.with_gyro: data = data[:3] # (3, T)
        if not self.with_acc: data = data[3:] # (3, T)
        
        if self.augment: data = self._augment_data(data)
        
        if self.classifier:
            basename = os.path.basename(files_idx[0])
            if self.is_synth:
                label = LBL[(basename.split('_')[0], basename.split('_')[2][0])]
                # length = data_synth_features[idx,3]
                # duration = data_synth_features[idx,2]
                # energy = data_synth_features[idx,2]
            elif self.is_stroke:
                label = LBL2[(basename.split('_')[0], basename.split('_')[2][0])]
                # length = data_stroke_features[idx,4]
                # duration = data_stroke_features[idx,4]
                # energy = data_stroke_features[idx,4]
            else:
                label = LBL2[(basename.split('_')[0], basename.split('_')[2][0])]
                # length = data_real_features[idx,6]
                # duration = data_real_features[idx,5]
                # energy = data_real_features[idx,8]
            return {
                'data': torch.FloatTensor(data), 'label': torch.FloatTensor([label]),
                # 'domain': torch.FloatTensor([self.is_synth]), 'length': torch.FloatTensor([length]), 'duration': torch.FloatTensor([duration]), 'energy': torch.FloatTensor([energy]), 'idx':torch.FloatTensor([idx]),
            }
        
        else:
            label = data_real_features[idx,4:6]
            return {'data': torch.FloatTensor(data), 'label': torch.FloatTensor(label), 'idx':torch.FloatTensor([idx])}
    
    def split_TrVaTe(self, per_subj=False, sizes=[0.8, 0.1, 0.1], seed=1):
        rng = np.random.default_rng(seed)
        
        if per_subj:
            if self.is_stroke: perm = rng.permutation(np.arange(len(SUBJIDXstroke)-1))
            else: perm = rng.permutation(np.arange(len(SUBJIDX)-1))
            
            sizes = (np.cumsum(sizes[:-1]) * len(perm)).astype(int)
            subsets_subjs = [np.sort(s) for s in np.split(perm, sizes)]
            
            if np.any(sizes == 0): print(f'Trying to split the data into {list(sizes)} subjects')
            print(*subsets_subjs)
            self.subsets_subjs = subsets_subjs
            
            if self.is_stroke: subsets_idx = [np.concatenate([np.arange(2*SUBJIDXstroke[sss], 2*SUBJIDXstroke[sss+1]) for sss in subset_subjs]) for subset_subjs in subsets_subjs]
            else: subsets_idx = [np.concatenate([np.arange(2*SUBJIDX[sss], 2*SUBJIDX[sss+1]) for sss in subset_subjs]) for subset_subjs in subsets_subjs]
            
            return tuple([Subset(self, subset_idx) for subset_idx in subsets_idx])
        
        else:
            return random_split(self, sizes)
    
    def _augment_data(self, data):
        if np.random.rand() > 0.5: data = self._jittering(data)
        if np.random.rand() > 0.5: data = self._scaling(data)
        if np.random.rand() > 0.5: data = self._time_warping(data)
        if np.random.rand() > 0.5: data = self._flipping(data)
        return data

    def _jittering(self, data, sigma=0.5):
        noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
        return data + noise

    def _scaling(self, data, sigma=0.3):
        factor = np.random.normal(loc=1.0, scale=sigma, size=(len(data),1))
        return data * factor

    def _time_warping(self, data, sigma=0.2):
        scale = np.random.uniform(1 - sigma, 1)
        def warp(x, s=1): return np.interp(np.arange(len(x)) * s, np.arange(len(x)), x)
        return np.apply_along_axis(warp, 1, data, s=scale)

    def _flipping(self, data):
        return np.flip(data, axis=1).copy()



class FbetaLoss(nn.Module):
    def __init__(self, beta=1, ratio=1, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.ratio = ratio
    
    def forward(self, y_score, y_true):
        tp = (y_score * y_true).sum()
        fp = (y_score * (1 - y_true)).sum()
        fn = ((1 - y_score) * y_true).sum()
        return - (1 + self.beta ** 2) * tp / ((1 + self.beta ** 2) * tp + self.ratio * fp + self.beta ** 2 * fn)


import glob

label_names_capture24 = ['bicycling', 'mixed', 'sit-stand', 'sleep', 'vehicle', 'walking']
label_numbers_capture24 = list(range(len(label_names_capture24)))
names2numbers_capture24 = dict([p for p in zip(label_names_capture24, label_numbers_capture24)])
numbers2names_capture24 = dict([p for p in zip(label_numbers_capture24, label_names_capture24)])

# Define class for capture24
class CAPTURE24(Dataset):
    def __init__(self, file_names, label_type='Willetts2018'):
        
        assert label_type in ['Willetts2018', 'Walmsley2020'], 'Incorrect label type!!!'
        labels = [os.path.basename(f).split('_')[0] for f in file_names]
        
        self.file_names = np.array(file_names)
        self.labels = np.array([names2numbers_capture24[t] for t in labels])
        self.subjects = np.char.lstrip(np.array(np.char.split(self.file_names, '/').tolist())[:,7], 'P').astype(int)
        self.targ_dict = numbers2names_capture24
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        data = np.load(self.file_names[index]).T
        subject = self.subjects[index]
        label = self.labels[index]
        
        if int(subject - 1) not in [10,11,15,19,28,38,41,50,58,60,63,65,66,67,68,77,81,94,96,97,
                                    101,107,113,119,122,129,131,138,139,142,144,149]:
            data = np.einsum('ij,jt->it', np.array([[-1,0,0],[0,-1,0],[0,0,1]]), data)
        
        return {'data': torch.FloatTensor(data), 'label': torch.LongTensor([label])}
    

# Define class for balanced capture24
class BalancedCAPTURE24(CAPTURE24):
    def __init__(self, file_names, label_type='Willetts2018', seed=12345):
        super().__init__(file_names, label_type)
        self.seed = seed
    
    def balance(self):
        minLabel = np.min(np.bincount(self.labels))
        if minLabel == 0: assert False, f'Dataset missing some class {np.bincount(self.labels)}'
        
        rng = np.random.default_rng(self.seed)
        shuffle = rng.permutation(np.arange(len(self.labels)))
        
        file_names = []
        labels = []
        labelCount = dict([(l,0) for l in label_numbers_capture24])
        for f, l in zip([self.file_names[s] for s in shuffle], [self.labels[s] for s in shuffle]):
            
            if labelCount[l] == minLabel: continue
            else:
                file_names += [f]
                labels += [l]
                labelCount[l] += 1
            if (np.array(list(labelCount.values())) == minLabel).all(): break
        
        assert (np.array(list(labelCount.values())) == minLabel).all(), f'Dataset not balanced {labelCount}'
        
        self.file_names = np.array(file_names)
        self.labels = np.array(labels)


        
label_names_wisdm = ['catch', 'chips', 'clapping', 'dribbling', 'drinking', 'folding',
                     'jogging', 'kicking', 'pasta', 'sandwich', 'sitting', 'soup',
                     'stairs', 'standing', 'teeth', 'typing', 'walking', 'writing']
label_numbers_wisdm = list(range(len(label_names_wisdm)))
names2numbers_wisdm = dict([p for p in zip(label_names_wisdm, label_numbers_wisdm)])
numbers2names_wisdm = dict([p for p in zip(label_numbers_wisdm, label_names_wisdm)])
# Define class for balanced WISDM
class BalancedWISDM(Dataset):
    def __init__(self, split, seed=12345, normalized=True):
        
        self.seed = seed
        
        P = np.load('/work/pi_sunghoonlee_umass_edu/Ignacio/wisdm/pid.npy').repeat(2)
        X = np.load('/work/pi_sunghoonlee_umass_edu/Ignacio/wisdm/X.npy').reshape(-1,150,3)
        Y = np.load('/work/pi_sunghoonlee_umass_edu/Ignacio/wisdm/Y.npy').repeat(2)
        
        idx = np.isin(P, split)
        
        P = P[idx]
        X = X[idx]
        Y = Y[idx]
        
        self.data = X if normalized else 9.81 * X
        self.subjects = P
        self.labels = np.array([names2numbers_wisdm[t] for t in Y])
        self.targ_dict = numbers2names_wisdm
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        data = self.data[index].T
        subject = self.subjects[index]
        label = self.labels[index]
        
        if int(subject - 1600) in [1,7,14,26,33,36,40,48]:
            data = np.einsum('ij,jt->it', np.array([[-1,0,0],[0,-1,0],[0,0,1]]), data)
            
        return {'data': torch.FloatTensor(data), 'label': torch.LongTensor([label])}
    
    def balance(self):
        minLabel = np.min(np.bincount(self.labels))
        if minLabel == 0: assert False, f'Dataset missing some class {np.bincount(self.labels)}'
        
        rng = np.random.default_rng(self.seed)
        shuffle = rng.permutation(np.arange(len(self.labels)))
        
        data = []
        subjects = []
        labels = []
        labelCount = dict([(l,0) for l in label_numbers_wisdm])
        for d, s, l in [(self.data[s], self.subjects[s], self.labels[s]) for s in shuffle]:
            
            if labelCount[l] == minLabel: continue
            else:
                data += [d]
                subjects += [s]
                labels += [l]
                labelCount[l] += 1
            if (np.array(list(labelCount.values())) == minLabel).all(): break
        
        assert (np.array(list(labelCount.values())) == minLabel).all(), f'Dataset not balanced {labelCount}'
        
        self.data = np.array(data)
        self.subjects = np.array(subjects)
        self.labels = np.array(labels)


label_names_realworld = ['climbingdown', 'climbingup', 'jumping', 'lying',
                         'running', 'sitting', 'standing', 'walking']
label_numbers_realworld = list(range(len(label_names_realworld)))
names2numbers_realworld = dict([p for p in zip(label_names_realworld, label_numbers_realworld)])
numbers2names_realworld = dict([p for p in zip(label_numbers_realworld, label_names_realworld)])
# Define class for balanced REALWORLD
class BalancedREALWORLD(Dataset):
    def __init__(self, split, seed=12345):
        
        self.seed = seed
        
        P = np.load('/work/pi_sunghoonlee_umass_edu/Ignacio/realworld/pid.npy').repeat(2).astype(int)
        X = np.load('/work/pi_sunghoonlee_umass_edu/Ignacio/realworld/X.npy').reshape(-1,150,3)
        Y = np.load('/work/pi_sunghoonlee_umass_edu/Ignacio/realworld/Y.npy').repeat(2)
        
        idx = np.isin(P, split)
        
        P = P[idx]
        X = X[idx]
        Y = Y[idx]
        
        self.data = X
        self.labels = np.array([names2numbers_realworld[t] for t in Y])
        self.targ_dict = numbers2names_realworld
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        data = self.data[index].T
        label = self.labels[index]
        return {'data': torch.FloatTensor(data), 'label': torch.LongTensor([label])}
    
    def balance(self):
        minLabel = np.min(np.bincount(self.labels))
        if minLabel == 0: assert False, f'Dataset missing some class {np.bincount(self.labels)}'
        
        rng = np.random.default_rng(self.seed)
        shuffle = rng.permutation(np.arange(len(self.labels)))
        
        data = []
        labels = []
        labelCount = dict([(l,0) for l in label_numbers_realworld])
        for d, l in zip([self.data[s] for s in shuffle], [self.labels[s] for s in shuffle]):
            
            if labelCount[l] == minLabel: continue
            else:
                data += [d]
                labels += [l]
                labelCount[l] += 1
            if (np.array(list(labelCount.values())) == minLabel).all(): break
        
        assert (np.array(list(labelCount.values())) == minLabel).all(), f'Dataset not balanced {labelCount}'
        
        self.data = np.array(data)
        self.labels = np.array(labels)