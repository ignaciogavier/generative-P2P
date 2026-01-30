import os
from types import SimpleNamespace
import numpy as np
from scipy.integrate import trapz
from scipy.signal import butter, sosfiltfilt, resample
from scipy.stats import mode, linregress
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation


def extract_movement_elements(obj, me_width=50, rotate=None):
    sI = []
    sF = []
    axis = []
    vel = []
    norm_vel = []
    acc = []
    jerk = []
    disp = []
    dur = []
    
    earth_vel = obj.earth_frame.vel
    earth_acc = obj.earth_frame.acc
    earth_jerk = obj.earth_frame.jerk
    
    if rotate is not None:
        rot = Rotation.from_euler('z', rotate, degrees=True)
        earth_vel = rot.apply(earth_vel)
        earth_acc = rot.apply(earth_acc)
        earth_jerk = rot.apply(earth_jerk)
    
    for ax in (0, 1, 2):
        zc = zero_crossings(earth_vel[:, ax])
        zc = [0, *zc, earth_vel.shape[0]-1]
 
        for start, end in zip(zc[:-1], zc[1:]):
            if end - start < 2:
                continue
                
            sI.append(start)
            sF.append(end)
            axis.append(ax)
            
            v = earth_vel[start:end, ax]
            a = earth_acc[start:end, ax]
            j = earth_jerk[start:end, ax]
            ts = obj.time[start:end]
            
            vel.append(v)
            norm_vel.append(resample(v / v.mean(), me_width))
            acc.append(a)
            jerk.append(j)
            disp.append(trapz(v, ts))
            dur.append(ts[-1] - ts[0])
    
    obj.me_start = np.array(sI)
    obj.me_end = np.array(sF)
    obj.me_axis = np.array(axis)
    obj.me_vel = jagged_list_to_padded_matrix(vel)
    obj.me_nvel = np.vstack(norm_vel)
    obj.me_acc = jagged_list_to_padded_matrix(acc)
    obj.me_jerk = jagged_list_to_padded_matrix(jerk)
    obj.me_disp = np.array(disp)
    obj.me_dur = np.array(dur)


def zero_crossings(x):
    s = np.sign(x)
    return np.where(s[:-1] != s[1:])[0] + 1


def jagged_list_to_padded_matrix(arrs, fill_value=np.nan):
    lens = np.array([len(a) for a in arrs])
    mask = lens[:, None] > np.arange(lens.max())
    X = np.full(mask.shape, fill_value)
    X[mask] = np.concatenate(arrs)
    return X


def vstack_jagged_matrixes(Xs, fill_value=np.nan):
    lens = np.array([m.shape[1] for m in Xs])
    pad = lens.max() - lens
    Xs = [np.pad(m, [(0, 0), (0, p)], mode='constant', constant_values=fill_value) for p, m in zip(pad, Xs)]
    return np.concatenate(Xs, axis=0)


def filter_mvmt_elems(obj, duration=0.055, displacement=0.001):
    mask = (obj.me_dur >= duration) & (np.abs(obj.me_disp) >= displacement)
    obj.me_start = obj.me_start[mask]
    obj.me_end = obj.me_end[mask]
    obj.me_axis = obj.me_axis[mask]
    obj.me_vel = obj.me_vel[mask]
    obj.me_nvel = obj.me_nvel[mask]
    obj.me_acc = obj.me_acc[mask]
    obj.me_jerk = obj.me_jerk[mask]
    obj.me_disp = obj.me_disp[mask]
    obj.me_dur = obj.me_dur[mask]


def cluster_mvmt_elems(norm_vel, eps, k=5, return_metrics=False):
    clusterer = DBSCAN(eps=eps, min_samples=k, metric='euclidean', n_jobs=-1).fit(norm_vel)
    
    main_cluster = mode(clusterer.labels_[clusterer.labels_ >= 0])[0][0]
    cluster_mask = clusterer.labels_ == main_cluster
    
    if return_metrics:
        cluster_size = cluster_mask.sum()
        cluster_std = norm_vel[cluster_mask].var(axis=0).mean()
        return clusterer, cluster_mask, cluster_size, cluster_std
    else:
        return clusterer, cluster_mask
    

class SavedCluster:
    def __init__(self, file):
        data = np.load(file)
        norm_vel = data['norm_vel']
        k = data['homog_k']
        eps = data['homog_eps']
        
        self.fit_cluster(norm_vel, eps, k)
        self.fit_pca(norm_vel)
    
    def fit_cluster(self, norm_vel, eps, k):
        self.model = BallTree(norm_vel)
        self.eps = eps
        self.k = k
    
    def apply_cluster(self, norm_vel):
        dist, _ = self.model.query(norm_vel, k=self.k, return_distance=True)
        return dist[:, -1] <= self.eps
    
    def fit_pca(self, norm_vel):
        self.pca = PCA().fit(norm_vel)
    
    def apply_pca(self, norm_vel):
        return self.pca.transform(norm_vel)


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    See https://stackoverflow.com/questions/23073170/calculate-bounding-polygon-of-alpha-shape-from-the-delaunay-triangulation
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def plot_bullet(ax, norm_vel, title=None):
    m_ = norm_vel.mean(axis=0)
    s_ = norm_vel.std(axis=0)
    ts = np.linspace(0, 100, m_.shape[0])
    ax.fill_between(ts, m_ - s_, m_ + s_, linewidth=0)
    ax.plot(ts, m_, color='k', linewidth=1)
    ax.set_xlabel('Duration (%)')
    ax.set_ylabel('Norm. Vel.')
    ax.set_ylim([-.1, 2])
    if title is not None:
        ax.set_title(title)


def plot_23_law(ax, log_dist, log_mean_speed, title=None, vmax=6e1):
    ax.hist2d(log_dist, log_mean_speed, bins=100, cmap='viridis', range=((-3.1, 1), (-3.1, 1)), vmax=vmax)
    ax.set_xticks(np.arange(5)-3)
    ax.set_yticks(np.arange(5)-3)
    ax.set_xlim([-3.1, 1])
    ax.set_ylim([-3.1, 1])
    ax.set_xlabel('log(|d|)')
    ax.set_ylabel('log(mean(|v|))')
    slope, intercept, _, _, _ = linregress(log_dist, log_mean_speed)
    if title is not None:
        ax.set_title(f'{title} $\\alpha={slope:.2f}$')
    else:
        ax.set_title(f'$\\alpha={slope:.2f}$')
    ax.plot(np.linspace(-3.1, 1, num=100), np.linspace(-3.1, 1, num=100) * slope + intercept, color='w', linewidth=.5)
