import numpy as np
from scipy.stats import iqr, skew, kurtosis, linregress
from scipy.stats import f as fcrv
from scipy.signal import find_peaks
from sklearn.decomposition import PCA


def extract_features(subj, homog, vel, norm_vel, acc, jerk, dur, disp, annotations, name_prefix='ALL'):
    n_peaks = calc_n_peaks(norm_vel)
    
    features = {}
    feature_names = []
    for s in np.unique(subj):
        smask = subj == s
        feats, names = [], []

        # pca = PCA(random_state=0).fit(norm_vel[~smask])  # Calculate PCA axes in LOSOCV manner
        pca = None
        
        # extract annot
        annot = annotations[s].copy()
        
        mask = smask
        f, n = extract_features_subset(
            vel[mask],
            norm_vel[mask],
            acc[mask],
            jerk[mask],
            dur[mask],
            disp[mask],
            n_peaks[mask],
            pca,
            annot,
            name_prefix=name_prefix
        )
        feats.extend(f)
        names.extend(n)
        
        if homog is not None:
            feats.append(homog[smask].mean())
            names.append('ALL>pct_homog')

            mask = smask & homog
            f, n = extract_features_subset(
                vel[mask],
                norm_vel[mask],
                acc[mask],
                jerk[mask],
                dur[mask],
                disp[mask],
                n_peaks[mask],
                pca,
                anont,
                name_prefix='HOMOG'
            )
            feats.extend(f)
            names.extend(n)

            mask = smask & ~homog
            f, n = extract_features_subset(
                vel[mask],
                norm_vel[mask],
                acc[mask],
                jerk[mask],
                dur[mask],
                disp[mask],
                n_peaks[mask],
                pca,
                annot,
                name_prefix='OUT'
            )
            feats.extend(f)
            names.extend(n)    
        
        features[s] = feats
        if len(feature_names) == 0:
            feature_names = names
    
    feature_names = np.array(feature_names)
    subjects = np.array([s for s in features])
    features = np.vstack([features[s] for s in features])
    return feature_names, subjects, features


# def extract_features_subset(vel, norm_vel, acc, jerk, dur, disp, n_peaks, pca, n_pcs=2, name_prefix=''):
#     features = []
#     names = []
    
#     log_dist = np.log10(np.abs(disp))
#     mean_speed = np.abs(np.nanmean(vel, axis=1))
#     log_mean_speed = np.log10(mean_speed)
#     peak_speed = np.nanmax(np.abs(vel), axis=1)
#     mean_abs_acc = np.nanmean(np.abs(acc), axis=1)
#     mean_abs_jerk = np.nanmean(np.abs(jerk), axis=1)
    
#     calc_aggregates(features, names, dur, f'{name_prefix}>duration')
#     calc_aggregates(features, names, np.abs(disp), f'{name_prefix}>distance') # 
#     calc_aggregates(features, names, log_dist, f'{name_prefix}>log_distance')
#     calc_aggregates(features, names, log_mean_speed, f'{name_prefix}>log_mean_speed')
#     calc_aggregates(features, names, mean_speed / peak_speed, f'{name_prefix}>speed_metric') # 
#     calc_aggregates(features, names, mean_abs_acc, f'{name_prefix}>mean_abs_acc') # 
#     calc_aggregates(features, names, mean_abs_jerk, f'{name_prefix}>mean_abs_jerk') # 
#     calc_aggregates(features, names, -mean_abs_jerk / peak_speed, f'{name_prefix}>jerk_metric')
#     calc_aggregates(features, names, calc_mapr(np.abs(vel)), f'{name_prefix}>mapr') # 
#     calc_aggregates(features, names, np.sqrt(np.mean(np.square(norm_vel), axis=1)), f'{name_prefix}>rms')
#     calc_aggregates(features, names, skew(norm_vel, axis=1), f'{name_prefix}>skewness') # 
#     calc_aggregates(features, names, n_peaks, f'{name_prefix}>n_peaks', percentiles=[])
#     calc_aggregates(features, names, norm_vel.std(axis=0), f'{name_prefix}>error_bar') # 
#     calc_aggregates(features, names, np.argmax(norm_vel, axis=1) / (norm_vel.shape[1] - 1), f'{name_prefix}>peak_pos') # 
#     # calc_aggregates(features, names, calc_pca_residual_mag(pca, n_pcs, norm_vel), f'{name_prefix}>pca_resid_magn') # 

#     # calc_hist(features, names, dur, np.linspace(0, 2, 21), f'{name_prefix}>duration', trim=(0, 1)) # 
#     # calc_hist(features, names, log_dist, np.linspace(-2, 1, 31), f'{name_prefix}>log_distance', trim=(-2, 0)) # 
#     # calc_hist(features, names, log_mean_speed, np.linspace(-2, 1, 31), f'{name_prefix}>log_mean_speed', trim=(-2, 0)) # 
    
#     # pcs = pca.transform(norm_vel)
#     # for i in range(n_pcs):
#     #     calc_aggregates(features, names, pcs[:, i], f'{name_prefix}>pc{i+1:d}')
    
#     features.append(dur.shape[0] / dur.sum())
#     names.append(f'{name_prefix}>me_per_sec')
    
#     alpha, *_ = linregress(log_dist, log_mean_speed)
#     features.append(alpha)
#     names.append(f'{name_prefix}>powerlaw_alpha')
    
#     features.append(norm_vel.shape[0])
#     names.append(f'{name_prefix}>#_of_mvmt_elems')
    
#     return features, names


def extract_features_subset(vel, norm_vel, acc, jerk, dur, disp, n_peaks, pca, annot, n_pcs=2, name_prefix=''):
    features = []
    names = []
    
    log_dist = np.log10(np.abs(disp))
    mean_speed = np.abs(np.nanmean(vel, axis=1))
    var_speed = np.abs(np.nanvar(vel, axis=1))
    std_speed = np.abs(np.nanstd(vel, axis=1))
    CV_speed = std_speed / mean_speed
    log_mean_speed = np.log10(mean_speed)
    peak_speed = np.nanmax(np.abs(vel), axis=1)
    mean_abs_acc = np.nanmean(np.abs(acc), axis=1)
    peak_acc = np.nanmax(np.abs(acc), axis=1)
    mean_abs_jerk = np.nanmean(np.abs(jerk), axis=1)
    peak_jerk = np.nanmax(np.abs(jerk), axis=1)
    
    # duration related features
    calc_aggregates(features, names, dur, f'{name_prefix}>duration')
    
    # distance related features
    calc_aggregates(features, names, np.abs(disp), f'{name_prefix}>distance')
    
    # speed related features
    calc_aggregates(features, names, mean_speed, f'{name_prefix}>mean_speed')
    calc_aggregates(features, names, std_speed, f'{name_prefix}>std_speed')
    calc_aggregates(features, names, CV_speed, f'{name_prefix}>CV_speed')
    calc_aggregates(features, names, peak_speed, f'{name_prefix}>peak_speed')
    calc_aggregates(features, names, mean_speed / peak_speed, f'{name_prefix}>speed_metric')
    # calc_aggregates(features, names, calc_mapr(np.abs(vel)), f'{name_prefix}>mapr')
    calc_aggregates(features, names, skew(vel, axis=1, nan_policy='omit'), f'{name_prefix}>speed_skewness')
    calc_aggregates(features, names, np.sqrt(np.nanmean(np.square(vel), axis=1)), f'{name_prefix}>speed_rms')
    calc_aggregates(features, names, n_peaks, f'{name_prefix}>n_peaks', percentiles=[])
    
    # # features related to normalized velocity
    # calc_aggregates(features, names, np.max(np.abs(norm_vel), axis=1), f'{name_prefix}>normvel_peak')
    # calc_aggregates(features, names, np.sqrt(np.mean(np.square(norm_vel), axis=1)), f'{name_prefix}>normvel_rms')
    # calc_aggregates(features, names, skew(norm_vel, axis=1), f'{name_prefix}>normvel_skewness')
    
    # acceleration related features
    calc_aggregates(features, names, mean_abs_acc, f'{name_prefix}>mean_abs_acc')
    calc_aggregates(features, names, peak_acc, f'{name_prefix}>peak_acc')
    
    # jerk related features
    calc_aggregates(features, names, mean_abs_jerk, f'{name_prefix}>mean_abs_jerk')
    calc_aggregates(features, names, peak_jerk, f'{name_prefix}>peak_jerk')
    calc_aggregates(features, names, -mean_abs_jerk / peak_speed, f'{name_prefix}>jerk_metric')
    
    # calc_aggregates(features, names, norm_vel.std(axis=0), f'{name_prefix}>error_bar') # 
    # calc_aggregates(features, names, np.argmax(norm_vel, axis=1) / (norm_vel.shape[1] - 1), f'{name_prefix}>peak_pos') # 
    # calc_aggregates(features, names, calc_pca_residual_mag(pca, n_pcs, norm_vel), f'{name_prefix}>pca_resid_magn') # 

    # calc_hist(features, names, dur, np.linspace(0, 2, 21), f'{name_prefix}>duration', trim=(0, 1)) # 
    # calc_hist(features, names, log_dist, np.linspace(-2, 1, 31), f'{name_prefix}>log_distance', trim=(-2, 0)) # 
    # calc_hist(features, names, log_mean_speed, np.linspace(-2, 1, 31), f'{name_prefix}>log_mean_speed', trim=(-2, 0)) # 
    
    # pcs = pca.transform(norm_vel)
    # for i in range(n_pcs):
    #     calc_aggregates(features, names, pcs[:, i], f'{name_prefix}>pc{i+1:d}')
    
    # features.append(dur.shape[0] / dur.sum())
    # names.append(f'{name_prefix}>me_per_sec')
    
    # alpha, *_ = linregress(log_dist, log_mean_speed)
    # features.append(alpha)
    # names.append(f'{name_prefix}>powerlaw_alpha')
    
    features.append(norm_vel.shape[0] / (annot['End'].iloc[-1] - annot['Begin'].iloc[0]))
    names.append(f'{name_prefix}>#_of_mvmt_elems')
    
    return features, names


def calc_aggregates(features, names, x, basename, percentiles=[10, 50, 90], use_skew=False, use_kurtosis=False, use_iqr=True):
    # percentiles=[10, 50, 90]
    
    features.append(x.mean())
    names.append(f'{basename}$mean')
    
    features.append(x.std(ddof=1))
    names.append(f'{basename}$std')
    
    if use_iqr:
        features.append(iqr(x))
        names.append(f'{basename}$iqr')
    
    for p in percentiles:
        features.append(np.percentile(x, p))
        names.append(f'{basename}${p:02d}pct')
    
    if use_skew:
        features.append(skew(x))
        names.append(f'{basename}$skew')
        
    if use_kurtosis:
        features.append(kurtosis(x))
        names.append(f'{basename}$kurtosis')
        

def calc_hist(features, names, x, bins, basename, trim=None):
    hist, _ = np.histogram(x, bins=bins, density=True)
    
    if trim is not None:
        start, end = trim
        mask = (bins[:-1] >= start) & (bins[1:] <= end)
        hist = hist[mask]
        
    for i, h in enumerate(hist):
        features.append(h)
        names.append(f'{basename}>$bin_{i}')


def calc_n_peaks(norm_vel):
    N, _ = norm_vel.shape
    n_peaks = np.zeros(N)
    for i in range(N):
        peaks, _ = find_peaks(norm_vel[i])
        n_peaks[i] = len(peaks)
    return n_peaks


def calc_pca_residual_mag(pca, n_pcs, norm_vel):
    X = pca.transform(norm_vel)
    X[:, n_pcs:] = 0  # Drop info / apply smoothing
    X = pca.inverse_transform(X)
    return np.linalg.norm(norm_vel - X, axis=1)


def calc_mapr(vel, thresh_pct=10):
    N, _ = vel.shape
    valid = ~np.isnan(vel)
    
    mapr = np.zeros(N)
    for i in range(N):
        v = vel[i, valid[i]]
        mapr[i] = (v > np.percentile(v, thresh_pct)).mean()
    return mapr


def icc(M, icc_type, alpha=0.05, r0=0):
    """
    Adapted from MATLAB version of the code:
    https://www.mathworks.com/matlabcentral/fileexchange/22099-intraclass-correlation-coefficient-icc
    A correction from the comments has been applied for icc_type='A-K'.
    
    icc_type updated to accept equivalent alternate notation.
    
    Values checked against the ones from R, here: https://stackoverflow.com/questions/40965579/intraclass-correlation-in-python-module
    There are some slight numerical differences, but values are otherwise the same for the first 3 or 4 significiant digits.
    A correction to dk2 for 'A-K' and 'A-1' were applied to match the numbers from R.
    """
    
    N, K = M.shape
    ss_total = M.var(ddof=1) * (N * K - 1)
    msr = M.mean(axis=1).var(ddof=1) * K
    msw = M.var(ddof=1, axis=1).sum() / N
    msc = M.mean(axis=0).var(ddof=1) * N
    mse  = (ss_total - msr * (N-1) - msc * (K-1)) / ((N-1) * (K-1))
    
    if icc_type in ('1-1', '(1, 1)'):
        r = (msr - msw) / (msr + (K-1) * msw)
        f = (msr / msw) * (1-r0) / (1 + (K-1) * r0)
        df1, df2 = N-1, N * (K-1)
        p = 1 - fcrv.cdf(f, df1, df2)
        fl = (msr / msw) / fcrv.ppf(1 - alpha/2, N-1, N * (K-1))
        fu = (msr / msw) * fcrv.ppf(1 - alpha/2, N * (K-1), N-1)
        lb = (fl - 1) / (fl + (K-1))
        ub = (fu - 1) / (fu + (K-1))
    elif icc_type in ('1-K', '(1, k)'):
        r = (msr - msw) / msr
        f = (msr / msw) * (1-r0)
        df1, df2 = N-1, N * (K-1)
        p = 1 - fcrv.cdf(f, df1, df2)
        fl = (msr / msw) / fcrv.ppf(1 - alpha/2, N-1, N * (K-1))
        fu = (msr / msw) * fcrv.ppf(1 - alpha/2, N * (K-1), N-1)
        lb = 1 - 1 / fl
        ub = 1 - 1 / fu
    elif icc_type in ('C-1', '(3, 1)'):
        r = (msr - mse) / (msr + (K-1) * mse)
        f = (msr / mse) * (1-r0) / (1 + (K-1) * r0)
        df1, df2 = N-1, (N-1) * (K-1)
        p = 1 - fcrv.cdf(f, df1, df2)
        fl = (msr / mse) / fcrv.ppf(1 - alpha/2, N-1, (N-1) * (K-1))
        fu = (msr / mse) * fcrv.ppf(1 - alpha/2, (N-1) * (K-1), N-1)
        lb = (fl - 1) / (fl + (K-1))
        ub = (fu - 1) / (fu + (K-1))
    elif icc_type in ('C-K', '(3, k)'):
        r = (msr - mse) / msr
        f = (msr / mse) * (1-r0)
        df1, df2 = N-1, (N-1) * (K-1)
        p = 1 - fcrv.cdf(f, df1, df2)
        fl = (msr / mse) / fcrv.ppf(1 - alpha/2, N-1, (N-1) * (K-1))
        fu = (msr / mse) * fcrv.ppf(1 - alpha/2, (N-1) * (K-1), N-1)
        lb = 1 - 1 / fl
        ub = 1 - 1 / fu
    elif icc_type in ('A-1', '(2, 1)'):
        r = (msr - mse) / (msr + (K-1)*mse + K*(msc-mse)/N)
        a = (K*r0) / (N*(1-r0))
        b = 1 + (K*r0*(N-1)) / (N*(1-r0))
        f = msr / (a*msc + b*mse)
        a = (K*r) / (N*(1-r))
        b = 1 + (K*r*(N-1)) / (N*(1-r))
        v = (a*msc + b*mse) ** 2 / ((a*msc)**2/(K-1) + (b*mse)**2/((N-1)*(K-1)))
        df1, df2 = N-1, (N-1) * (K-1)
        p = 1 - fcrv.cdf(f, df1, df2)
        fs = fcrv.ppf(1 - alpha/2, N-1, v)
        lb = N * (msr-fs*mse) / (fs * (K*msc + (K*N-K-N)*mse) + N*msr)
        fs = fcrv.ppf(1 - alpha/2, v, N-1)
        ub = N * (fs*msr-mse) / (K*msc + (K*N-K-N)*mse + N*fs*msr)
    elif icc_type in ('A-K', '(2, k)'):
        r = (msr - mse) / (msr + (msc-mse)/N)
        c = r0 / (N*(1-r0))
        d = 1 + (r0*(N-1)) / (N*(1-r0))
        f = msr / (c*msc + d*mse)
        a = r / (N*(1-r))
        b = 1 + (r*(N-1)) / (N*(1-r))
        v = (a*msc + b*mse) ** 2 / ((a*msc)**2/(K-1) + (b*mse)**2/((N-1)*(K-1)))
        df1, df2 = N-1, (N-1) * (K-1)
        p = 1 - fcrv.cdf(f, df1, df2)
        fs = fcrv.ppf(1 - alpha/2, N-1, v)
        lb = N * (msr-fs*mse) / (fs * (msc-mse) + N*msr)
        fs = fcrv.ppf(1 - alpha/2, v, N-1)
        ub = N * (fs*msr-mse) / (msc - mse + N*fs*msr)
    else:
        raise ValueError('Invalid ICC Type: {}'.format(icc_type))
        
    return r, lb, ub, f, df1, df2, p


def eliminate_features(X, rcutoff=0.75):
    N, R, D = X.shape
    mask = np.ones((D,), dtype=bool)
    for i in range(D):
        r, *_ = icc(X[:, :, i], '(3, 1)')
        if r < rcutoff:
            mask[i] = False
    return mask
