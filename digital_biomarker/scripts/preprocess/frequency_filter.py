import numpy as np
from scipy.fft import rfft, fftfreq
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from utils import HeeyonADL


def calc_rejection_mask(window_acc, adl_activity, subj, dx):
    X = _extract_freq_coefs(window_acc, dx)
    y = adl_activity == HeeyonADL.LOTION
    return _calc_mask_losocv(X, y, subj)
    

def _extract_freq_coefs(window_acc, dx, freq_cutoff=6):
    X_mag = np.linalg.norm(window_acc, axis=1)
    fft = rfft(X_mag, axis=1)
    freqs = np.abs(fftfreq(X_mag.shape[1], d=dx)[:fft.shape[1]])
    return np.abs(fft)[:, freqs <= 6]


def _calc_mask_losocv(X, y, subj, decision_threshold=0.5):
    scores = np.zeros(y.shape)
    for i, s in enumerate(np.unique(subj)):
        train_mask = subj != s
        test_mask = subj == s

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        
        params = {
            'C': np.logspace(-3, 3, 6*3+1),
            'max_iter': (1e6,),
            'class_weight': ('balanced',),
        }
        grid = GridSearchCV(LogisticRegression(), params, cv=5, scoring='f1', n_jobs=-1).fit(X_train, y_train)
        model = LogisticRegression(**grid.best_params_).fit(X_train, y_train)
        scores[test_mask] = model.predict_proba(X_test)[:, 1]
    return scores >= decision_threshold
