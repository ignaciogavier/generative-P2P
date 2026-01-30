import numpy as np
from types import SimpleNamespace


SYSTEMIC_VARIANCE = 2e-3


def activity_index(window, relative=True):
    '''See Bai, et. al, "An Activity Index for Raw Accelerometry Data"'''
    ai = np.var(window.local.raw_acc, ddof=1, axis=0).sum() - SYSTEMIC_VARIANCE
    if relative:
        ai = ai / SYSTEMIC_VARIANCE
    return np.sqrt(np.maximum(ai / 3, 0))


def preprocess_window(window):
    window.activity_index = activity_index(window)
    return window
