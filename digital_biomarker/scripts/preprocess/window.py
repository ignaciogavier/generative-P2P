import numpy as np
from types import SimpleNamespace
from enum import IntEnum

TAXONOMY = ['NULL', 'IDLE', 'STABILIZE', 'REACH', 'TRANSPORT', 'REPOSITION']


def _create_shimmer_window(data, start_idx, end_idx, idx=None):
    if idx is None:
        idx = np.s_[start_idx:end_idx]
    else:
        start_idx, end_idx = idx[0], idx[-1]
    
    window = SimpleNamespace()
    window.subject = data.subject
    window.location = data.location
    window.start_idx = start_idx
    window.end_idx = end_idx
    window.time = data.time[idx]

    # Local frame
    window.local = SimpleNamespace()
    window.local.raw_acc = data.local.raw_acc[idx]
    window.local.acc = data.local.acc[idx]
    window.local.vel = data.local.vel[idx]
    # window.local.disp = data.local.disp[idx]
    window.local.gyro = data.local.gyro[idx]
    window.local.ang_vel = data.local.ang_vel[idx]
    # window.local.quat = data.local.quat[idx]

    # # Earth frame
    # window.earth = SimpleNamespace()
    # window.earth.raw_acc = data.earth.raw_acc[idx]
    # window.earth.acc = data.earth.acc[idx]
    # window.earth.vel = data.earth.vel[idx]
    # # window.earth.disp = data.earth.disp[idx]
    # # window.earth.gyro = data.earth.gyro[idx]
    # window.earth.ang_vel = data.earth.ang_vel[idx]

    # # Local frame gravity free
    # window.local_gravity_free = SimpleNamespace()
    # window.local_gravity_free.acc = data.local_gravity_free.acc[idx]
    # window.local_gravity_free.vel = data.local_gravity_free.vel[idx]
    # window.local_gravity_free.disp = data.local_gravity_free.disp[idx]

    # # EMG
    # window.EMG = data.EMG.norm_EMG[idx]

    return window


def _create_formsense_window(data, start_idx, end_idx, idx=None):
    if idx is None:
        idx = np.s_[start_idx:end_idx]
    else:
        start_idx, end_idx = idx[0], idx[-1]

    window = SimpleNamespace()
    window.subject = data.subject
    window.location = data.location
    window.start_idx = start_idx
    window.end_idx = end_idx
    window.time = data.time[idx]
    window.acc_magnitude = data.acc_magnitude[idx]
    return window


def _create_mocap_window(data, start_idx, end_idx, idx=None):
    if idx is None:
        idx = np.s_[start_idx:end_idx]
    else:
        start_idx, end_idx = idx[0], idx[-1]
    
    window = SimpleNamespace()
    window.subject = data.subject
    window.location = data.location
    window.start_idx = start_idx
    window.end_idx = end_idx
    window.time = data.time[idx]
    window.pos = data.pos[idx]
    window.raw_pos = data.raw_pos[idx]
    window.vel = data.vel[idx]
    window.raw_vel = data.raw_vel[idx]
    window.acc = data.acc[idx]
    window.raw_acc = data.raw_acc[idx]
    return window


def _create_stroke_window(data, start_idx, end_idx, idx=None):
    if idx is None:
        idx = np.s_[start_idx:end_idx]
    else:
        start_idx, end_idx = idx[0], idx[-1]
    
    window = SimpleNamespace()
    window.subject = data.subject
    window.location = data.location
    window.start_idx = start_idx
    window.end_idx = end_idx
    window.time = data.time[idx]
    
    window.local = SimpleNamespace()
    window.local.raw_acc = data.local.raw_acc[idx]
    window.local.acc = data.local.acc[idx]
    # window.local.vel = data.local.vel[idx]
    window.local.gyro = data.local.gyro[idx]
    window.local.ang_vel = data.local.ang_vel[idx]
    # window.quat = data.quat[idx]

    return window


def _create_dose_trial_data_window(data, start_idx, end_idx, idx=None):
    if idx is None:
        idx = np.s_[start_idx:end_idx]
    else:
        start_idx, end_idx = idx[0], idx[-1]
    
    window = SimpleNamespace()
    window.subject = data.subject
    window.location = data.location
    window.start_idx = start_idx
    window.end_idx = end_idx
    window.acc = data.acc[idx]
    
    return window, np.std(np.linalg.norm(data.acc[idx], axis=1))


def extract_windows(data, sensor_type, width_sec=3, overlap=1/3):
    if overlap <= 0 or overlap > 1:
        raise ValueError('Invalid overlap, must be > 0 or <= 1.')

    sample_rate = data.sample_rate
    t_N = data.time.shape[0]
    W = int(round(sample_rate * width_sec))
    step = int(np.round(W * overlap))
    t = 0

    windows = []
    acc_mag_std = []
    while t + W < t_N:
        start_idx = t
        end_idx = t + W
        if sensor_type == 'shimmer':
            windows.append(_create_shimmer_window(data, start_idx, end_idx))
        elif sensor_type == 'formsense':
            windows.append(_create_formsense_window(data, start_idx, end_idx))
        elif sensor_type == 'mocap':
            windows.append(_create_mocap_window(data, start_idx, end_idx))
        elif sensor_type == 'stroke':
            windows.append(_create_stroke_window(data, start_idx, end_idx))
        elif sensor_type == 'dose_trial_data':
            output = _create_dose_trial_data_window(data, start_idx, end_idx)
            windows.append(output[0])
            acc_mag_std.append(output[1])
            del output
        else:
            raise ValueError('Wrong sensor_type')
        t += step
    if len(acc_mag_std) == 0:
        return np.array(windows, dtype=object)
    else:
        return np.array(windows, dtype=object), np.array(acc_mag_std)


def extract_annot_windows(data, annotation, sensor_type):
    '''
    For each annotation, extract the window corresponding to that
    '''
    time = data.time

    windows = []
    for _, annot in annotation.iterrows():
        tier, begin, end, label = annot
        if label in mvmt_type:
            start_idx = np.argmin(np.abs(time - begin))
            end_idx = np.argmin(np.abs(time - end))
            if sensor_type == 'shimmer':
                wnd = _create_shimmer_window(data, start_idx, end_idx)
            elif sensor_type == 'formsense':
                wnd = _create_formsense_window(data, start_idx, end_idx)
            elif sensor_type == 'mocap':
                wnd = _create_mocap_window(data, start_idx, end_idx)
            else:
                raise ValueError('Wrong sensor_type')
            wnd.label = ClassLabel.resolve(label)
            windows.append(wnd)
    return np.array(windows)


class ClassLabel(IntEnum):
    NULL = 0
    IDLE = 1
    STABILIZE = 2
    REACH = 3
    TRANSPORT = 4
    REPOSITION = 5
    
    @staticmethod
    def resolve(label_str):
        if label_str == 'idle':
            return ClassLabel.IDLE
        elif label_str == 'stabilize':
            return ClassLabel.STABILIZE
        elif label_str == 'reach':
            return ClassLabel.REACH
        elif label_str == 'transport':
            return ClassLabel.TRANSPORT
        elif label_str == 'reposition':
            return ClassLabel.REPOSITION
        # elif label_str == 'TBD' or label_str == 'other' or label_str == 'motor overflow':
        #     return ClassLabel.NULL
        else:
            raise ValueError('Unrecognized class label string: {}'.format(label_str))


def is_contained(annotations, window):
    return (annotations['Begin'] >= window.time[0]) & (annotations['End'] <= window.time[-1])


def calc_overlap_pct(annotations, window):
    begin = np.maximum(window.time[0], annotations['Begin'])
    end = np.minimum(window.time[-1], annotations['End'])
    # Overlap pct is negative if there is no overlap
    return (end-begin) / (window.time[-1]-window.time[0]) # calculate the window percentage


def find_annotations(annotations, window, thresh=1/3, return_overlap=True):
    if thresh < 0 or thresh > 1:
        raise ValueError('Invalid threshold, must be in 0-1.')
    
    overlap_pct = calc_overlap_pct(annotations, window)
    mask = (overlap_pct >= thresh) | is_contained(annotations, window)

    if return_overlap:
        return annotations[mask], overlap_pct[mask]
    else:
        return annotations[mask]


def discard_windows(annotations, windows):
    mask = np.ones(windows.shape[0], dtype=bool)
    for i, w in enumerate(windows):
        if 'discard' in find_annotations(annotations, w, thresh=0, return_overlap=False)['Label'].unique():
            mask[i] = False
        elif 'ambulatory' in find_annotations(annotations, w, thresh=0, return_overlap=False)['Label'].unique():
            mask[i] = False
    return windows[mask]


def resolve_competing_labels(labels, overlap_pct):
    # # Select the label wih the most overlap
    # sort_idx = np.argsort(overlap_pct)
    # return ClassLabel.resolve(labels[sort_idx][-1])

    # Merge the same movements
    uniq_labels = []
    uniq_overlap_pct = []
    for l in np.unique(labels):
        if l in ['idle', 'stabilize']:
            # if there are only idle and/or stabilize and no P2P movements
            # they sum of percentages has to be 100%
            # meaning that the whole window should only contain inactive movements
            # then we choose whichever is the longest
            if len(set(('reach', 'transport', 'reposition')).intersection(set(labels))) == 0:
                if np.sum(overlap_pct) == 1:
                    uniq_labels.append(l)
                    uniq_overlap_pct.append(np.sum(overlap_pct[labels == l]))
                else:
                    continue
            # if there are idle/stabilize and other P2P movements,
            # then we do not consider idle/stabilize at all
            # only choose the P2P movement that has the longest percentage
            else:
                continue
        else:
            # if there are only P2P movements
            # only choose the P2P movement that has the longest percentage
            uniq_labels.append(l)
            uniq_overlap_pct.append(np.sum(overlap_pct[labels == l]))

    uniq_labels = np.array(uniq_labels)
    uniq_overlap_pct = np.array(uniq_overlap_pct)
    
    # Select the label wih the most overlap
    if uniq_labels.shape[0] == 0:
        return ClassLabel.NULL
    else:
        sort_idx = np.argsort(uniq_overlap_pct)
        return ClassLabel.resolve(uniq_labels[sort_idx][-1])


def annotate_windows(annotations, windows, overlap_thresh=1/3):
    windows = discard_windows(annotations, windows)  # Discard any windows that overlap a discard label
    # windows = annotate_repetitive(annotations, windows)  # Set a flag for any windows that contain a repetitive movements label
    
    for w in windows:
        overlap_annot, overlap_pct = find_annotations(annotations, w, thresh=overlap_thresh)
        # Remove annotation if it is not in ['idle', 'stabilize', 'reach', 'transport', 'reposition']
        inclusion_mask = overlap_annot['Label'].isin(['idle', 'stabilize', 'reach', 'transport', 'reposition'])
        overlap_annot = overlap_annot[inclusion_mask]
        overlap_pct = overlap_pct[inclusion_mask]
        del inclusion_mask

        # If it is reach, transport, or reposition,
        # and it is not fully contained in the window
        # we need to remove those from annotation
        mask = ((overlap_annot['Label'].isin(['reach', 'transport', 'reposition'])) & (~is_contained(overlap_annot, w)))
        overlap_annot = overlap_annot[~mask]
        overlap_pct = overlap_pct[~mask]
        del mask

        uniq_labels = overlap_annot['Label'].unique()
        if uniq_labels.shape[0] == 0: # No valid annotations -> NULL class
            w.label = ClassLabel.NULL
        elif uniq_labels.shape[0] == 1:  # Single label type -> Use that label
            # if the label is idle or stabilize, the pct must be 100%
            if (uniq_labels[0] in ['idle', 'stabilize']):
                if overlap_pct.iloc[0] == 1:
                    w.label = ClassLabel.resolve(uniq_labels[0])
                else:
                    w.label = ClassLabel.NULL
            else:
                w.label = ClassLabel.resolve(uniq_labels[0])
        else:
            w.label = resolve_competing_labels(overlap_annot['Label'].to_numpy(), overlap_pct.to_numpy())
    return windows


def extract_and_annotate(sensor_data, annot, sensor_type, width_sec=3, overlap=1/3, overlap_thresh=1/3):
    windows = extract_windows(sensor_data, sensor_type, width_sec=width_sec, overlap=overlap)
    windows = annotate_windows(annot, windows, overlap_thresh=overlap_thresh)
    return windows
