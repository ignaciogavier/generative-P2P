import gc
import numpy as np
from types import SimpleNamespace
from scipy.spatial.transform import Rotation as R
from scipy.signal import filtfilt, sosfilt, butter, resample, detrend
from scipy.integrate import cumulative_trapezoid as cumtrapz
from tqdm import tqdm

from sklearn.decomposition import PCA
from scripts.ahrs_application import AHRSApplication
from ahrs.filters import Tilt
from scripts.preprocess.inertial import activity_index


def fuse_and_rotate(data_obj, ahrs_filter_type):
    # Extract data
    sample_rate = data_obj.sample_rate
    local_acc = data_obj.local.acc.copy()
    local_gyr = data_obj.local.ang_vel.copy() # deg/s
    
    # Conversion
    # ahrs filter requires rad/s
    local_gyr = np.deg2rad(local_gyr)
    
    # Calculate orientation
    ahrs_filter = AHRSApplication(filter_type=ahrs_filter_type, gyr=local_gyr, acc=local_acc, frequency=sample_rate)
    orient = ahrs_filter.get_Q()
    
    # Convert quaternion to rotation matrix
    orient = R.from_quat(orient).as_matrix()
    if ahrs_filter_type == 'aqua':
        orient = np.linalg.inv(orient)
    
    # Save data
    data_obj.local.orient = orient
    
    del sample_rate, local_acc, local_gyr
    del ahrs_filter, orient
    gc.collect() 
    

def rotation_wrist(wrist_obj, trunk_obj, ahrs_filter_type='angularrate'):
    # load wrist data
    local_time = wrist_obj.time.copy()
    local_acc = wrist_obj.local.acc.copy()
    local_gyr = wrist_obj.local.ang_vel.copy()
    local_orient = wrist_obj.local.orient.copy()
    if len(local_orient.shape) == 2:
        local_orient = R.from_quat(local_orient).as_matrix()
        
    # rotation R(wrist -> G)
    earth_acc = R.from_matrix(local_orient).apply(local_acc)
    earth_gyr = R.from_matrix(local_orient).apply(local_gyr)

    # remove gravity
    if ahrs_filter_type == 'angularrate':
        gravity_vector = local_acc[:50].mean(axis=0, keepdims=True)
    else:
        # gravity_vector = np.array([[0, 0, 9.8]])
        gravity_vector = np.mean(earth_acc, axis=0, keepdims=True)
        # earth_acc = detrend(earth_acc, axis=0)
    earth_acc -= gravity_vector

    if trunk_obj is not None:
        # Load trunk data
        trunk_time = trunk_obj.time.copy()
        trunk_acc = trunk_obj.local.acc.copy()
        trunk_orient = trunk_obj.local.orient.copy()
        
        if ahrs_filter_type == 'angularrate':
            # rotation R(G1 -> mutual_global)
            orient = Tilt(acc = gravity_vector).Q[:, [1, 2, 3, 0]] 
            earth_acc = R.from_quat(orient).apply(earth_acc)
            earth_gyr = R.from_quat(orient).apply(earth_gyr)
            del orient
            gc.collect()
            
            # rotation R(mutual_global -> G2)
            orient = Tilt(acc = trunk_acc[:50].mean(axis=0, keepdims=True)).Q[:, [1, 2, 3, 0]]
            earth_acc = R.from_quat(orient).apply(earth_acc, inverse=True)
            earth_gyr = R.from_quat(orient).apply(earth_gyr, inverse=True)
            del orient
            gc.collect()

        # Compare trunk_time and wrist_time to trim the data
        if trunk_time[0] < local_time[0]:
            idx = np.argmin(np.abs(local_time[0] - trunk_time))
            trunk_time = trunk_time[idx:]
            trunk_orient = trunk_orient[idx:]
        else:
            idx = np.argmin(np.abs(local_time - trunk_time[0]))
            local_time = local_time[idx:]
            local_acc = local_acc[idx:]
            local_gyr = local_gyr[idx:]
            local_orient = local_orient[idx:]

            earth_acc = earth_acc[idx:]
            earth_gyr = earth_gyr[idx:]

            # Save local data
            wrist_obj.time = wrist_obj.time[idx:]
            wrist_obj.local.raw_acc = wrist_obj.local.raw_acc[idx:]
            wrist_obj.local.acc = wrist_obj.local.acc[idx:]
            wrist_obj.local.gyro = wrist_obj.local.gyro[idx:]
            wrist_obj.local.ang_vel = wrist_obj.local.ang_vel[idx:]
            wrist_obj.local.orient = wrist_obj.local.orient[idx:]

        # rotation R(G2 -> trunk)
        trunk_orient = trunk_orient[:len(local_orient)]
        assert len(trunk_orient) == len(local_orient)
        earth_acc = R.from_matrix(trunk_orient).apply(earth_acc, inverse=True)
        earth_gyr = R.from_matrix(trunk_orient).apply(earth_gyr, inverse=True)
        del trunk_time, trunk_orient

    # Save earth data
    wrist_obj.earth = SimpleNamespace()
    wrist_obj.earth.raw_acc = earth_acc
    wrist_obj.earth.gyro = earth_gyr

    del local_time, local_acc, local_gyr, local_orient
    del earth_acc, earth_gyr
    

def bandpass_filter_gravity(wrist_obj, order, fc_lp, fc_hp):
    sample_freq = wrist_obj.sample_rate
    local_acc = wrist_obj.local.acc.copy()
    local_gyr = wrist_obj.local.gyro.copy()
    
    # Define filter
    b, a = butter(order, [fc_lp/(sample_freq/2), fc_hp/(sample_freq/2)], btype='bandpass')
    earth_acc = filtfilt(b, a, local_acc, axis=0)
    
    # Save
    wrist_obj.earth = SimpleNamespace()
    wrist_obj.earth.raw_acc = earth_acc
    # wrist_obj.earth.gyro = local_gyr
    
    
def trim_stroke_data_by_time(data_obj, offset, start_time, end_time):
    # Adding offset to the time
    data_obj.time = data_obj.time - offset / 1000

    # Create exclusion mask
    exclusion_mask = (data_obj.time < start_time) | (data_obj.time > end_time)

    data_obj.time = data_obj.time[~exclusion_mask]
    data_obj.local.raw_acc = data_obj.local.raw_acc[~exclusion_mask]
    data_obj.local.acc = data_obj.local.acc[~exclusion_mask]
    data_obj.local.gyro = data_obj.local.gyro[~exclusion_mask]
    data_obj.local.ang_vel = data_obj.local.ang_vel[~exclusion_mask]
    data_obj.local.orient = data_obj.local.orient[~exclusion_mask]
    
    data_obj.earth.raw_acc = data_obj.earth.raw_acc[~exclusion_mask]
    if 'gyro' in dir(data_obj.earth):
        data_obj.earth.gyro = data_obj.earth.gyro[~exclusion_mask]


def filter_and_integrate(data_obj, filter_type='lowpass'):
    # Filtering data using 6th order butterworth
    # filter with cut-off freq of 10Hz
    if filter_type is None:
        data_obj.earth.acc = data_obj.earth.raw_acc
    else:
        sample_freq = data_obj.sample_rate
        dx = 1 / sample_freq
        fc_lp = 10
        fc_hp = 0.1
        b1, a1 = butter(6, fc_lp/(sample_freq/2), btype='lowpass')
        b2, a2 = butter(2, [fc_hp/(sample_freq/2), fc_lp/(sample_freq/2)], btype='bandpass')

        acc = data_obj.earth.raw_acc

        jerk = np.gradient(acc, dx, axis=0)
        jerk = filtfilt(b1, a1, jerk, axis=0)
        data_obj.earth.jerk = jerk

        acc = filtfilt(b1, a1, acc, axis=0)
        # acc = acc - acc.mean(axis=0)
        data_obj.earth.acc = acc

        vel = cumtrapz(acc, dx=dx, axis=0, initial=0)
        vel = filtfilt(b2, a2, vel, axis=0)
        # vel = vel - vel.mean(axis=0)
        data_obj.earth.vel = vel

        pos = cumtrapz(vel, dx=dx, axis=0, initial=0)
        pos = filtfilt(b2, a2, pos, axis=0)
        # pos = pos - pos.mean(axis=0)
        data_obj.earth.pos = pos


def detect_zero_crossings(v):
    zc = [0, v.shape[0]-1]
    for i in range(v.shape[0]-1):
        if (v[i] < 0 and v[i+1] >= 0) or (v[i] > 0 and v[i+1] <= 0):
            # zc.append(i)
            # because of low resampling rate, the end point of movement element is not
            # close to x-axis. here include different point for zero crossing
            zc.append(i) if (np.absolute(v[i]) <= np.absolute(v[i+1])) else zc.append(i+1) 
    zc = np.unique(zc)
    sI, sF = zc[:-1], zc[1:]
    return sI, sF


def decompose_all_mvmt_elem(data_obj, annot):
    elements = []
    for axis in range(data_obj.earth.vel.shape[1]):
        sI, sF = detect_zero_crossings(data_obj.earth.vel[:, axis])
        for i in range(1, sI.shape[0] - 1): # discard first and last MEs
            element = SimpleNamespace()
            start, end = sI[i], sF[i]
            
            if end - start < 3:
                continue
                
            if np.abs(data_obj.earth.pos[end - 1, axis] - data_obj.earth.pos[start, axis]) <= 0.001:
                continue

            element.sI = start
            element.sF = end
            element.axis = axis
            element.subject = data_obj.subject
            element.location = data_obj.location
            element.time = data_obj.time[start:end]
            element.dur = element.time[-1] - element.time[0]

            # earth frame
            element.jerk = data_obj.earth.jerk[start:end, axis]
            element.acc = data_obj.earth.acc[start:end, axis]
            element.vel = data_obj.earth.vel[start:end, axis]
            element.norm_vel = resample(element.vel / np.mean(element.vel), 100)
            element.disp = data_obj.earth.pos[start:end, axis]
            element.disp = element.disp[-1] - element.disp[0]
            
            element.start_idx = data_obj.start_idx + start
            element.end_idx = data_obj.start_idx + end
            
            # Annotation is either None or discard
            if annot is None:
                element.discard = False
            else:
                # If not overlapped with discard
                # then mask.all() is True
                # but discard should be False
                mask = (annot['End'] < element.time[0])
                mask |= (annot['Begin'] > element.time[-1])
                element.discard = ~mask.all()
                
            if 'p2p_id' in dir(data_obj):
                element.p2p_id = data_obj.p2p_id
            else:
                element.p2p_id = 0

            # Save
            elements.append(element)
    return np.array(elements, dtype=object)


def decompose_p2p_mvmt_elem(data_obj, annot):
    elements = []
    for axis in range(data_obj.earth.vel.shape[1]):
        sI, sF = detect_zero_crossings(data_obj.earth.vel[:, axis])
        for i in range(1, sI.shape[0] - 1): # discard first and last MEs
            element = SimpleNamespace()
            start, end = sI[i], sF[i]
            
            if end - start < 3:
                continue
                
            if np.abs(data_obj.earth.pos[end - 1, axis] - data_obj.earth.pos[start, axis]) <= 0.001:
                continue

            element.sI = start
            element.sF = end
            element.axis = axis
            element.subject = data_obj.subject
            element.location = data_obj.location
            element.time = data_obj.time[start:end]
            element.dur = element.time[-1] - element.time[0]

            # earth frame
            element.jerk = data_obj.earth.jerk[start:end, axis]
            element.acc = data_obj.earth.acc[start:end, axis]
            element.vel = data_obj.earth.vel[start:end, axis]
            element.norm_vel = resample(element.vel / np.mean(element.vel), 100)
            element.disp = data_obj.earth.pos[start:end, axis]
            element.disp = element.disp[-1] - element.disp[0]
            
            element.start_idx = data_obj.start_idx + start
            element.end_idx = data_obj.start_idx + end
            
            # Annotation is reach
            mask = (annot['End'] < element.time[0])
            mask |= (annot['Begin'] > element.time[-1])
            element.discard = mask.all()
            
            if element.discard:
                element.p2p_id = -1
            else:
                element.p2p_id = annot[~mask].index[0]
            
            # Save
            elements.append(element)
    return np.array(elements, dtype=object)


def decompose_non_p2p_mvmt_elem(data_obj, annot):
    elements = []
    for axis in range(data_obj.earth.vel.shape[1]):
        sI, sF = detect_zero_crossings(data_obj.earth.vel[:, axis])
        for i in range(1, sI.shape[0] - 1): # discard first and last MEs
            element = SimpleNamespace()
            start, end = sI[i], sF[i]
            
            if end - start < 3:
                continue
                
            if np.abs(data_obj.earth.pos[end - 1, axis] - data_obj.earth.pos[start, axis]) <= 0.001:
                continue

            element.sI = start
            element.sF = end
            element.axis = axis
            element.subject = data_obj.subject
            element.location = data_obj.location
            element.time = data_obj.time[start:end]
            element.dur = element.time[-1] - element.time[0]

            # earth frame
            element.jerk = data_obj.earth.jerk[start:end, axis]
            element.acc = data_obj.earth.acc[start:end, axis]
            element.vel = data_obj.earth.vel[start:end, axis]
            element.norm_vel = resample(element.vel / np.mean(element.vel), 100)
            element.disp = data_obj.earth.pos[start:end, axis]
            element.disp = element.disp[-1] - element.disp[0]
            
            element.start_idx = data_obj.start_idx + start
            element.end_idx = data_obj.start_idx + end
            
            # Annotation is reach
            mask = (annot['End'] < element.time[0])
            mask |= (annot['Begin'] > element.time[-1])
            element.discard = ~mask.all()
            
            element.p2p_id = -1
            
            # Save
            elements.append(element)
    return np.array(elements, dtype=object)


def decompose_p2p_wnd(data_obj, annot, delta, use_pca=True, n_components=1):
    sample_rate = data_obj.sample_rate
    dx = 1 / sample_rate
    
    elements = []
    for i, (_, begin, end, _) in annot.iterrows():
        # Adjust begin and end by delta
        begin -= delta
        end += delta

        # Extract P2P movements
        start_idx = np.argmin(np.abs(data_obj.time - begin))
        end_idx = np.argmin(np.abs(data_obj.time - end))
        
        # Segment data
        # if end_idx - start_idx < 21:
        #     continue
            
        p2p_obj = create_p2p_obj(data_obj, start_idx, end_idx)
        
        # Identify main direction
        if use_pca:
            p2p_obj.earth.vel = PCA(n_components=n_components, random_state=0).fit_transform(p2p_obj.earth.vel)

            # Save result
            p2p_obj.earth.acc = np.gradient(p2p_obj.earth.vel, dx, axis=0)
            p2p_obj.earth.jerk = np.gradient(p2p_obj.earth.acc, dx, axis=0)
            p2p_obj.earth.pos = cumtrapz(p2p_obj.earth.vel, dx=dx, axis=0)
            p2p_obj.p2p_id = i
            
        # Movement elements extraction
        elements.extend(decompose_all_mvmt_elem(p2p_obj, None)) 
        del begin, end, start_idx, end_idx
        del p2p_obj
    return elements


def create_stroke_window(data_obj, start_idx, end_idx, annot):
    idx = np.s_[start_idx:end_idx]
    
    window = SimpleNamespace()
    window.subject = data_obj.subject
    window.location = data_obj.location
    window.start_idx = start_idx
    window.end_idx = end_idx
    window.time = data_obj.time[idx]
    
    # Local frame
    window.local = SimpleNamespace()
    window.local.raw_acc = data_obj.local.raw_acc[idx]
    
    # Earth frame
    window.earth = SimpleNamespace()
    # window.earth.jerk = data_obj.earth.jerk[idx]
    window.earth.acc = data_obj.earth.acc[idx]
    # window.earth.vel = data_obj.earth.vel[idx]
    # window.earth.pos = data_obj.earth.pos[idx]
    
    mask = (annot['End'] < window.time[0])
    mask |= (annot['Begin'] > window.time[-1])
    if mask.all():
        window.activity_index = activity_index(window)
    else:
        window.activity_index = np.nan
    return window


def extract_windows(data_obj, annot, width_sec=1, stp_sz=1):
    # use annot to modify the activity_index of windows annotated as dicard
    if stp_sz <= 0 or stp_sz > 1:
        raise ValueError('Invalid stp_sz, must be > 0 or <= 1.')

    sample_rate = data_obj.sample_rate
    t_N = data_obj.time.shape[0]
    W = int(round(sample_rate * width_sec))
    step = int(np.round(W * stp_sz))
    t = 0
    
    windows = []
    while t + W < t_N:
        start_idx = t
        end_idx = t + W
        windows.append(create_stroke_window(data_obj, start_idx, end_idx, annot))
        t += step
    return np.array(windows, dtype=object)


def create_p2p_obj(data_obj, start_idx, end_idx):
    # create p2p_obj
    p2p_obj = SimpleNamespace()
    p2p_obj.subject = data_obj.subject
    p2p_obj.location = data_obj.location
    p2p_obj.sample_rate = data_obj.sample_rate
    p2p_obj.time = data_obj.time[start_idx:end_idx+1].copy()
    p2p_obj.start_idx = start_idx
    p2p_obj.end_idx = end_idx
    
    p2p_obj.earth = SimpleNamespace()
    p2p_obj.earth.jerk = data_obj.earth.jerk[start_idx:end_idx+1].copy()
    p2p_obj.earth.acc = data_obj.earth.acc[start_idx:end_idx+1].copy()
    p2p_obj.earth.vel = data_obj.earth.vel[start_idx:end_idx+1].copy()
    p2p_obj.earth.pos = data_obj.earth.pos[start_idx:end_idx+1].copy()
    return p2p_obj
        
    
def create_pca_obj(data_obj, start_idx, end_idx):
    '''
    start_idx and end_idx are the start and end
    index of P2P movements
    '''
    # # create p2p_obj
    # p2p_obj = SimpleNamespace()
    # 
    # p2p_obj.subject = data_obj.subject
    # p2p_obj.location = data_obj.location
    # p2p_obj.sample_rate = data_obj.sample_rate
    # p2p_obj.time = data_obj.time[start_idx:end_idx+1].copy()
    # 
    # p2p_obj.earth = SimpleNamespace()
    # p2p_obj.earth.acc = data_obj.earth.acc[start_idx:end_idx+1]
    # p2p_obj.earth.vel = data_obj.earth.vel[start_idx:end_idx+1]
    # p2p_obj.earth.pos = data_obj.earth.pos[start_idx:end_idx+1]
    
    sample_freq = 50
    dx = 1 / sample_freq
    fc_lp = 10
    fc_hp = 0.1
    b1, a1 = butter(6, fc_lp/(sample_freq/2), btype='lowpass')
    b2, a2 = butter(2, [fc_hp/(sample_freq/2), fc_lp/(sample_freq/2)], btype='bandpass')
    
    # Create p2p_obj
    p2p_obj = SimpleNamespace()
    p2p_obj.subject = data_obj.subject
    p2p_obj.location = data_obj.location
    p2p_obj.sample_rate = data_obj.sample_rate
    p2p_obj.time = data_obj.time[start_idx:end_idx+1].copy()
    
    p2p_obj.earth = SimpleNamespace()
    
    # Filter acceleration
    acc = data_obj.earth.raw_acc[start_idx:end_idx+1].copy()
    acc = filtfilt(b1, a1, acc, axis=0)
    acc = acc - acc.mean(axis=0)
    p2p_obj.earth.acc = acc
    
    # Calculate velocity
    vel = cumtrapz(acc, dx=dx, axis=0)
    vel = filtfilt(b2, a2, vel, axis=0)
    vel = vel - vel.mean(axis=0)
    p2p_obj.earth.vel = vel
    
    return p2p_obj