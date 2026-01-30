import os
import re
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from collections import defaultdict
from types import SimpleNamespace
import xml.etree.ElementTree as ET

taxonomy = ['idle', 'stabilize', 'reach', 'reposition', 'transport']

SHIMMER_SENSOR_LOCATION = ['left_wrist', 'right_wrist']
STROKE_SENSOR_LOCATION = ['left_wrist', 'right_wrist', 'trunk']


def load_ELAN_annotations(path, vocabulary):
    # Load annotation
    annotations = pd.read_csv(
        path,
        delimiter='\t',
        # skiprows=2,
        comment='"',
        usecols=(0, 3, 5, 8),
        names=['Tier', 'Begin', 'End', 'Label'],
        index_col=False,
        header=None,
        engine='c'
    )
  
    for i in range(len(annotations['Label'])):
        label = annotations['Label'].iloc[i]
        label = label.split(';')[0]
        annotations['Label'].iloc[i] = label
        
        if label not in vocabulary:
            limb  = annotations['Tier'].iloc[i]
            begin = annotations['Begin'].iloc[i]
            end   = annotations['End'].iloc[i]
            print('%s, %.3f-%.3f, %s is not in the label dictionary' % (limb, begin, end, label))
        del label
    
    return annotations


def load_ELAN_offset(path):
    tree = ET.parse(path)
    root = tree.getroot()
    if 'time-origin' in root[0].keys():
        return int(root[0].get('time-origin'))
    else:
        return 0


def load_sensor_indices(sensor_indices_dir, kwargs={}):
    if sensor_indices_dir.endswith('xlsx'):
        sensor_indices_file = pd.read_excel(sensor_indices_dir, engine='openpyxl', **kwargs)
    elif sensor_indices_dir.endswith('csv'):
        sensor_indices_file = pd.read_csv(sensor_indices_dir, delimiter=',', **kwargs)
    sensor_indices = {}
    for i in range(len(sensor_indices_file)):
        subj = sensor_indices_file['subject'].iloc[i]    
        indices = dict(tuple(sensor_indices_file.iloc[i].items())[1:])
        sensor_indices[subj] = indices
        del subj, indices
    del sensor_indices_file
    return sensor_indices


def load_stroke_data(path):
    # Load data
    data = pd.read_csv(path)
    
    # Save data
    shimmer_data = SimpleNamespace()
    shimmer_data.time = data['time'].to_numpy()
    
    shimmer_data.local = SimpleNamespace()
    shimmer_data.local.raw_acc  = data[['acc_X', 'acc_Y', 'acc_Z']].to_numpy()
    shimmer_data.local.gyro = data[['gyro_X', 'gyro_Y', 'gyro_Z']].to_numpy()
    return shimmer_data


def combine_stroke_data(path, sample_rate=50, save_trunk=False):
    folders = path.split(os.path.sep)
    for i in range(len(folders)):
        if 'RingSensor' in folders[i]:
            break
    subject = re.match(r'RingSensor(\d+)[_]?[a-zA-Z]*', folders[i])[1]
    subject = int(subject)
    
    shimmer_data = SimpleNamespace()
    
    for filename in sorted(os.listdir(path)):
        # Only choose data from wrist and trunk
        if 'Wrist' not in filename and 'Trunk' not in filename:
            continue
        
        # Load data
        data = load_stroke_data(os.path.join(path, filename))
        data.subject = subject
        data.sample_rate = sample_rate
        
        # Save data
        if 'WristL' in filename:
            data.location = 'left_wrist'
            shimmer_data.left_wrist = data
        elif 'WristR' in filename:
            data.location = 'right_wrist'
            shimmer_data.right_wrist = data
        elif 'Trunk' in filename and save_trunk:
            data.location = 'trunk'
            shimmer_data.trunk = data
 
        del filename, data
        gc.collect()
    return shimmer_data


def load_stroke_demographics(path):
    data = pd.read_excel(path, engine='openpyxl')
    
    demographics = defaultdict(SimpleNamespace)
    for i, line in data.iterrows():
        subj_id = int(line['patientID'][1:-1])
        demographics[subj_id].age = line['Age']
        demographics[subj_id].sex = line['Sex']
        demographics[subj_id].chronicity = line['Chronicity']
        demographics[subj_id].dominant_side = line['Domant_side'][1:-1].lower()
        demographics[subj_id].impaired_side = line['Affected_side'][1:-1].lower()
        demographics[subj_id].FM = line['FMA']
        demographics[subj_id].MAL_AoU = line['MAL_AoU']
        demographics[subj_id].MAL_QoM = line['MAL_QoM']
        demographics[subj_id].WMFT_TR = line['WMFT_TR']
        demographics[subj_id].WMFT_FAS = line['WMFT_FAS']
        del i, subj_id, line
    return demographics