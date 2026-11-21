"""Phase II: Blender simulation of synthetic IMU windows from Phase I trajectories (needs the `bpy` Python module).

One call = one subject and one class (P2P or non-P2P), e.g.
  python phase2_simulate_movement.py --subject_target 15 --is_not_P2P 0 \
      --traj_file trajectories/local_H0p35/S15_p2p.npy --out_dir synthetic_dataset/local_H0p35 --count 10000 --seed 777
phase1b_synthesize_trajectories.py writes the full task list (tasks.txt) with the seeds used in the paper
(seed = 777 + task index). Output: <out_dir>/S{n}_{W}s{FS}Hz/{p|n}{left|right}_<k>_{l|r}{acc,gyr}.npy, 3 s at 50 Hz.
--fixed_sensor 1 = ablation C (no random sensor roll, fixed avatar scale).
"""
import bpy
import numpy as np
from scipy.signal import butter, filtfilt
from mathutils import *
import math
import glob
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_utils import blender

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--subject_target', type=int, required=True)
parser.add_argument('--is_not_P2P', type=int, default=0)
parser.add_argument('--sampling_freq', type=int, default=50)
parser.add_argument('--window_size', type=int, default=3)
parser.add_argument('--traj_file', type=str, required=True, help='skeleton-format trajectories [log10 T, hand, 18 keyframe coords]')
parser.add_argument('--out_dir', type=str, required=True, help='dataset root; files go to <out_dir>/S{n}_{W}s{FS}Hz/')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--count', type=int, default=-1)
parser.add_argument('--fixed_sensor', type=int, default=0)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()
if args.seed is not None: np.random.seed(args.seed)

subjectTarget = args.subject_target
isnotP2P = args.is_not_P2P
FS = args.sampling_freq
WLEN = args.window_size
WSZ = FS * WLEN
deltaT = 1. / FS
timeInterp = np.arange(WSZ, dtype=float) / FS
b, a = butter(3, 8 * 2 * deltaT, 'lowpass')

synth_trajectories = np.load(args.traj_file)
stop = len(synth_trajectories) if args.count < 0 else min(len(synth_trajectories), args.start + args.count)
synthDir = args.out_dir
os.makedirs(f'{synthDir}/S{subjectTarget}_{WLEN}s{FS}Hz/', exist_ok=True)
print(f'S{subjectTarget} isnotP2P={isnotP2P} traj={args.traj_file} [{args.start},{stop}) fixed_sensor={args.fixed_sensor} -> {synthDir}', flush=True)

for n in range(args.start, stop):
    hand, pos_rot = blender(synth_trajectories[n], FS=FS, WLEN=WLEN, pos_rot=True, fixed_scale=bool(args.fixed_sensor))
    pos_rot = pos_rot.reshape(-1,8,3).dot(np.array([[-1,0,0],[0,-1,0],[0,0,1]])).reshape(-1,24)
    
    positionLeft = pos_rot[:,0:3]
    vXLeft, vYLeft, vZLeft = pos_rot[:,3:6], pos_rot[:,6:9], pos_rot[:,9:12]
    positionRight = pos_rot[:,12:15]
    vXRight, vYRight, vZRight = pos_rot[:,15:18], pos_rot[:,18:21], pos_rot[:,21:24]
    
    # hand, pos = blender(synth_trajectories[n], pos_rot=False)
    # pos_rot = pos_rot.reshape(-1,6,3).dot(np.array([[-1,0,0],[0,-1,0],[0,0,1]])).reshape(-1,18)
    
    # positionLeft = pos[:,0:3]
    # vXLeft = pos[:,0:3] - pos[:,3:6]
    # vZLeft = np.cross(pos[:,6:9] - pos[:,3:6], vXLeft, 1, 1)
    # vYLeft = np.cross(vZLeft, vXLeft, 1, 1)
    # positionRight = pos[:,9:12]
    # vXRight = pos[:,9:12] - pos[:,12:15]
    # vZRight = np.cross(pos[:,12:15] - pos[:,15:18], vXRight, 1, 1)
    # vYRight = np.cross(vZLeft, vXRight, 1, 1)
    
    
    # Compute global velocity and acceleration
    positionLeft = filtfilt(b, a, positionLeft, axis=0)
    positionRight = filtfilt(b, a, positionRight, axis=0)
    velocityLeft = filtfilt(b, a, np.gradient(positionLeft, timeInterp, axis=0, edge_order=1), axis=0)
    velocityRight = filtfilt(b, a, np.gradient(positionRight, timeInterp, axis=0, edge_order=1), axis=0)
    accelerationLeft = filtfilt(b, a, np.gradient(velocityLeft, timeInterp, axis=0, edge_order=1), axis=0)
    accelerationRight = filtfilt(b, a, np.gradient(velocityRight, timeInterp, axis=0, edge_order=1), axis=0)
    
    # Compute local velocity and acceleration
    orientationLeft = np.stack([vXLeft, vYLeft, vZLeft], 1)
    t = 0.0 if args.fixed_sensor else np.random.randn() * np.pi / 4
    orientationLeft = np.einsum('nij,jk->nik', orientationLeft,
                                np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]]))
    orientationLeft = filtfilt(b, a, orientationLeft, axis=0)
    orientationDerLeft = filtfilt(b, a, np.gradient(orientationLeft, deltaT, edge_order=1, axis=0), axis=0)
    gyroLeft = np.matmul(orientationLeft.transpose(0,2,1), orientationDerLeft)
    gyroLeft = np.stack([(gyroLeft[:,2,1] - gyroLeft[:,1,2]) / 2,
                         (gyroLeft[:,0,2] - gyroLeft[:,2,0]) / 2,
                         (gyroLeft[:,1,0] - gyroLeft[:,0,1]) / 2], axis=1)
    
    orientationRight = np.stack([vXRight, vYRight, vZRight], 1)
    t = 0.0 if args.fixed_sensor else np.random.randn() * np.pi / 4
    orientationRight = np.einsum('nij,jk->nik', orientationRight,
                                 np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]]))
    orientationRight = filtfilt(b, a, orientationRight, axis=0)
    orientationDerRight = filtfilt(b, a, np.gradient(orientationRight, deltaT, edge_order=1, axis=0), axis=0)
    gyroRight = np.matmul(orientationRight.transpose(0,2,1), orientationDerRight)
    gyroRight = np.stack([(gyroRight[:,2,1] - gyroRight[:,1,2]) / 2,
                          (gyroRight[:,0,2] - gyroRight[:,2,0]) / 2,
                          (gyroRight[:,1,0] - gyroRight[:,0,1]) / 2], axis=1)
    
    velocityLeftLoc = np.matmul(orientationLeft, velocityLeft[...,None]).squeeze(2)
    velocityRightLoc = np.matmul(orientationRight, velocityRight[...,None]).squeeze(2)
    accelerationLeftLoc = np.matmul(orientationLeft, (accelerationLeft + np.array([[0, 9.81, 0]]))[...,None]).squeeze(2)
    accelerationRightLoc = np.matmul(orientationRight, (accelerationRight + np.array([[0, 9.81, 0]]))[...,None]).squeeze(2)
    
    np.save(f'{synthDir}/S{subjectTarget}_{WLEN}s{FS}Hz/' +
            f'{"n" if isnotP2P else "p"}{hand}_{n+isnotP2P*10000}_lacc.npy', accelerationLeftLoc)
    np.save(f'{synthDir}/S{subjectTarget}_{WLEN}s{FS}Hz/' +
            f'{"n" if isnotP2P else "p"}{hand}_{n+isnotP2P*10000}_racc.npy', accelerationRightLoc)
    np.save(f'{synthDir}/S{subjectTarget}_{WLEN}s{FS}Hz/' +
            f'{"n" if isnotP2P else "p"}{hand}_{n+isnotP2P*10000}_lgyr.npy', gyroLeft)
    np.save(f'{synthDir}/S{subjectTarget}_{WLEN}s{FS}Hz/' +
            f'{"n" if isnotP2P else "p"}{hand}_{n+isnotP2P*10000}_rgyr.npy', gyroRight)