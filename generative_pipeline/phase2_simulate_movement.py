import bpy
import numpy as np
from scipy.signal import butter, filtfilt
from mathutils import *
import math
import glob
import os
import argparse
from blender import blender


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--subject_target', type=int, default=10, help='Subject target')
parser.add_argument('--is_not_P2P', type=int, default=0, help='Is not P2P')
parser.add_argument('--sampling_freq', type=int, default=30, help='Sampling frequency')
parser.add_argument('--window_size', type=int, default=5, help='Window size')
# parser.add_argument('--is_P2P', action='store_true', help='Is P2P') 
args = parser.parse_args()

subjectTarget = args.subject_target
isnotP2P = args.is_not_P2P
FS = args.sampling_freq
WLEN = args.window_size

WSZ = FS * WLEN
deltaT = 1. / FS
timeInterp = np.arange(WSZ, dtype=float) / FS

b, a = butter(3, 8 * 2 * deltaT, 'lowpass')

synth_trajectories = np.load(f'/home/igavier_umass_edu/Documents/synthetic-pipeline/mocap-synthtraj-feat/skeletons_from_mocapS{subjectTarget}_10000trajectory{f"_neg{isnotP2P}" if isnotP2P else ""}_v2.npy') # if using MoCap
# synth_trajectories = np.load(f'/home/igavier_umass_edu/Documents/synthetic-pipeline/mocap-synthtraj-feat/skeletons_from_GRAB_10000trajectory{f"_neg{isnotP2P}" if isnotP2P else ""}.npy') # if using GRAB
N = len(synth_trajectories) # np.random.randint(len(synth_trajectories)) # number of samples generated

synthDir = '/work/pi_sunghoonlee_umass_edu/Ignacio/P2P-classifier/synthetic_dataset'

# Create folder if it doesn't exist
if not os.path.exists(f'{synthDir}/S{subjectTarget}_{WLEN}s{FS}Hz/'):
    os.makedirs(f'{synthDir}/S{subjectTarget}_{WLEN}s{FS}Hz/', exist_ok=True)

for n in range(N):
    hand, pos_rot = blender(synth_trajectories[n], FS=FS, WLEN=WLEN, pos_rot=True)
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
    t = np.random.randn() * np.pi / 4
    orientationLeft = np.einsum('nij,jk->nik', orientationLeft,
                                np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]]))
    orientationLeft = filtfilt(b, a, orientationLeft, axis=0)
    orientationDerLeft = filtfilt(b, a, np.gradient(orientationLeft, deltaT, edge_order=1, axis=0), axis=0)
    gyroLeft = np.matmul(orientationLeft.transpose(0,2,1), orientationDerLeft)
    gyroLeft = np.stack([(gyroLeft[:,2,1] - gyroLeft[:,1,2]) / 2,
                         (gyroLeft[:,0,2] - gyroLeft[:,2,0]) / 2,
                         (gyroLeft[:,1,0] - gyroLeft[:,0,1]) / 2], axis=1)
    
    orientationRight = np.stack([vXRight, vYRight, vZRight], 1)
    t = np.random.randn() * np.pi / 4
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