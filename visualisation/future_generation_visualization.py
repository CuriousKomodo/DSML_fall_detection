
import os
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from openpose_kinect_integrate import *
from read_real_time_skeletons import *
from itertools import groupby
import collections
import matplotlib.gridspec as gridspec
joints_OP, joints_K, connection = joints_to_extract()

X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_aug.npy')
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed25.npy')
y_test_r = np.repeat(y_test,4)
y_test = np.concatenate([y_test, y_test_r])
y_test_bool = np.asarray([1 if x==43 else 0 for x in y_test])

result_path = '/Users/kefei/Documents/results/win60/future/VAE/non_residual/'
print(X_test.shape)
X_fall = X_test[y_test_bool==1]
print(X_fall.shape)
recon_fall = np.load(result_path + 'recon_fall.npy')
X_non_fall = X_test[y_test_bool==0]
recon_non_fall = np.load(result_path + 'recon_non_fall.npy')
TS_non_fall =np.mean(np.square(recon_non_fall - X_non_fall[:100,15:,:]), axis=(0,-1))
TS_fall =np.mean(np.square(recon_fall - X_fall[:100,15:,:]), axis=(0,-1))

print(TS_fall)

plt.plot(TS_fall)
plt.plot(TS_non_fall)
plt.legend(['Fall','Non-fall'])
plt.title('Average prediction error over time frame')
plt.xlabel('frame')
plt.ylabel('MSE')
plt.show()

'''
example = 80
sk_list = []
fall=False
if fall==True:
    X_fall = X_test[y_test_bool==1]
    del X_test
    recon_fall = np.load(result_path + 'recon_fall.npy')
    for example in [17, 91, 35 ,42, 31, 43, 44 ,74, 83, 84, 26]:
    #for i in range(5):
        sk_list.append(X_fall[example])
        recon_one = np.concatenate([X_fall[example,:15,:],recon_fall[example]], axis=0)
        sk_list.append(recon_one)
        #example+=1



else: #[82,89]
    X_non_fall = X_test[y_test_bool==0]
    print(X_non_fall.shape)
    del X_test
    #for i in range(10):
    for example in [60,61,62,63,64,65,66,82,68,89]:
        print('action=',y_test[y_test_bool==0][example])
        sk_list.append(X_non_fall[example])
        recon_non_fall = np.load(result_path + 'recon_non_fall.npy')
        recon_one = np.concatenate([X_non_fall[example,:15,:],recon_non_fall[example]], axis=0)
        sk_list.append(recon_one)
        #example+=1

if 2-1==1:
    #Plot the sequences
    fig, ax = plt.subplots(20,20,figsize=(40, 40))
    fig.patch.set_visible(False)

    poses_list = [x.reshape(x.shape[0],3,17) for x in sk_list]
    #poses_list = [x.reshape(x.shape[0],3,25) for x in sk_list]
    colors_list = ['b','b','b','b','b','b','b','b','b','b']
    #title_list = ['ground truth','a','b','c','d']
    #connection = [[0, 1], [1, 2], [2, 3],[8,9],[4,5],[9,10],[5,6],[16,17],[12,13],[17,18],[13,14],
                    #[14,15],[18,19],[10,11],[6,7],[11,23],[7,21],[2,20],[20,8],[20,4],[0,12],[0,16]]

    for pose in range(20):
        pose_frames = poses_list[pose][0:30:1,:,:]
        pose_frames[:,1,:] = pose_frames[:,1,:] *-1.0
        #print(pose_frames)
        #plt.title(title_list[pose])
        for i in np.arange(20):
            if (pose%2==1) & (i>=5):
                color = 'r'
            else:
                color = 'b'

            pose_fr = pose_frames[i+10]
            x = pose_fr[0, :]
            y = pose_fr[1, :]
            sc = ax[pose,i].scatter(x, y, s=5)


            for bone in connection:
              ax[pose,i].plot([pose_fr[0,bone[0]], pose_fr[0,bone[1]]], [pose_fr[1,bone[0]], pose_fr[1,bone[1]]],
              color)

            ax[pose,i].axis('off')
            ax[pose,i].spines['top'].set_visible(False)
            ax[pose,i].spines['right'].set_visible(False)
            ax[pose,i].spines['bottom'].set_visible(False)
            ax[pose,i].spines['left'].set_visible(False)

        if (pose>1)&(pose%2==1):
            gs0 = gridspec.GridSpec(2, 1)
            gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0], hspace=0)
            ax1 = fig.add_subplot(gs00[1], sharex=ax0)

    plt.show()
