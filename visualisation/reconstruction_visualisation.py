#Generates the representations

import os
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from openpose_kinect_integrate import *
from read_real_time_skeletons import *

joints_OP, joints_K, connection = joints_to_extract()


X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_trimmed.npy')
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed.npy')
example = 1

fall = True
if fall:
    sk_list = []
    truth = X_test[y_test==43][:100]
    sk_list.append(truth[example])
    for kl_type in ['klanneal','klcyclic']:
        result_path1 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
        result_path2 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type
        recon1 = np.load(result_path1 + 'recon_fall.npy')
        recon2 = np.load(result_path2 + 'recon_fall.npy')
        recon1 = recon1[example][::-1]
        recon2 = recon2[example][::-1]

        sk_list.append(recon1)
        sk_list.append(recon2)

else:
    sk_list = []
    truth = X_test[y_test==27][:100]
    activity = y_test[y_test==27][:100]
    print('activity=',activity[example])
    sk_list.append(truth[example])
    for kl_type in ['klanneal','klcyclic']:
        result_path1 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
        result_path2 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type
        recon1 = np.load(result_path1 + 'recon_non_fall.npy')
        recon2 = np.load(result_path2 + 'recon_non_fall.npy')
        recon1 = recon1[example][::-1]
        recon2 = recon2[example][::-1]

        sk_list.append(recon1)
        sk_list.append(recon2)

result_path0 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/AE/with_classifier/'
recon0 = np.load(result_path0+'recon_fall.npy')
recon0 = recon0[example][::-1]
sk_list.append(recon0)
'''
truth = np.load('/Users/kefei/Documents/Dataset/one_pose.npy')
rotated = np.load('/Users/kefei/Documents/Dataset/one_rotated_pose.npy')
sk_list = [truth, rotated]
'''
#Plot the sequences
fig, ax = plt.subplots(6,10,figsize=(25, 10))
fig.patch.set_visible(False)


poses_list = [x.reshape(x.shape[0],3,17) for x in sk_list]
#poses_list = [x.reshape(x.shape[0],3,25) for x in sk_list]
colors_list = ['b','r','r','r','r','r']
#title_list = ['ground truth','a','b','c','d']
#connection = [[0, 1], [1, 2], [2, 3],[8,9],[4,5],[9,10],[5,6],[16,17],[12,13],[17,18],[13,14],
                #[14,15],[18,19],[10,11],[6,7],[11,23],[7,21],[2,20],[20,8],[20,4],[0,12],[0,16]]

for pose in range(6):

    pose_frames = poses_list[pose][0:30:3,:,:]
    pose_frames[:,1,:] = pose_frames[:,1,:] *-1.0
    #print(pose_frames)
    #plt.title(title_list[pose])
    for i in range(10):

        pose_fr = pose_frames[i]
        x = pose_fr[0, :]
        y = pose_fr[1, :]
        sc = ax[pose,i].scatter(x, y, s=40)

        for bone in connection:
          ax[pose,i].plot([pose_fr[0,bone[0]], pose_fr[0,bone[1]]], [pose_fr[1,bone[0]], pose_fr[1,bone[1]]],
          colors_list[pose])

        ax[pose,i].axis('off')
        ax[pose,i].spines['top'].set_visible(False)
        ax[pose,i].spines['right'].set_visible(False)
        ax[pose,i].spines['bottom'].set_visible(False)
        ax[pose,i].spines['left'].set_visible(False)
plt.show()

'''
example = 5
activity = labels_r[example]

recon_pose = recon[example].copy()
recon_pose = np.flip(recon_pose,axis=0)
recon_pose=np.reshape(recon_pose,(recon_pose.shape[0], 3, 25))

true_pose =truth[example].copy()
true_pose=np.reshape(true_pose,(true_pose.shape[0], 3, 25))

#bone_list = [[0, 1], [1, 2], [2, 3],[2, 4], [2,8],[8,9],[4,5],[9,10],[5,6],[16,17],[12,13],[17,18],[13,14]]

number_of_postures = len(np.arange(0,30,2))
print(number_of_postures)

#Plot the sequences
fig, ax = plt.subplots(2,number_of_postures,figsize=(45, 10))
poses_list = [true_pose, recon_pose1, recon_pose2, recon_pose3, recon_pose4]
colors_list = ['b','r']

for pose in range(2):
    pose_frames = poses_list[pose][0:30:2,:,:]
    for i in range(number_of_postures):
        plt.title('Skeleton')
        pose_fr = pose_frames[i]
        x = pose_fr[0, :]
        y = pose_fr[1, :]
        sc = ax[pose,i].scatter(x, y, s=40)

        for bone in bone_list:
          ax[pose,i].plot([pose_fr[0,bone[0]], pose_fr[0,bone[1]]], [pose_fr[1,bone[0]], pose_fr[1,bone[1]]], colors_list[pose])

#plt.title('Ground truth vs. reconstruction,30 frames with step size = 5, action = %s' %activity)
plt.show()
'''
