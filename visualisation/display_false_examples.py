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
from itertools import groupby
import collections

joints_OP, joints_K, connection = joints_to_extract()
example = 70
action_dict = {'16':'put on a shoe','42':'staggering',
                '6':'pick up', '17':'take off a shoe', '48':'nausea/vomiting', '14':'put on jacket','0':'others'}
top_examples = ['16','42','6','17','48','14']
kl_type ='klanneal'
false = 'false_positives'
if false == 'false_positives':
    if kl_type =='klanneal':
        model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/with_classifier/'%kl_type
        result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type

        false_positives = np.load(result_path + 'false_positives.npy')
        false_positives_recon = np.load(result_path + 'false_positives_recon.npy')
        false_positives_label = np.load(result_path + 'false_positives_label.npy')
        print(false_positives.shape)
        sk_list=[]
        sk_list.append(false_positives[example])
        sk_list.append(false_positives_recon[example][::-1])
        print(false_positives_label[example])
else:
    kl_type ='klanneal'
    model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/with_classifier/'%kl_type
    result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
    false_negatives = np.load(result_path + 'false_negatives.npy')
    false_negatives_recon = np.load(result_path + 'false_negatives_recon.npy')
    sk_list=[]
    print(false_negatives.shape)
    for i in range(10):
        example +=1
        sk_list.append(false_negatives[-example])
        #sk_list.append(false_negatives_recon[example][::-1])

    '''
    a =  [str(x) for x in false_positives_label]
    a = [x if x in top_examples else '0' for x in a]
    print(a)
    a = [action_dict[x] for x in a]
    print(a)
    counts=collections.Counter(a)

    plt.pie([float(v) for v in counts.values()], labels=[str(k) for k in counts],
           autopct='%1.1f%%')
    #freq = [len(list(group)) for key, group in groupby(a)]
    plt.title('Number of false positive by action class')
    plt.show()
    '''

#Plot the sequences
fig, ax = plt.subplots(10,10,figsize=(25, 60))
fig.patch.set_visible(False)

poses_list = [x.reshape(x.shape[0],3,17) for x in sk_list]
#poses_list = [x.reshape(x.shape[0],3,25) for x in sk_list]
colors_list = ['b','b','b','b','b','b','b','b','b','b']
#title_list = ['ground truth','a','b','c','d']
#connection = [[0, 1], [1, 2], [2, 3],[8,9],[4,5],[9,10],[5,6],[16,17],[12,13],[17,18],[13,14],
                #[14,15],[18,19],[10,11],[6,7],[11,23],[7,21],[2,20],[20,8],[20,4],[0,12],[0,16]]

for pose in range(10):
    pose_frames = poses_list[pose][0:30:3,:,:]
    pose_frames[:,1,:] = pose_frames[:,1,:] *-1.0
    #print(pose_frames)
    #plt.title(title_list[pose])
    for i in range(10):

        pose_fr = pose_frames[i]
        x = pose_fr[0, :]
        y = pose_fr[1, :]
        sc = ax[pose,i].scatter(x, y, s=10)

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
