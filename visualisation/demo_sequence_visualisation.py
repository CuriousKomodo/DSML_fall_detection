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
file_path = '/Users/kefei/Documents/real_time_skeletons/joints24.npy'
skeleton,confidence = obtain_realtime_integrated_joints(keep_3D=False, file_path = file_path)
skeleton = skeleton[0:100]
sk_list = []
if skeleton.shape[0]>100:
    sk_list.append(skeleton[:100])
    sk_list.append(skeleton[100:])
else:
    sk_list.append(skeleton[:100])

#Plot the sequences
fig, ax = plt.subplots(1,16,figsize=(9, 1))
fig.patch.set_visible(False)
fall_interval = [[20,50],[220,250]]
poses_list = [x.reshape(x.shape[0],2,17) for x in sk_list]
#poses_list = [x.reshape(x.shape[0],3,25) for x in sk_list]
for pose in range(1):
    n = max(skeleton.shape[0],90)
    pose_frames = poses_list[pose][0:n:3,:,:]
    pose_frames[:,1,:] = pose_frames[:,1,:] *-1.0
    #print(pose_frames)
    #plt.title(title_list[pose])
    for i in range(16):
        if (i>(fall_interval[pose][0]/3.0)) & (i<(fall_interval[pose][1]/3.0)):
            color = 'r'
        else:
            color = 'b'
        pose_fr = pose_frames[i]
        x = pose_fr[0, :]
        y = pose_fr[1, :]

        #sc = ax[pose,i].scatter(x, y, s=40)
        sc = ax[i].scatter(x,y,s=40)
        for bone in connection:
            joint1 = bone[0]
            joint2 = bone[1]
            conf1 = confidence[i,joint1]
            conf2 = confidence[i,joint2]
            if (conf1 > 0.2) & (conf2 >0.2):
                #cv2.line(img, (int(pose[t,joint1,0]),int(pose[t,joint1,1])), (int(pose[t,joint2,0]),int(pose[t,joint2,1])),color, 1)
                ax[i].plot([pose_fr[0,bone[0]], pose_fr[0,bone[1]]], [pose_fr[1,bone[0]], pose_fr[1,bone[1]]],color)

        ax[i].axis('off')
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
plt.show()
