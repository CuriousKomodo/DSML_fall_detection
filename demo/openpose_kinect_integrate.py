from pathlib import Path
import numpy as np
import os
import cv2
import numpy as np

#bodypart: [openpose_joint,kinect_joint]
def joints_to_extract():
    joint_dict = {'Head':[0,3], 'Neck':[1,20], 'Lshoulder':[2,8],'Rshoulder':[5,4],
     'LElbow':[3,9], 'RElbow':[6,5], 'LWrist':[4,10],'RWrist':[7,6],
    'Spinebase':[8,0], 'LHip':[9,16],'RHip':[12,12],
    'LKnee':[10,17], 'RKnee':[13,13],'LAnkle':[11,18],'RAnkle':[14,14],'LFoot':[22,19],'RFoot':[19,15]}

    #we aim for an integrated joint structure with the following indices:
    integrate_dict = {'Head':0, 'Neck':1, 'Lshoulder':2, 'Rshoulder':3,
     'LElbow':4, 'RElbow':5, 'LWrist':6, 'RWrist':7,
    'Spinebase':8, 'LHip':9, 'RHip':10,
    'LKnee':11,'RKnee':12, 'LAnkle':13,'RAnkle':14,'LFoot':15,'RFoot':16}

    integrated_connections = [[0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[1,8],[8,9],[8,10],[9,11],[10,12],[11,13],[12,14],[13,15],[14,16]]

    #joints to collect
    arr = list(joint_dict.values())
    joints_OP = [x[0] for x in arr]
    joints_K = [x[1] for x in arr]
    return joints_OP, joints_K, integrated_connections
#extract the openpose joints, then convert to the kinect system
#17 common joints.

#we reshape all sequences, we want to read from 75 instead of (3,25)
def alter_kinect_dataset(joints_K, kinect_poses, keep_3D=True):
    if keep_3D:
        dims = 3
    else:
        dims = 2

    kinect_poses = np.reshape(kinect_poses, (kinect_poses.shape[0], kinect_poses.shape[1], 3, 25))
    print('finish reshape')
    transposed = np.zeros((kinect_poses.shape[0],kinect_poses.shape[1],dims,len(joints_K)))
    for i in range(len(joints_K)):
        print('i=',i)
        K_ind = joints_K[i]
        transposed[:,:,:dims,i] = kinect_poses[:,:,:dims,K_ind]
    transposed[:,:,1,:] = transposed[:,:,1,:]*-1.0 #the y axis needs to be inverted!
    transposed = transposed.reshape(kinect_poses.shape[0],kinect_poses.shape[1],dims*len(joints_K))
    return transposed

#we reshape one sequence.
def transpose_openpose(joints_OP, op_pose, keep_3D=True, add_noise=False):
    if keep_3D:
        dim=3
    else:
        dim=2

    if add_noise:
        transposed = np.random.rand(op_pose.shape[0],len(joints_OP),dim)#random matrix generated over N(0,1)
    else:
        transposed = np.zeros((op_pose.shape[0],len(joints_OP),dim))

    confidence = np.zeros((op_pose.shape[0],len(joints_OP)))

    for i in range(len(joints_OP)):
        op_ind = joints_OP[i]
        transposed[:,i,:2] = op_pose[:,op_ind,:2] #Obtains (x,y) coordinates.
        confidence[:,i] =   op_pose[:,op_ind,-1] #extract the confidence scores
    return transposed, confidence #returns an array of (T, 25, 3), an an array of (T,)

#to make the openpose poses the same orientation as kinect...
def reshape_openpose(transposed_op_pose, keep_3D=True):
    if keep_3D:
        dims = 3
    else:
        dims = 2
    print(transposed_op_pose.shape)
    transposed_op_pose = np.transpose(transposed_op_pose,(0,2,1)) #reshape to (T,dims,17)
    joints = transposed_op_pose.shape[-1]
    reshaped_op_pose = np.reshape(transposed_op_pose,(transposed_op_pose.shape[0],dims*joints))
    return reshaped_op_pose

def normalize_openpose(pose):
    vmax = pose.max() #maximum observed in all past frames
    vmin = pose.min() #minimum observed in all past frames
    pose_n = (pose - vmin)/(vmax-vmin)
    return pose_n
#want to convert openpose to the form we want.
'''
// Result for BODY_25 (25 body parts consisting of COCO + foot)
// const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MidHip"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "LBigToe"},
//     {20, "LSmallToe"},
//     {21, "LHeel"},
//     {22, "RBigToe"},
//     {23, "RSmallToe"},
//     {24, "RHeel"},
//     {25, "Background"}
// };


kinectron.SPINEBASE = 0;
kinectron.SPINEMID = 1;
kinectron.NECK = 2;
kinectron.HEAD = 3;
kinectron.SHOULDERLEFT = 4;
kinectron.ELBOWLEFT = 5;
kinectron.WRISTLEFT = 6;
kinectron.HANDLEFT = 7;
kinectron.SHOULDERRIGHT = 8;
kinectron.ELBOWRIGHT = 9;
kinectron.WRISTRIGHT = 10;
kinectron.HANDRIGHT = 11;
kinectron.HIPLEFT = 12;
kinectron.KNEELEFT = 13;
kinectron.ANKLELEFT = 14;
kinectron.FOOTLEFT = 15;
kinectron.HIPRIGHT = 16;
kinectron.KNEERIGHT = 17;
kinectron.ANKLERIGHT = 18;
kinectron.FOOTRIGHT = 19;
kinectron.SPINESHOULDER = 20;
kinectron.HANDTIPLEFT  = 21;
kinectron.THUMBLEFT = 22;
kinectron.HANDTIPRIGHT = 23;
kinectron.THUMBRIGHT = 24;

'''
