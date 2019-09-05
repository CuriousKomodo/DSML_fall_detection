from pathlib import Path
import numpy as np
import os
import cv2
import numpy as np
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from openpose_kinect_integrate import *

def reset_realtime_skeleton(file_path):
    #if not file_path:
        #file_path='/Users/kefei/Documents/real_time_skeletons/joints.npy'
    zeros = np.zeros((1,25,3))
    np.save(file_path, zeros)
    return None

#this reads the skeletons extracted by Openpose
def read_realtime_skeleton(file_path):
    #if not file_path:
        #file_path='/Users/kefei/Documents/real_time_skeletons/joints.npy'
    p = Path(file_path)
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f))) #(1,25,3)
    return out

#Expect poses generated in the form of t x (3 x 25).
#truth = np.load('/Users/kefei/Documents/Dataset/one_pose.npy')
def obtain_realtime_integrated_joints(keep_3D=True,file_path=None):
    truth = read_realtime_skeleton(file_path)
    joints_OP, joints_K, connection = joints_to_extract()
    transposed_op_pose, confidence = transpose_openpose(joints_OP, truth, keep_3D=keep_3D)
    pose = reshape_openpose(transposed_op_pose, keep_3D=keep_3D)
    return pose, confidence

#print(pose.shape)
