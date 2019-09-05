import numpy as np
import cv2
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from openpose_kinect_integrate import *
from read_real_time_skeletons import *

keep_3D = True
X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_fall_aug.npy')
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_fall_aug.npy')

joints_OP, joints_K, connection = joints_to_extract()
transposed_X_train = alter_kinect_dataset(joints_K, X_train, keep_3D=keep_3D)
transposed_X_test = alter_kinect_dataset(joints_K, X_test, keep_3D=keep_3D)

print(transposed_X_train.shape)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_fall_aug_int.npy',transposed_X_train)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_fall_aug_int.npy',transposed_X_test)
