import numpy as np
import math
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from functions import *

#clean the dataset.
def trim_dataset(data,label,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    data = data[length>=threshold,:,:]
    label =label[length>=threshold]
    return data,label

X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_trimmed.npy')
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses//integrated/3D/X_test_60_2_trimmed.npy')

print(X_train.shape)
print(X_test.shape)
y_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_train_fall_aug_trimmed.npy')
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses//integrated/3D/y_test_60_2_trimmed.npy')
print('finish loading dataset')

X_train,y_train = trim_dataset(X_train,y_train,25)
X_test,y_test = trim_dataset(X_test,y_test,25)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_train_60_2_trimmed25.npy',np.array(y_train))
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed25.npy',np.array(y_test))
print(X_train.shape)
print(X_test.shape)

def flip_dataset(data,axis):
    data_aug = np.flip(data,axis=axis)
    return data_aug

def rotation_matrix(axis, theta):
    """
     Euler–Rodrigues formula
     Return the rotation matrix associated with counterclockwise rotation about
     the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_45(v, axis=(0,1,0), theta = np.pi/8.0):
    r = np.dot(rotation_matrix(axis, theta), v)
    return r

def rotate_90(v, axis=(0,1,0), theta = np.pi/6.0):
    r = np.dot(rotation_matrix(axis, theta), v)
    return r

def rotate_135(v, axis=(0,1,0), theta = np.pi/3.0):
    r = np.dot(rotation_matrix(axis, theta), v)
    return r

def rotate_180(v, axis=(0,1,0), theta = np.pi):
    r = np.dot(rotation_matrix(axis, theta), v)
    return r


def rotate_all(data):
    augumented_all_data = []
    for i in range(data.shape[0]):
        if i%10000 ==0:
            print('i=',i)

        rotated_sequences = []
        sequence = data[i,:,:]
        sequence = np.reshape(sequence,(sequence.shape[0], 3, 17))
        rotated_sequence_45 = np.apply_along_axis(rotate_45, 1, sequence)
        rotated_sequence_45 = rotated_sequence_45.reshape(rotated_sequence_45.shape[0],51)
        rotated_sequence_90 = np.apply_along_axis(rotate_90, 1, sequence)
        rotated_sequence_90 = rotated_sequence_90.reshape(rotated_sequence_90.shape[0],51)
        rotated_sequence_135 = np.apply_along_axis(rotate_135, 1, sequence)
        rotated_sequence_135 = rotated_sequence_135.reshape(rotated_sequence_135.shape[0],51)
        rotated_sequence_180 = np.apply_along_axis(rotate_180, 1, sequence)
        rotated_sequence_180 = rotated_sequence_135.reshape(rotated_sequence_180.shape[0],51)
        rotated_sequences.append(rotated_sequence_45)
        rotated_sequences.append(rotated_sequence_90)
        rotated_sequences.append(rotated_sequence_135)
        rotated_sequences.append(rotated_sequence_180)
        rotated_sequences = np.stack(rotated_sequences)
        augumented_all_data.append(rotated_sequences)

    augumented_all_data = np.concatenate(augumented_all_data) #contains data augumented at 3 different degress
    data = np.concatenate([data,augumented_all_data],axis=0) #concatenate
    return data

X_train_aug = rotate_all(X_train)
X_test_aug = rotate_all(X_test)

print(X_train_aug.shape)

#y_train_aug = np.repeat(y_train,2)
#y_train_aug = np.concatenate(np.array(y_train),np.array(y_train_aug), axis=0)
#y_test_aug = np.repeat(y_test,2)
#y_test_aug = np.concatenate(np.array(y_test),np.array(y_test_aug), axis=0)

#rint(np.array(y_train_aug).shape)

print('completed all augumentation')
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_aug.npy',X_train_aug)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_aug.npy',X_test_aug)

'''
axis_list = [0,2,(0,2)]
X_train_aug = np.zeros((3*X_train.shape[0],X_train.shape[1],X_train.shape[-1]))
X_test_aug = np.zeros((3*X_test.shape[0],X_test.shape[1],X_test.shape[-1]))

for i in range(len(axis_list)):
    axis = axis_list[i]
    print('flip axis=',axis)
    #axis = axis_list[i]
    ##Noo no no no no
    train_aug = np.flip(X_train, axis=axis)
    test_aug = np.flip(X_test, axis=axis)
    X_train_aug[i*X_train.shape[0]:(i+1)*X_train.shape[0],:,:] = train_aug
    X_test_aug[i*X_test.shape[0]:(i+1)*X_test.shape[0],:,:] = test_aug

print('completed augumentation')
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_30_aug.npy',X_train_aug)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_30_aug.npy',X_test_aug)
'''
