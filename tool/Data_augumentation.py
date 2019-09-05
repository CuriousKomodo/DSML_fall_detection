import numpy as np
import math

X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train.npy')
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test.npy')
y_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_train.npy')
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_test.npy')
print('finish loading dataset')

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


def rotate_45(v, axis=(0,1,0), theta = np.pi/4.0):
    r = np.dot(rotation_matrix(axis, theta), v)
    return r

def rotate_90(v, axis=(0,1,0), theta = np.pi/2.0):
    r = np.dot(rotation_matrix(axis, theta), v)
    return r

def rotate_135(v, axis=(0,1,0), theta = np.pi*3.0/4.0):
    r = np.dot(rotation_matrix(axis, theta), v)
    return r

def rotate_180(v, axis=(0,1,0), theta = np.pi):
    r = np.dot(rotation_matrix(axis, theta), v)
    return r

#only want to augument the fall num_fall_examples
num_fall_test = sum(y_test==43)
print('old number of fall examples in test=', num_fall_test)
num_fall_train = sum(y_train==43)
print('old number of fall examples in train=', num_fall_train)

def rotate_all(data):
    augumented_fall_data = []
    for i in range(data.shape[0]):
        rotated_sequences = []
        sequence = data[i,:,:]
        sequence = np.reshape(sequence,(sequence.shape[0], 3, 25))
        rotated_sequence_45 = np.apply_along_axis(rotate_45, 1, sequence)
        rotated_sequence_45 = rotated_sequence_45.reshape(rotated_sequence_45.shape[0],75)
        rotated_sequence_90 = np.apply_along_axis(rotate_90, 1, sequence)
        rotated_sequence_90 = rotated_sequence_90.reshape(rotated_sequence_90.shape[0],75)
        rotated_sequence_135 = np.apply_along_axis(rotate_135, 1, sequence)
        rotated_sequence_135 = rotated_sequence_135.reshape(rotated_sequence_135.shape[0],75)
        rotated_sequences.append(rotated_sequence_45)
        rotated_sequences.append(rotated_sequence_90)
        rotated_sequences.append(rotated_sequence_135)
        rotated_sequences = np.stack(rotated_sequences)
        augumented_fall_data.append(rotated_sequences)


def rotate_fall(data, num_fall):
    augumented_fall_data = []
    for i in range(num_fall):#loop through the examples of falling
        rotated_sequences = []
        sequence = data[-num_fall + i,:,:]
        sequence = np.reshape(sequence,(sequence.shape[0], 3, 25))
        #sequence = np.zeros((sequence.shape[0], 3, 25))

        rotated_sequence_45 = np.apply_along_axis(rotate_45, 1, sequence)
        rotated_sequence_45 = rotated_sequence_45.reshape(rotated_sequence_45.shape[0],75)
        rotated_sequences.append(rotated_sequence_45)


        rotated_sequence_90 = np.apply_along_axis(rotate_90, 1, sequence)
        rotated_sequence_90 = rotated_sequence_90.reshape(rotated_sequence_90.shape[0],75)
        rotated_sequences.append(rotated_sequence_90)


        rotated_sequence_135 = np.apply_along_axis(rotate_135, 1, sequence)
        rotated_sequence_135 = rotated_sequence_135.reshape(rotated_sequence_135.shape[0],75)
        rotated_sequences.append(rotated_sequence_135)

        rotated_sequence_180 = np.apply_along_axis(rotate_180, 1, sequence)
        rotated_sequence_180 = rotated_sequence_180.reshape(rotated_sequence_180.shape[0],75)
        rotated_sequences.append(rotated_sequence_180)

        rotated_sequences = np.stack(rotated_sequences)
        augumented_fall_data.append(rotated_sequences)
        #print(rotated_sequences.shape)
        if i%50==0:
            print('i=',i)
            print(rotated_sequences.max())
            print(rotated_sequences.min())


    augumented_fall_data = np.concatenate(augumented_fall_data)
    print('shape of augumented', augumented_fall_data.shape)
    data = np.concatenate([data,augumented_fall_data],axis=0)
    new_num_fall = augumented_fall_data.shape[0]
    return data, new_num_fall

X_train_aug,new_num_fall_train = rotate_fall(X_train, num_fall_train)
X_test_aug,new_num_fall_test = rotate_fall(X_test, num_fall_test)
y_train_aug =  list(y_train)
y_train_aug.extend([43]*(new_num_fall_train))
y_test_aug =  list(y_test)
y_test_aug.extend([43]*(new_num_fall_test))

def normalize_data(data):
    v_min = data.min(axis=(1,2), keepdims=True)
    v_max = data.max(axis=(1,2), keepdims=True)
    data_n = (data - v_min)/(v_max - v_min)
    #mask = np.all(np.isnan(data_n) | np.equal(data_n, 0), axis=(1,2))
    #data_n = data_n[~mask,:,:]
    #label_n = label[~mask]
    #length_nn  = length[~mask]
    #data_nn = data_n
    #label_nn = label_n
    return data_n

'''
X_train_aug= normalize_data(X_train_aug)
print(X_train_aug.min(),X_train_aug.max())
X_test_aug = normalize_data(X_test_aug)
'''

print('new number of fall examples in train=',new_num_fall_train)
print('new number of fall examples in test=',new_num_fall_test)

print(X_train_aug.shape)
print(np.array(y_train_aug).shape)


print('completed fall augumentation')
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_fall_aug.npy',X_train_aug)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_fall_aug.npy',X_test_aug)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_train_fall_aug.npy',np.array(y_train_aug))
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_test_fall_aug.npy',np.array(y_test_aug))


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
