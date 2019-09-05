import numpy as np
from sklearn.model_selection import train_test_split

def train_valid_split_past():
    X = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/X_train_60_2_untrimmed.npy')
    y = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_train_fall_aug.npy')
    print(X.shape)
    print('finish loading datasets')
    y_r = np.tile(y,24)
    ind = range(X.shape[0])

    ind_train, ind_valid = train_test_split(ind, test_size=0.15, random_state=42)
    print('finish spliting')
    X_train = X[ind_train]
    print('obtained X train')
    X_valid = X[ind_valid]
    y_train = y_r[ind_train]
    y_valid = y_r[ind_valid]

    print(X_train.shape)
    print(X_valid.shape)

    print('finish allocating the train and validation set')
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/X_train_60_2_train.npy', X_train)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/X_train_60_2_valid.npy', X_valid)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/y_train_60_2_train.npy', y_train)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/y_train_60_2_valid.npy', y_valid)
