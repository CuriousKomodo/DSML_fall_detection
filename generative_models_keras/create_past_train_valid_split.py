import numpy as np
from sklearn.model_selection import train_test_split

def train_valid_split_past():
    X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/X_train_60_2_untrimmed.npy')
    y_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_train_fall_aug.npy')
    y_train_r = np.tile(y_train,24)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train_r, test_size=0.15, random_state=42)
    #train = data_list[0]
    print(X_train.shape)
    print(X_valid.shape)

    print('finish allocating the train and validation set')
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/X_train_60_2_train.npy', X_train)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/X_train_60_2_valid.npy', X_valid)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/y_train_60_2_train.npy', y_train)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/y_train_60_2_valid.npy', y_valid)

train_valid_split_past()
