import numpy as np
from sklearn.model_selection import train_test_split

def train_valid_split(keep_3D=True):
    X = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_fall_aug_int.npy')
    y = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_train_fall_aug.npy')

    if not keep_3D:
        X= X.reshape(X.shape[0],X.shape[1],3,int(X.shape[-1]/3))
        X = X[:,:,:2,:] #only extract x,y cooridnates
        X = X.reshape(X.shape[0],X.shape[1],2*X.shape[-1])

    print(X.shape)
    print('finish loading datasets')

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)
    print('finish spliting')
    print(X_train.shape)
    print(X_valid.shape)

    print('finish allocating the train and validation set')

    X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[-1], order='F')
    X_valid = X_valid.reshape(X_valid.shape[0]*X_valid.shape[1], X_valid.shape[-1], order='F')
    print(X_train.shape)

    #if data_augumentation:

    if keep_3D:
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_train.npy', X_train)
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_valid.npy', X_valid)
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_train_train.npy', y_train)
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_train_valid.npy', y_valid)
    else:
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D_DAE/X_train_train.npy', X_train)
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D_DAE/X_train_valid.npy', X_valid)
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D_DAE/y_train_train.npy', y_train)
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D_DAE/y_train_valid.npy', y_valid)


'''
        X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_fall_aug_int.npy')
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],3,int(X_test.shape[-1]/3))
        X_test = X_test[:,:,:2,:] #only extract x,y cooridnates
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],2*X_test.shape[-1])
        X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1],X_test.shape[-1], order='F')
        X = X.reshape(X.shape[0]*X.shape[1],X.shape[-1], order='F')
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D_DAE/X_train.npy',X)
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D_DAE/X_test.npy',X_test)
'''
#train_valid_split(keep_3D=False)
