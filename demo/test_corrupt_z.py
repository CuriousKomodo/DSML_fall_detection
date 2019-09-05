#corrupt the third dimension
import numpy as np
import cv2
import sys
from load_models import *
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import gc
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from functions import *

#load model!
past_ae_weight_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/dropout/past_VAE_enc_model5_weights.h5'
past_ae_weight_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/past_VAE_enc_model4_weights.h5'
pca_dir = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/pca.pkl'
svm_dir = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/svm.pkl'

enc_model = load_encoder(weight_path=past_ae_weight_path)
pca,svm = load_classifier(pca_dir,svm_dir)
print('complete loading models')

'''
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_untrimmed.npy')
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_test_fall_aug.npy')
y_test_r = np.tile(y_test,24)


def trim_dataset(data,label,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    data = data[length>=threshold,:,:]
    label =label[length>=threshold]
    return data, label

X_test = crop_2D(X_test)
X_test,y_test_r = trim_dataset(X_test,y_test_r)
target_test = [1 if x==43 else 0 for x in y_test_r]
gc.collect()
'''
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/X_test_cropped.npy')
target_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/target_test.npy')
print('finish loading data')
#Make the third dimension gaussian noise.


def corrupt_z(data, noise=False, keep_3D = False):
    if keep_3D:
        dim = 3
    else:
        dim = 2
    if noise:
        corrupted_data = np.random.rand(data.shape[0],data.shape[1],dim,17)
    else:
        corrupted_data = np.zeros((data.shape[0],data.shape[1],dim,17))
        corrupted_data[:,:,:,-1] = data[:,:,:,-1].mean()

    corrupted_data[:,:,:,:2] = data[:,:,:,:2]
    return corrupted_data

#Maybe pick what joints to remove and see which one has the most impact?
def remove_joints(data, num_remove = 2, joints_to_remove=None):
    incomplete_data = data.copy()
    #print(data.shape)
    if not joints_to_remove:
        for i in range(data.shape[0]):
            if i%10000==0:
                print(i)
            joints_to_remove = np.random.randint(17, size=num_remove+1) #pick random joint to remove for each data
            #print(joints_to_remove )
            for x in joints_to_remove:
                incomplete_data[i,:,:,x] = 0.0
    else:
        for x in joints_to_remove: #remove the given joints for the whole dataset
            incomplete_data[:,:,:,x] = 0.0
    return incomplete_data

def corrupt_dataset(data, noisy_z, missing_joints, joints_to_remove=None,keep_3D= False):
    if keep_3D:
        dim = 3
    else:
        dim = 2
    data = data.reshape(data.shape[0],data.shape[1],dim,17)
    if noisy_z:
        data = corrupt_z(data)
    if missing_joints:
        data = remove_joints(data,joints_to_remove=joints_to_remove)
    data = data.reshape(data.shape[0],data.shape[1],dim*17)
    return data

for i in range(17):
    print('removed joint=',i)
    corrupted_data = corrupt_dataset(X_test, noisy_z=False, missing_joints=True,joints_to_remove=[i])
    #corrupted_data = X_test
    print('finish corrupting data')
    gc.collect()
    z_mean, z_log_var =enc_model.predict(corrupted_data)
    features = pca.transform(np.concatenate([z_mean,z_log_var],axis=-1))
    pred = svm.predict(features)
    print('finish prediction')
    classification_report_model = classification_report(target_test,pred)
    confusion_matrix_model = confusion_matrix(target_test,pred)

    print('classification report=', classification_report_model)
    print('confusion_matrix=', confusion_matrix_model)
