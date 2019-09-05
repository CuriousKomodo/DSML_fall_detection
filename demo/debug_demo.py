import numpy as np
import os
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from read_real_time_skeletons import *
from load_models import *
sys.path.insert(0, '/Users/kefei/Documents/openpose/build/examples/tutorial_api_python/')
#import capture_realtime_skeletons
import time
from datetime import datetime
import gc
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/generative_models_keras/')
import Past_Autoencoder_model
from Past_Autoencoder_model import *


#load model!
def crop_2D(X):#takes 3D skeleton
    X = X.reshape(X.shape[0],X.shape[1],3,int(X.shape[-1]/3))
    X = X[:,:,:2,:] #only extract x,y cooridnates
    X = X.reshape(X.shape[0],X.shape[1],2*X.shape[-1])
    return X

def trim_dataset(data,label,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    data = data[length>=threshold,:,:]
    label =label[length>=threshold]
    return data, label

weight_path='/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/dropout/past_VAE_enc_model2_weights.h5'
params = {'latent_dim': 256, 'input_dim':34,'hidden_dim':512, 'GPU':False}
past_ae = past_LSTM_autoencoder(**params)
enc_model  =past_ae.enc_model
enc_model.compile(optimizer='adam',loss=past_ae.vae_loss)
enc_model.load_weights(weight_path)

print('finish loading model')
#enc_model = load_encoder()
#pca,svm = load_classifier()
#DAE = load_DAE()
print(enc_model.summary())
print('complete loading models')


save_data_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/'
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_untrimmed.npy')
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_test_fall_aug.npy')
y_test_r = np.tile(y_test,24)

X_test,y_test_r = trim_dataset(X_test,y_test_r)
X_test = crop_2D(X_test)
z_mean_test = np.load(save_data_dir+'z_mean_test.npy')
z_log_var_test= np.load(save_data_dir+'z_log_var_test.npy')


z_mean_pred, z_log_var_pred =enc_model.predict(X_test)
#np.save(save_data_dir+'debug_z_mean_test.npy',z_mean_pred)
print(np.sum((z_mean_pred - z_mean_test)**2))

np.save(save_data_dir+'X_test_crop.npy',X_test)
