#Create a training set for Future VAE
import numpy as np
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/generative_models_keras/')

#import Past_Autoencoder_model
#from Past_Autoencoder_model import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

VAE=True
current_step = 14

if VAE:
    weight_path='/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/past_VAE_enc_model1_weights.h5'
else:
    weight_path='/Users/kefei/Documents/mm_fall_detection/models/win60/past/AE/past_AE_enc_model8_weights.h5'
#from generative_models_keras.Past_Autoencoder_model import *

X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_60_2_untrimmed.npy')
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_60_2_untrimmed.npy')
y_train= np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_train_fall_aug.npy')
y_train_r = np.tile(y_train,24)
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_test_fall_aug.npy')
y_test_r = np.tile(y_test,24)

#clean the dataset.
def trim_dataset(data,label,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    data = data[length>=threshold,:,:]
    label =label[length>=threshold]
    return data, label

X_train,y_train_r = trim_dataset(X_train,y_train_r,20)
X_test,y_test_r = trim_dataset(X_test,y_test_r,20)
print('X_train.shape=',X_train.shape)
print('y_train.shape=',y_train_r.shape)
print('X_test.shape=',X_test.shape)

'''
#apply the encoder model to obtain representations, at a certain step size
hidden_dim = 1024
latent_dim = 128
params = {'hidden_dim': hidden_dim, 'VAE':VAE, 'latent_dim':latent_dim}
past_ae = past_LSTM_autoencoder(**params)
model,enc_model  = past_ae.model, past_ae.enc_model
enc_model.load_weights(weight_path)
print('finish loading model')

if VAE:
    z_mean_train, z_log_var_train = enc_model.predict(X_train[:,:current_step+1,:])
    z_mean_test, z_log_var_test = enc_model.predict(X_test[:,:current_step+1,:]) #make sure current frame is also encoded!
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/z_mean_train.npy',z_mean_train)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/z_log_var_train.npy', z_log_var_train)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/z_mean_test.npy', z_mean_test)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/z_log_var_test.npy', z_log_var_test)

else:
    hidden_train = enc_model.predict(X_train[:,:current_step+1,:])
    hidden_test = enc_model.predict(X_test[:,:current_step+1,:])
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/hidden_train_aug.npy', hidden_train)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/hidden_test_aug.npy', hidden_test)

print('finish predictions')
'''
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_train.npy', X_train[:,current_step:,:])#include the current step.
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_test.npy', X_test[:,current_step:,:])
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/y_train_trimmed.npy', y_train)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/y_test_trimmed.npy', y_test)
print('saved future sequences and labels')


'''
target_train = [1 if x==43 else 0 for x in y_train_r]
target_test = [1 if x==43 else 0 for x in y_test_r]
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/classification/target_train_aug.npy', target_train)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/classification/target_test_aug.npy', target_test)
#Not sure if i should concatenate hidden train and test and form a new training/test set...
'''
