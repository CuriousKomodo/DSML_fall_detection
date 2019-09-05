import numpy as np
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/generative_models_keras/')
import Past_Autoencoder_model
from Past_Autoencoder_model import *
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
VAE=True
datatype='integrated' #or 'clean'
keep_3D = False
demo=True
latent_dim = 128
hidden_dim = 512
dropout = True

def extract_2D(data):
    input_dim = 34
    data = data.reshape(data.shape[0],data.shape[1],3,int(data.shape[-1]/3)) #reshape it back to (n,T,3,J)
    data = data[:,:,:2,:] #only extract x,y cooridnates
    data = data.reshape(data.shape[0],data.shape[1],2*data.shape[-1])#reshape it to (n,T,2,J)
    return data

#clean the dataset.
def trim_dataset(data,label,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    data = data[length>=threshold,:,:]
    label =label[length>=threshold]
    return data, label

if keep_3D:
    if not demo:
        weight_path='/Users/kefei/Documents/mm_fall_detection/models/win60/past_AE_enc_model8_weights.h5'
    else:
        weight_path='/Users/kefei/Documents/mm_fall_detection/demo/models/past/3D/past_AE_enc_model1_weights.h5'
else:
    if dropout:
        weight_path='/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/dropout/past_VAE_enc_model5_weights.h5'
        data_save_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/dropout/'
    else:
        weight_path='/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/past_VAE_enc_model1_weights1.h5'
        data_save_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/''

#from generative_models_keras.Past_Autoencoder_model import *

X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/%s/3D/X_train_60_2_untrimmed.npy'%datatype)
y_train= np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_train_fall_aug.npy')
y_train_r = np.tile(y_train,24)

X_train,y_train_r = trim_dataset(X_train,y_train_r)
print('finishing trimming training')

X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/%s/3D/X_test_60_2_untrimmed.npy'%datatype)
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_test_fall_aug.npy')
y_test_r = np.tile(y_test,24)

X_test,y_test_r = trim_dataset(X_test,y_test_r)
print('finishing trimming test')

if not keep_3D:
    X_train = extract_2D(X_train)
    X_test = extract_2D(X_test)

print('X_train.shape=',X_train.shape)
print('y_train.shape=',y_train_r.shape)
print('X_test.shape=',X_test.shape)

#apply the encoder model to obtain representations, at a certain step size
if not keep_3D:
    input_dim = 34
else:
    input_dim = 51

params = {'latent_dim': latent_dim, 'input_dim':input_dim, 'hidden_dim':hidden_dim}
past_ae = past_LSTM_autoencoder(**params)
model,enc_model  = past_ae.model, past_ae.enc_model
enc_model.load_weights(weight_path)
print('finish loading model')

if VAE:
    z_mean_train, z_log_var_train = enc_model.predict(X_train)
    np.save(data_save_dir + 'z_mean_train.npy', z_mean_train)
    np.save(data_save_dir + 'z_log_var_train.npy', z_log_var_train)
    del X_train
    del z_mean_train
    del z_log_var_train

    z_mean_test, z_log_var_test = enc_model.predict(X_test)
    np.save(data_save_dir + 'z_mean_test.npy', z_mean_test)
    np.save(data_save_dir + 'z_log_var_test.npy', z_log_var_test)
    del X_test
    del z_mean_test
    del z_log_var_test

else:
    hidden_train = enc_model.predict(X_train)
    hidden_test = enc_model.predict(X_test)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/classification/hidden_train_aug.npy', hidden_train)
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/classification/hidden_test_aug.npy', hidden_test)
print('finish prediction')


target_train = [1 if x==43 else 0 for x in y_train_r]
target_test = [1 if x==43 else 0 for x in y_test_r]
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/target_train_aug.npy', target_train)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/target_test_aug.npy', target_test)
#Not sure if i should concatenate hidden train and test and form a new training/test set...
