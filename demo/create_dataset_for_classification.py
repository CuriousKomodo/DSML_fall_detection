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

def crop_2D(X):#takes 3D skeleton
    X = X.reshape(X.shape[0],X.shape[1],3,int(X.shape[-1]/3))
    X = X[:,:,:2,:] #only extract x,y cooridnates
    X = X.reshape(X.shape[0],X.shape[1],2*X.shape[-1])
    return X

VAE=True
keep_3D=False
dropout = True

if keep_3D:
    input_dim=51
    weight_path='/Users/kefei/Documents/mm_fall_detection/demo/models/3D/past/past_VAE_enc_model3_weights.h5'
else:
    input_dim=34
    #weight_path='/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/past_VAE_enc_model1_weights1.h5'
    weight_path='/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/dropout/past_VAE_enc_model5_weights.h5'
#apply the encoder model to obtain representations, at a certain step size
latent_dim = 128
hidden_dim= 512

params = {'latent_dim': latent_dim, 'input_dim':input_dim,'hidden_dim':hidden_dim,'GPU':False}
past_ae = past_LSTM_autoencoder(**params)
enc_model  =past_ae.enc_model
print(enc_model.summary())
enc_model.compile(optimizer='adam',loss=past_ae.vae_loss)

enc_model.load_weights(weight_path)
print('finish loading model')

X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_untrimmed.npy')
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_untrimmed.npy')
y_train= np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_train_fall_aug.npy')
y_train_r = np.tile(y_train,24)
y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_test_fall_aug.npy')
y_test_r = np.tile(y_test,24)
print('finish loading dataset')

if keep_3D:
    save_data_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/3D/'
else:
    if dropout:
        save_data_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/dropout/'
    else:
        save_data_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/'

print(save_data_dir)

if not keep_3D:
    X_train = crop_2D(X_train)
    X_test = crop_2D(X_test)

#clean the dataset.
def trim_dataset(data,label,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    data = data[length>=threshold]
    label =label[length>=threshold]
    return data, label


def remove_joints(data, max_num_remove=2,num_joints=17,dim=2, noise_density=0.05):
    incomplete_data = data.reshape(data.shape[0],data.shape[1], dim, num_joints)
    for i in range(data.shape[0]):
        #pick a random number of joints to remove
        r = np.random.randint(max_num_remove+1)
        joints_to_remove = np.random.randint(num_joints, size=r) #pick random joint to remove for each data
        for x in joints_to_remove:
            incomplete_data[i,:,:,x] = 0.0 #or let be a random number??
            #if noise_density>0:
            #    incomplete_data[i] = noisy_joints(data, num_noisy_joints=2, num_joints=17,density=0.05)
    incomplete_data = incomplete_data.reshape(data.shape[0],data.shape[1], dim*num_joints)
    return incomplete_data

X_train,y_train_r = trim_dataset(X_train,y_train_r)
X_test,y_test_r = trim_dataset(X_test,y_test_r)
#np.save(save_data_dir+'X_test_croppped.npy', X_test)
print(X_train.shape)

target_train = [1 if x==43 else 0 for x in y_train_r]
target_test = [1 if x==43 else 0 for x in y_test_r]



if dropout:
    np.save(save_data_dir+'target_train.npy', np.tile(target_train,2))
else:
    np.save(save_data_dir+'target_train.npy', target_train)
    
np.save(save_data_dir+'target_test.npy', target_test)

print('finish saving targets')
del target_test
del y_test_r
del target_train
del y_train_r

#Create a duplicate of X_train, where the joints are randomly removed?
if dropout:
    X_train_corrupt = remove_joints(X_train, max_num_remove=2)
    print('finish corrupting trianing set')

if VAE:
    if dropout:
        z_mean_train, z_log_var_train = enc_model.predict(np.concatenate([X_train,X_train_corrupt], axis=0))
    else:
        z_mean_train, z_log_var_train = enc_model.predict(X_train)
    #del X_train_corrupt
    del X_train
    np.save(save_data_dir+'z_mean_train.npy',z_mean_train)
    np.save(save_data_dir+'z_log_var_train.npy',z_log_var_train)
    print('finish train set')

print('finish saving training')

if VAE:
    z_mean_test, z_log_var_test = enc_model.predict(X_test)
    del X_test
    np.save(save_data_dir+'z_mean_test.npy',z_mean_test)
    np.save(save_data_dir+'z_log_var_test.npy',z_log_var_test)
else:
    hidden_train = enc_model.predict(X_train)
    hidden_test = enc_model.predict(X_test)
    np.save(save_data_dir+'hidden_train.npy',hidden_train)
    np.save(save_data_dir+'hidden_test.npy',hidden_test)
print('finish saving test')


#Not sure if i should concatenate hidden train and test and form a new training/test set...
