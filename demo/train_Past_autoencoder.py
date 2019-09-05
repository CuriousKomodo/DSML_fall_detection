import random
import numpy as np
import os
import sys
sys.setrecursionlimit(10000)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend.tensorflow_backend as K
import keras
from keras import callbacks, layers, losses, models, optimizers
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/generative_models_keras/')
from Past_Autoencoder_model import *
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from create_past_train_valid_split import *
from functions import *

window=60
t =  5 #attempt
repeats = 1
VAE=True
keep_3D=False
#THIS PART IS ONLY FOR THE DEMO.
valid_data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_untrimmed.npy')
dropout = True

if not dropout:
    model_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/'
    result_path = '/Users/kefei/Documents/mm_fall_detection/demo/results/2D/past/'
else:
    model_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/past/dropout/'
    result_path = '/Users/kefei/Documents/mm_fall_detection/demo/results/2D/past/dropout/'

print(model_path)
print(result_path)

if not keep_3D: #Only extract x,y coordinate
    input_dim = 34
    valid_data = crop_2D(valid_data)
else:
    input_dim = 51

valid_target = valid_data[:,::-1,:]

params = {'hidden_dim': 512,'latent_dim': 128, 'dropout_rate': 0.05, 'kl_weight':0.01,
        'VAE':VAE, 'input_dim':input_dim,'joint_dropout':dropout}

params_train = {'train': True,'keep_3D':False}
#params_valid = {'train': False}
optimizer=optimizers.Adam(0.0002)
#optimizer = optimizers.RMSprop(lr=0.0005,decay=0.5)
past_ae = past_LSTM_autoencoder(**params)
print('finish setting the model')
if past_ae.vae_loss:
    print('VAE approach')
    past_ae.model.compile(optimizer ,loss=past_ae.vae_loss)
else:
    past_ae.model.compile(optimizer ,loss=losses.mean_squared_error)

print('finish compile')
print(past_ae.model.summary())
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=0, min_lr=0.00005)
history_callback = past_ae.model.fit_generator(generator=Past_DataGenerator(**params_train),verbose=1,
                                            epochs=20, callbacks=[keras.callbacks.EarlyStopping(patience=1),reduce_lr],
                                            validation_data=(valid_data,valid_target))#Past_DataGenerator(**params_valid)


#make prediction
valid_recon = past_ae.model.predict(valid_data)
np.save(result_path+'past_vae_recon%s.npy'%t,valid_recon)
print('save reconstruction')

if VAE:
    mean, log_sigma = past_ae.enc_model.predict(valid_data)
    #np.save(result_path+'z_mean%s.npy'%t,mean)
    #np.save(result_path+'z_log_var%s.npy'%t,log_sigma)
    print('save the mean and log variance')
else:
    #valid_hidden = past_ae.enc_model.predict(valid_data)
    #np.save(result_path+'past_ae_hidden%s.npy'%t,valid_recon)
    print('save hidden')


if VAE:
    past_ae.model.save_weights(model_path+"past_VAE_model%s_weights.h5"%t)
    print('save weights')
    past_ae.enc_model.save_weights(model_path+"past_VAE_enc_model%s_weights.h5"%t)
    print('save hidden model weights')
    past_ae.enc_model.save( model_path+"/past_VAE_enc_model%s.h5"%t)
    print('save hidden model')
    past_ae.model.save(model_path+"past_VAE_model%s.h5"%t)
    print('save model')
else:
    past_ae.model.save_weights(model_path+"past_AE_model%s_weights.h5"%t)
    print('save weights')
    past_ae.enc_model.save_weights(model_path+"past_AE_enc_model%s_weights.h5"%t)
    print('save hidden model weights')
    past_ae.enc_model.save( model_path+"/past_AE_enc_model%s.h5"%t)
    print('save hidden model')
    past_ae.model.save(model_path+"past_AE_model%s.h5"%t)
    print('save model')
