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
from Dropout_Autoencoder import *
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/demo/')
from past_AE_DAE_model import *
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from create_past_train_valid_split import *

optimizer=optimizers.Adam(0.0002)
past_params = {}
#optimizer = optimizers.RMSprop(lr=0.0005,decay=0.5)
past_ae = past_LSTM_autoencoder(**past_params)
print('finish setting the model')

if past_ae.vae_loss:
    print('VAE approach')
    past_ae.model.compile(optimizer ,loss=past_ae.vae_loss)
else:
    past_ae.model.compile(optimizer ,loss=losses.mean_squared_error)

val_loss_matrix = np.zeros((repeats, len(latent_dims), len(dropout_rates)))

for r in range(repeats):
    train_valid_split_past()
    print('complete split')
    valid_data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/X_test_60_2_untrimmed.npy')
    valid_data = valid_data*0.8+0.1
    #if DAE:
    valid_data = remove_joints(valid_data)
    valid_target = valid_data[:,::-1,:]
    print('finish loading valid dataset')

    for i in range(len(latent_dims)):
        for j in range(len(dropout_rates)):
            past_ae = None
            latent_dim = latent_dims[i]
            print(latent_dim)
            dropout_rate = dropout_rates[j]
            past_params = {'hidden_dim': hidden_dim,'latent_dim': latent_dim, 'dropout_rate': dropout_rate, 'kl_weight':0.01, 'VAE':VAE, 'input_dim':51}
            generator_params = {'train': True,'DAE':True} #So the corrupted data is loaded as input
            optimizer=optimizers.Adam(0.0002)
            #optimizer = optimizers.RMSprop(lr=0.0005,decay=0.5)

            past_ae = past_LSTM_autoencoder(**past_params)
            print('finish setting the model')
            if past_ae.vae_loss:
                print('VAE approach')
                past_ae.model.compile(optimizer ,loss=past_ae.vae_loss)
            else:
                past_ae.model.compile(optimizer ,loss=losses.mean_squared_error)

            print('finish compile')
            print(past_ae.model.summary())
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.00005)
            history_callback = past_ae.model.fit_generator(generator=Past_DataGenerator(**generator_params),verbose=1,
                                                        epochs=20, callbacks=[keras.callbacks.EarlyStopping(patience=3),reduce_lr],
                                                        validation_data=(valid_data,valid_target))#Past_DataGenerator(**params_valid)
            K.clear_session()

        loss_history = history_callback.history["loss"]
        val_loss_history = history_callback.history['val_loss']
        val_loss_matrix[r,i,j] = val_loss_history[-1] #final validation loss.
