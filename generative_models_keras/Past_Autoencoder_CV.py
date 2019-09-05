#$ -l gpu=2
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend.tensorflow_backend as K
import keras
from keras.backend.tensorflow_backend import set_session
from keras import callbacks, layers, losses, models, optimizers
from Past_Autoencoder_model import *



def load_data(normalize=True, remove_joints=None ,list_as_input=False):  # temporary.
    X = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_30_untrimmed.npy')

    #X = np.concatenate([X_train ,X_val] ,axis=0)
    m = X.shape[0]

    if list_as_input:
        X_list = []
        Y_list = []
        lengths = np.sum(np.sign(np.sum(np.sign(X) ,axis=-1)) ,axis=-1)
        for i in range(len(lengths)):
            X_example = X[i ,:int(lengths[i]) ,:]
            Y_example = X_example[::-1 ,:]
            if normalize:
                X_list.append(X_example *0.8 +0.1)
                Y_list.append(Y_example *0.8 +0.1)
            else:
                X_list.append(X_example)
                Y_list.append(Y_example)
        X = X_list
        Y = Y_list


    else:
        Y = np.flip(X ,axis=1)

    dummy = np.zeros(m)
    return X ,dummy ,Y


def train_and_evaluate_model(model, data_train, target_train, data_val, target_val, optimizer=optimizers.Adam(0.0005)):
    model.compile(optimizer ,loss=losses.mean_squared_error)
    print('model compiled')
    history_callback = model.fit(data_train, target_train, validation_data=(target_val, target_val),batch_size=150,epochs=10,
                                 callbacks=[keras.callbacks.EarlyStopping(patience=2)])
    loss_history = history_callback.history["loss"]
    val_loss_history = history_callback.history['val_loss']  # returns the last errors
    return loss_history[-1] ,val_loss_history[-1]


def Cross_Validation(n_folds, latent_dim_list, dropout_rate_list, optimizer):
    train_losses_array = np.zeros((len(latent_dim_list), len(dropout_rate_list)))
    val_losses_array = np.zeros((len(latent_dim_list), len(dropout_rate_list)))
    data, dummy_target, target = load_data(normalize=True, remove_joints=None, list_as_input=False)
    # KFold(n_splits=2, random_state=None, shuffle=False)
    print('data_loaded')
    for l in range(len(latent_dim_list)):
        for d in range(len(dropout_rate_list)):
            latent_dim = latent_dim_list[l]
            dropout_rate = dropout_rate_list[d]
            print("latent_dim=", latent_dim, "dropout_rate=", dropout_rate)
            params = {'latent_dim': latent_dim, 'dropout_rate': dropout_rate}
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
                # skf.get_n_splits(data, target)
            print('splited training/validation set')
            train_losses = np.zeros(n_folds)
            val_losses = np.zeros(n_folds)
            i = 0

            for train, test in skf.split(data, dummy_target):
                print("Running Fold", i + 1, "/", n_folds)
                past_ae = past_LSTM_autoencoder(**params)
                print('model compiled')
                train_loss, val_loss = train_and_evaluate_model(past_ae.model, data[train], target[train],
                                                                    data[test], target[test], optimizer)
                train_losses[i] = train_loss
                val_losses[i] = val_loss
                i += 1

            val_losses_array[l, d] = np.mean(val_losses)
            train_losses_array[l, d] = np.mean(train_losses)
    return train_losses_array, val_losses_array


#parameter
latent_dim_list=[128,256,512]
dropout_rate_list=[0.05, 0.1, 0.2]

#hyperparameter tuning...
lr_list = [0.0005, 0.001, 0.01]
#optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
optimizer=optimizers.Adam(0.0001)
train_losses_array,val_losses_array = Cross_Validation(10,latent_dim_list,dropout_rate_list,optimizer)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/CV_RMS_train_losses.npy',train_losses_array)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/CV_RMS_train_losses.npy',val_losses_array)
