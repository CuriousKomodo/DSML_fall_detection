#Train the dropout AE

from sklearn.model_selection import StratifiedKFold
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

import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from create_DAE_dataset import *
import gc

'''
hidden_dims = [256,512,1024,2048]
dropouts = [0.01,0.05,0.1]

repeats = 2
val_losses = np.zeros((repeats, len(hidden_dims),len(dropouts)))

for r in range(repeats):
    train_valid_split(keep_3D=False)
    print('finish spliting')
    target_valid = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D/X_train_valid.npy')
    input_valid = remove_joints(target_valid, max_num_remove=5,num_joints=17,dim=2)

    if r==0:
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D/dropout_train_valid.npy', input_valid)

    for i in range(len(hidden_dims)):
        for j in range(len(dropouts)):
            hidden_dim = hidden_dims[i]
            dropout = dropouts[j]
            optimizer=optimizers.Adam(0.0002)
            model_params = {'hidden_dim': hidden_dim, 'input_dim':34, 'dropout':dropout}
            print(model_params)
            dae = DAE(**model_params)
            model = dae.model
            model.compile(optimizer=optimizer, loss=losses.mean_squared_error)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=4, min_lr=0.00005)
            train = model.fit_generator(generator=DAE_generator(), epochs=15,validation_data=(input_valid, target_valid),
                                callbacks=[keras.callbacks.EarlyStopping(patience=3),reduce_lr])
            val_loss = train.history['val_loss'][-1]
            model_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/DAE/'
            #model.save_weights(model_path+"DAE_model%s_weights.h5"%t)
            val_losses[r,i,j] = val_loss
            K.clear_session()

print('average val loss:', np.mean(val_losses, axis=0))
np.save('/Users/kefei/Documents/mm_fall_detection/demo/results/2D/DAE/',val_losses)
'''

hidden_dim = 2048
dropout = 0.01
optimizer=optimizers.Adam(0.0002)
train_data_dir ='/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D/X_train.npy'
model_params = {'hidden_dim': hidden_dim, 'input_dim':34, 'dropout':dropout}
generator_params = {'train_data_dir':train_data_dir}
print(model_params)
dae = DAE(**model_params)

test_target = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D/X_test.npy')
test_data = remove_joints(test_target,max_num_remove=5)

model = dae.model
model.compile(optimizer=optimizer, loss=losses.mean_squared_error)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=1, min_lr=0.00005)

train = model.fit_generator(generator=DAE_generator(**generator_params), epochs=15,validation_data=(test_data, test_target),
                            callbacks=[keras.callbacks.EarlyStopping(patience=3),reduce_lr])

test_recon = model.predict(test_data)
np.save('/Users/kefei/Documents/mm_fall_detection/demo/results/2D/DAE_recon.npy',test_recon)

train_loss = train.history['loss']
val_loss = train.history['val_loss']
print('train_loss=', train_loss)
print('validation_loss=', val_loss)
model_path = '/Users/kefei/Documents/mm_fall_detection/demo/models/2D/DAE/'
model.save_weights(model_path+'weights2048.h5')


'''
#Maybe save a few examples, and compare with reconstruction

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
            model_params = {'hidden_dim': hidden_dim, 'input_dim':34}
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
                # skf.get_n_splits(data, target)
            print('splited training/validation set')
            train_losses = np.zeros(n_folds)
            val_losses = np.zeros(n_folds)
            i = 0

            for train, test in skf.split(data, dummy_target):
                print("Running Fold", i + 1, "/", n_folds)

                optimizer=optimizers.Adam(0.0002)
                dae = DAE()
                model = dae.model
                model.compile(optimizer=optimizer, loss=losses.mean_squared_error)
                #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.00005)
                #model.fit_generator(generator=DAE_generator(), epochs=10,validation_data=(input_valid, target_valid),callbacks=[keras.callbacks.EarlyStopping(patience=3),reduce_lr])
                print('model compiled')
                #train_loss, val_loss = train_and_evaluate_model(model, data[train], target[train],
                #                                                    data[test], target[test], optimizer)
                train_losses[i] = train_loss
                val_losses[i] = val_loss
                i += 1

            val_losses_array[l, d] = np.mean(val_losses)
            train_losses_array[l, d] = np.mean(train_losses)
    return train_losses_array, val_losses_array

'''
