#Evaluates the best parameters on the test dataset
#Save z_mean, z_log_var for PCA evaluation for all 4 models
#Save reconstruction (10 non-fall,  10 fall) for all 4 models - MSE
#Compute the average recon loss over time, visualise all 4 models - KL
#For classifier addition models, visualise the scores (i.e. clasification report)  - CE


import random
import numpy as np
import os

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
from keras.utils.vis_utils import plot_model
from keras.callbacks import LambdaCallback
from Past_Autoencoder_model import *
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from functions import *
from sklearn.metrics import classification_report
import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt import dump, load
#kl_type = Categorical(categories = ['constant_kl', 'kl_anneal','kl_cyclic'], name='kl_weight_type')


def load_data():  # temporary.
    X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_trimmed.npy')
    X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_trimmed.npy')
    #y_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_train_fall_aug_trimmed.npy')
    X_train = X_train[:500]
    #y_train = y_train[:500]
    X_test = X_test[:500]
    #y_test = y_test[:500]
    print(X_test.shape)
    #y_train = np.asarray([1 if x==43 else 0 for x in y_train])
    #y_test = np.asarray([1 if x==43 else 0 for x in y_test])
    return X_train, X_test


def fitness(kl_type, hyperparams, result_path, model_path): #annealtime
    hidden_dim = hyperparams[0]
    latent_dim = hyperparams[1]
    hidden_W_regularizer = hyperparams[2]
    dropout_rate = hyperparams[3]
    classifier_dense_dim = hyperparams[4]
    learning_rate = hyperparams[5]
    if kl_type=='klcyclic':
        num_cycles = hyperparams[6]
    else:
        annealtime = hyperparams[6]
    maximum_kl_weight = hyperparams[7]

    X_train, X_test = load_data()
    weight = K.variable(0.0)
    params =   {'hidden_dim': hidden_dim,'latent_dim': latent_dim,'W_regularizer_val': hidden_W_regularizer,
            'dropout_rate': dropout_rate, 'classifier':True,'classifier_dense_dim': classifier_dense_dim,
            'kl_weight':weight, 'VAE':True, 'input_dim':51, 'joint_dropout':False,
            'GPU':True,'add_losses':True}

    past_ae = past_LSTM_autoencoder(**params)
    past_ae.model.compile(optimizer,loss={'reconstructed_poses':past_ae.vae_loss},
                            loss_weights = {'reconstructed_poses':1.0}
                            ) #without classifier
    past_ae.enc_model.load_weights(model_path + 'enc_model.h5')

    z_mean, z_log_var = past_ae.enc_model.predict(X_train)
    np.save(result_path+'z_mean_train.npy',z_mean)
    np.save(result_path+'z_log_var_train.npy',z_log_var)

    z_mean, z_log_var = past_ae.enc_model.predict(X_test)
    np.save(result_path+'z_mean_test.npy',z_mean)
    np.save(result_path+'z_log_var_test.npy',z_log_var)
    K.clear_session()

    return None

def collect_result(kl_type):

    if kl_type=='klanneal':
        hyperparameters = [1024, 128, 0.004875173292623599, 0.04085806061902107, 32, 0.00025378664002876164, 861, 0.003950403393695316]
        model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/without_classifier/'%kl_type
        result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type

    elif kl_type=='klcyclic':
        hyperparameters = [768, 256, 0.001, 0.12, 32, 0.0005, 10, 0.0001]
        model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/with_classifier/'%kl_type
        result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type

    fitness(hyperparameters, kl_type, result_path, model_path)

#already know for klanneal with classifier, just without reconstructed poses

#collect_result(kl_type='klanneal', classifier_bool=True)
#collect_result(kl_type='klanneal', classifier_bool=False)
collect_result(kl_type='klanneal')
collect_result(kl_type='klcyclic')
