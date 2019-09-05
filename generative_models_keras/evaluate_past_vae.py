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
config.gpu_options.per_process_gpu_memory_fraction = 0.9
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
from sklearn.metrics import classification_report, confusion_matrix

import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt import dump, load
#kl_type = Categorical(categories = ['constant_kl', 'kl_anneal','kl_cyclic'], name='kl_weight_type')
#y_test_original = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed25.npy')
y_test_original = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed.npy')
'''
def return_false_positives(y_test,pred):
    diff = pred - y_test #FP: 1, FN:-1
    np.save(result_path + 'FP_map.npy', diff)
    return

def return_false_negatives(X_test,  X_test_recon, y_test, pred):
    diff = pred - y_test #FP: 1, FN:-1
    np.save(result_path + 'FN_map.npy')
    false_negatives = X_test[diff==-1]
    print('number of false positives=', false_negatives.shape[0])
    false_negatives_recon = X_test_recon[diff==-1]
    return false_negatives,false_negatives_recon
'''

class KLLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('kl_loss'))

class MSELossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('mse_loss'))

class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self, weight,kl_annealtime, max_kl_weight):
        self.klstart = 0
        self.kl_weight = weight
        self.kl_annealtime = kl_annealtime
        self.max_kl_weight = max_kl_weight

    def on_epoch_end (self, epoch, logs={}):
        if epoch > self.klstart :#grows linearly towards one
            new_weight = min(K.get_value(self.kl_weight) + (self.max_kl_weight/ self.kl_annealtime), self.max_kl_weight)
            #print(new_weight)
            K.set_value(self.kl_weight, new_weight)
        #print ("Current KL Weight is " + str(K.get_value(self.kl_weight)))
class CyclicAnnealingCallback(keras.callbacks.Callback):
    def __init__(self, weight,M, max_kl_weight):
        self.kl_weight = weight
        self.R = 0.5 #keep this
        self.M = M
        self.T = 20
        self.max_kl_weight = max_kl_weight

    def on_epoch_end (self, epoch, logs={}):
        tau = np.mod(epoch, int(self.T/self.M))*1.0 /int(self.T/self.M)
        if tau>=self.R:
            new_weight = self.max_kl_weight  #capped at 0.0002, because posterior collapose happens!
        else:
            new_weight = tau * self.max_kl_weight

        K.set_value(self.kl_weight, new_weight)
        #print ("Current KL Weight is " + str(K.get_value(self.kl_weight)))




def load_data():  # temporary.
    #X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_untrimmed.npy')
    #y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_test_fall_aug.npy')
    #y_test = np.tile(y_test, 24)
    X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_trimmed.npy')
    y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed.npy')
    X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_trimmed.npy')
    y_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/labels/y_train_fall_aug_trimmed.npy')
    print(X_train.shape, y_train.shape)
    #X_train = X_train[:500]
    #y_train = y_train[:500]
    #X_test = X_test[:500]
    #y_test = y_test[:500]
    print(X_test.shape)
    y_train = np.asarray([1 if x==43 else 0 for x in y_train])
    y_test = np.asarray([1 if x==43 else 0 for x in y_test])
    return X_train, y_train, X_test, y_test


def fitness_baseline(result_path, model_path):
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=0, min_lr=0.00005)
    early_stopping = keras.callbacks.EarlyStopping(patience=3)
    file_path = '/Users/kefei/Documents/results/win60/past/VAE/kl_tuning/call_back_result_klcyclic_with_classifier.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=False,
            save_weights_only=True, mode='auto', period=1)
    call_backs = [reduce_lr,early_stopping, checkpoint]
    past_ae = None
    learning_rate = 0.0001
    hidden_dim = 128
    W_regularizer_val = 0.0001
    dropout_rate = 0.1
    classifier_dense_dim = 16

    params = {'hidden_dim': hidden_dim,'W_regularizer_val': W_regularizer_val,
            'dropout_rate': dropout_rate, 'classifier_dense_dim': classifier_dense_dim,
             'input_dim':51, 'joint_dropout':False,'GPU':True}

    print('params=',params)
    X_train, y_train, X_test, y_test = load_data()
    print('finish loading data')
    past_ae = baseline_LSTM(**params)

    optimizer=optimizers.Adam(learning_rate)
    past_ae.model.compile(optimizer,loss='binary_crossentropy')
    print(past_ae.model.summary())


    history_callback = past_ae.model.fit(
                                x=X_train,
                                y=y_train,
                                epochs=10,
                                verbose = 2,
                                callbacks=call_backs,
                                validation_data=(X_test,y_test),
                                batch_size=128
                                ) #class_weight = class_weight
    print(past_ae.model.summary())

    pred = past_ae.model.predict(X_test)
    result_path = '/Users/kefei/Documents/results/win60/past/VAE/'
    #result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
    print(pred.shape)

    del X_train
    K.clear_session()
    pred = [np.round(x) for x in pred]
    np.save(result_path+'pred_lstm1_2.npy', pred)
    print(confusion_matrix(pred,y_test))
    print(classification_report(pred,y_test))

def fitness(hyperparams, kl_type, classifier_bool, result_path, model_path): #annealtime
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

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=0, min_lr=0.00005)
    early_stopping = keras.callbacks.EarlyStopping(patience=3)
    file_path = '/Users/kefei/Documents/results/win60/past/VAE/kl_tuning/call_back_result_klcyclic_with_classifier.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=True, mode='auto', period=1)
    call_backs = [reduce_lr,early_stopping, checkpoint]
    past_ae = None
    weight = K.variable(0.0)
    if kl_type=='klcyclic':
        call_backs.append(CyclicAnnealingCallback(weight,M=num_cycles,max_kl_weight=maximum_kl_weight))
    elif kl_type=='klanneal':
        call_backs.append(AnnealingCallback(weight,annealtime,max_kl_weight=maximum_kl_weight))

    params = {'hidden_dim': hidden_dim,'latent_dim': latent_dim,'W_regularizer_val': hidden_W_regularizer,
        'dropout_rate': dropout_rate, 'classifier':classifier_bool,'classifier_dense_dim': classifier_dense_dim,
        'kl_weight':weight, 'VAE':True, 'input_dim':51, 'joint_dropout':False,
        'GPU':True,'add_losses':True}
    print('params=',params)
    X_train, y_train, X_test, y_test = load_data()
    print('finish loading data')
    past_ae = past_LSTM_autoencoder(**params)
    beta = K.variable(1.) #initialised at 100, but decreased over time with decay rate 0.5. set min_beta = 0.01
        #call_backs.append(BetaCallback(beta))


    kl_history = KLLossHistory()
    recon_history = MSELossHistory()
    call_backs.append(kl_history)
    call_backs.append(recon_history)
    optimizer=optimizers.Adam(learning_rate)
    if classifier_bool:
        past_ae.model.compile(optimizer,loss={'reconstructed_poses':past_ae.vae_loss,'classified_label':'binary_crossentropy'},
                        loss_weights = {'reconstructed_poses':beta,'classified_label':0.01},
                        metrics = {'classified_label':['accuracy']}
                        ) #with classifier
        past_ae.recon_model.compile(optimizer,loss={'reconstructed_poses':past_ae.vae_loss,'classified_label':'binary_crossentropy'},
                        loss_weights = {'reconstructed_poses':beta,'classified_label':0.01},
                        metrics = {'classified_label':['accuracy']}
                        )
    else:
        past_ae.model.compile(optimizer,loss={'reconstructed_poses':past_ae.vae_loss},
                            loss_weights = {'reconstructed_poses':beta}
                            ) #without classifier
    add_losses = True
    if add_losses:
        past_ae.model.metrics_tensors.append(past_ae.kl_loss)
        past_ae.model.metrics_names.append("kl_loss")
        past_ae.model.metrics_tensors.append(past_ae.recon_loss)
        past_ae.model.metrics_names.append("mse_loss")
        #custom_validation_loss = CustomValidationLoss(past_ae.model, beta)
        #call_backs.append(custom_validation_loss)
        class_weight = {0: 1.,1: 70} #times 5 as fall is more important.
        #the ratio is 1-r/r = 13.888283251805264
    '''
    if classifier_bool:
        history_callback = past_ae.model.fit(
                            x=X_train,
                            y={'reconstructed_poses': X_train[:,::-1,:], 'classified_label':y_train},
                            epochs=30,
                            verbose = 2,
                            callbacks=call_backs,
                            validation_data=(X_test,{'reconstructed_poses':X_test[:,::-1,:],'classified_label':y_test}),
                            batch_size=128
                            ) #class_weight = class_weight

    else:
        history_callback = past_ae.model.fit(
                                x=X_train,
                                y={'reconstructed_poses': X_train[:,::-1,:]},
                                epochs=40,
                                verbose = 2,
                                callbacks=call_backs,
                                validation_data=(X_test,{'reconstructed_poses':X_test[:,::-1,:]}),
                                batch_size=128
                                ) #class_weight = cl


    print(history_callback.history["val_loss"][-1])
    loss = history_callback.history["loss"]
    val_loss = history_callback.history["val_loss"]
    kl_loss = history_callback.history["kl_loss"]
    val_kl_loss = history_callback.history["val_kl_loss"]
    mse_loss = history_callback.history["mse_loss"]
    val_mse_loss = history_callback.history["val_mse_loss"]

    np.save(result_path + 'loss.npy',loss)
    np.save(result_path + 'val_loss.npy',val_loss)
    np.save(result_path + 'kl_loss.npy',kl_loss)
    np.save(result_path + 'val_kl_loss.npy',val_kl_loss)
    np.save(result_path + 'mse_loss.npy',mse_loss)
    np.save(result_path + 'val_mse_loss.npy',val_mse_loss)

    '''
    #past_ae.enc_model.save_weights(model_path + 'enc_model.h5')
    past_ae.enc_model.load_weights(model_path + 'enc_model.h5')

    '''
    if classifier_bool:
        past_ae.recon_model.save_weights(model_path + 'recon_model.h5')
        past_ae.classifier_model.save_weights(model_path + 'classifier_model.h5')
        recon = past_ae.recon_model.predict(X_test)
    else:
        past_ae.model.save_weights(model_path + 'recon_model.h5')
        recon = past_ae.model.predict(X_test)
    #save the posterior parameters
    '''
    past_ae.classifier_model.load_weights(model_path + 'classifier_model.h5')
    past_ae.recon_model.load_weights(model_path + 'recon_model.h5')
    recon = past_ae.recon_model.predict(X_test)

    z_mean, z_log_var = past_ae.enc_model.predict(X_test)
    np.save(result_path+'z_mean_test.npy',z_mean)
    np.save(result_path+'z_log_var_test.npy',z_log_var)

    z_mean, z_log_var = past_ae.enc_model.predict(X_train)
    np.save(result_path+'z_mean_train.npy',z_mean)
    np.save(result_path+'z_log_var_train.npy',z_log_var)

    #save reconstruction_loss
    used = np.sign(np.max(abs(X_test),axis=-1))
    length = np.sum(used, axis=1)

    MSE_TS = np.mean(np.square(X_test[length>=30][:,::-1,:]-recon[length>=30]), axis=(0,2))
    np.save(result_path + 'recon_loss_overtime.npy', MSE_TS)


    recon_non_fall = recon[y_test==0][:100]
    recon_fall = recon[y_test==1][:100]
    np.save(result_path + 'recon_non_fall.npy', recon_non_fall)
    np.save(result_path + 'recon_fall.npy', recon_fall)

    if classifier_bool:
        pred_scores = past_ae.classifier_model.predict(X_test)
        pred = [np.round(x) for x in pred_scores]
        np.save(result_path + 'pred_scores.npy', recon_non_fall)
        np.save(result_path + 'pred.npy', recon_non_fall)
        K.clear_session()
        #false_positives, false_positives_recon, false_positives_label = return_false_positives(X_test, recon, y_test, y_test_original,pred)
        #false_negatives, false_negatives_recon = return_false_negatives(X_test, recon, y_test, pred)
        diff = []
        for i in range(len(pred)):
            diff_i = int(pred[i]) - int(y_test[i])
            diff.append(diff_i)

        print('diff=',diff[:10])
        diff = np.array(diff)
        false_positives = X_test[diff>0]
        false_positives_recon = recon[diff>0]
        false_positives_label = y_test_original[diff>0]
        false_negatives = X_test[diff<0]
        false_negatives_recon = recon[diff<0]
        print('false positives=', false_positives.shape[0])
        print('false negatives=', false_negatives.shape[0])
        #print('pred=',pred)
        np.save(result_path + 'false_positives.npy',false_positives)
        np.save(result_path + 'false_positives_recon.npy',false_positives_recon)
        np.save(result_path + 'false_positives_label.npy',false_positives_label)
        np.save(result_path + 'false_negatives.npy',false_negatives)
        np.save(result_path + 'false_negatives_recon.npy',false_negatives_recon)

        print(confusion_matrix(pred,y_test))
        print(classification_report(pred,y_test))

    K.clear_session()

    return None

def collect_result(kl_type, classifier_bool):

    if kl_type=='klanneal':
        if classifier_bool:
            hyperparameters = [1280, 64, 0.001, 0.01, 96, 0.0005, 512, 0.0001]
            model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/with_classifier/'%kl_type
            result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
        else:
            hyperparameters = [1024, 128, 0.004875173292623599, 0.04085806061902107, 32, 0.00025378664002876164, 861, 0.003950403393695316]
            model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/without_classifier/'%kl_type
            result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type

    elif kl_type=='klcyclic':
        if classifier_bool:
            hyperparameters = [768, 128, 0.001, 0.01, 128, 0.0005, 2, 0.0001]
            model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/with_classifier/'%kl_type
            result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
        else:
            hyperparameters = [768, 256, 0.001, 0.12, 32, 0.0005, 10, 0.0001]
            model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/without_classifier/'%kl_type
            result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type

    fitness(hyperparameters, kl_type, classifier_bool, result_path, model_path)

def collect_result_baseline():
    fitness_baseline( None, None)

#already know for klanneal with classifier, just without reconstructed poses
collect_result_baseline()
#collect_result(kl_type='klanneal', classifier_bool=True)
#collect_result(kl_type='klanneal', classifier_bool=False)
#collect_result(kl_type='klcyclic', classifier_bool=True)
#collect_result(kl_type='klcyclic', classifier_bool=False)
