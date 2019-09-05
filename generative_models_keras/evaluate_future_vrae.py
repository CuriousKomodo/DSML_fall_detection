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
from VRAE_model import *
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
    X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_aug.npy')
    X_train = X_train[:600000]

    Y_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_train_60_2_trimmed25.npy')
    Y_train_r = np.repeat(Y_train,4)
    Y_train = np.concatenate([Y_train, Y_train_r])

    Y_train_r = np.repeat(Y_train,4)
    Y_train = np.concatenate([Y_train, Y_train_r])
    Y_train = Y_train[:600000]
    print(X_train.shape)
    print(Y_train.shape)


    X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_aug.npy')
    Y_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed25.npy')
    X_test = X_test[:150000]

    Y_test_r = np.repeat(Y_test,4)
    Y_test = np.concatenate([Y_test, Y_test_r])
    Y_test = Y_test[:150000]

    Y_train = np.asarray([1 if x==43 else 0 for x in Y_train])
    Y_test = np.asarray([1 if x==43 else 0 for x in Y_test])
    return X_train, Y_train, X_test, Y_test


def fitness(hyperparams, residual_bool, result_path, model_path): #annealtime

    #[hidden_dim, latent_dim, hidden_W_regularizer, past_dropout_rate, future_dropout_rate,
    #        classifier_dense_dim, learning_rate,teacher_force,lstm_layer,residual,
    #        annealtime, maximum_kl_weight]

    hidden_dim = hyperparams[0]
    latent_dim = hyperparams[1]
    hidden_W_regularizer = hyperparams[2]
    past_dropout_rate = hyperparams[3]
    future_dropout_rate = hyperparams[4]
    classifier_dense_dim = hyperparams[5]
    learning_rate = hyperparams[6]
    teacher_force = hyperparams[7]
    lstm_layer = hyperparams[8]
    residual = hyperparams[9]
    annealtime = hyperparams[10]
    maximum_kl_weight = hyperparams[11]
    classifier_bool = True
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=0, min_lr=0.00005)
    early_stopping = keras.callbacks.EarlyStopping(patience=3)
    file_path = '/Users/kefei/Documents/results/win60/future/VAE/call_back_result_klanneal_with_classifier.hdf5'
    checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=False,
        save_weights_only=True, mode='auto', period=1)
    call_backs = [reduce_lr,early_stopping, checkpoint]
    future_vrae = None
    weight = K.variable(0.0)
    call_backs.append(AnnealingCallback(weight,annealtime,max_kl_weight=maximum_kl_weight))

    params = {'hidden_dim': hidden_dim,'latent_dim': latent_dim,'W_regularizer_val': hidden_W_regularizer,
            'past_dropout_rate': past_dropout_rate,'future_dropout_rate': future_dropout_rate,
            'classifier':True,'classifier_dense_dim': classifier_dense_dim,
            'kl_weight':weight, 'VAE':True, 'input_dim':51, 'joint_dropout':False,
            'GPU':True,'add_losses':True,'teacher_force':teacher_force, 'residual':residual_bool, 'layers':lstm_layer}

    print('params=',params)
    X_train, y_train, X_test, y_test = load_data()
    print('finish loading data')
    future_vrae = future_LSTM_VAE(**params)
    beta = K.variable(1.) #initialised at 100, but decreased over time with decay rate 0.5. set min_beta = 0.01
        #call_backs.append(BetaCallback(beta))
    optimizer=optimizers.Adam(learning_rate)

    kl_history = KLLossHistory()
    recon_history = MSELossHistory()
    call_backs.append(kl_history)
    call_backs.append(recon_history)
    if classifier_bool:
        future_vrae.model.compile(optimizer,loss={'future_pred':future_vrae.vae_loss,'classified_label':'binary_crossentropy'},
                        loss_weights = {'future_pred':beta,'classified_label':0.01},
                        metrics = {'classified_label':['accuracy']}
                        ) #with classifier
    else:
        future_vrae.model.compile(optimizer,loss={'future_pred':future_vrae.vae_loss},
                            loss_weights = {'future_pred':beta}
                            ) #without classifier
    add_losses = True
    if add_losses:
        future_vrae.model.metrics_tensors.append(future_vrae.kl_loss)
        future_vrae.model.metrics_names.append("kl_loss")
        future_vrae.model.metrics_tensors.append(future_vrae.recon_loss)
        future_vrae.model.metrics_names.append("mse_loss")
        #custom_validation_loss = CustomValidationLoss(future_vrae.model, beta)
        #call_backs.append(custom_validation_loss)
        class_weight = {0: 1.,1: 70} #times 5 as fall is more important.
        #the ratio is 1-r/r = 13.888283251805264
    '''
    if classifier_bool:
        history_callback = future_vrae.model.fit(
                            x=[X_train[:,:15,:], X_train[:,14:,:]],
                            y={'future_pred': X_train[:,15:,:], 'classified_label':y_train},
                            epochs=25,
                            verbose = 2,
                            callbacks=call_backs,
                            validation_data=([X_test[:,:15,:], X_test[:,14:,:]],
                            {'future_pred':X_test[:,15:,:],'classified_label':y_test}),
                            batch_size=128
                            ) #class_weight = class_weight

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

    future_vrae.enc_model.save_weights(model_path + 'enc_model.h5')
    '''
    if classifier_bool:
        #future_vrae.recon_model.save_weights(model_path + 'recon_model.h5')
        #future_vrae.classifier_model.save_weights(model_path + 'classifier_model.h5')
        #recon = future_vrae.recon_model.predict([X_test[:,:15,:], X_test[:,14:,:]])
        future_vrae.classifier_model.load_weights(model_path + 'classifier_model.h5')
        future_vrae.recon_model.load_weights(model_path + 'recon_model.h5')
    else:
        future_vrae.model.save_weights(model_path + 'recon_model.h5')
        recon = future_vrae.model.predict([X_test[:,:15,:], X_test[:,14:,:]])

    #save the posterior parameters
    '''
    z_mean, z_log_var = future_vrae.enc_model.predict(X_train[:,:15,:])
    np.save(result_path+'z_mean_train.npy',z_mean)
    np.save(result_path+'z_log_var_train.npy',z_log_var)

    z_mean, z_log_var = future_vrae.enc_model.predict(X_test[:,:15,:])
    np.save(result_path+'z_mean_test.npy',z_mean)
    np.save(result_path+'z_log_var_test.npy',z_log_var)

    #save reconstruction_loss
    MSE_TS = np.mean(np.square(X_test[:,15:,:]-recon), axis=(0,2))
    np.save(result_path + 'recon_loss_overtime.npy', MSE_TS)

    recon_non_fall = recon[y_test==0][:100]
    recon_fall = recon[y_test==1][:100]
    np.save(result_path + 'recon_non_fall.npy', recon_non_fall)
    np.save(result_path + 'recon_fall.npy', recon_fall)
    '''

    if classifier_bool:
        pred = future_vrae.classifier_model.predict([X_test[:,:15,:], X_test[:,14:,:]])
        np.save(result_path + 'pred_score.npy',pred)
        pred = [np.round(x) for x in pred]
        np.save(result_path + 'pred.npy',pred)
        #print('pred=',pred)
        print(classification_report(pred,y_test))
        print(confusion_matrix(pred,y_test))

    K.clear_session()

    return None

def collect_result(residual_bool):
    if not residual_bool:
        hyperparameters =[768, 192, 0.0028269470115122606, 0.07962675839794084, 0.04781778231696922, 96,
        0.0004816813096902684, False, 1, False, 875, 0.0022991914320305137]
        result_path = '/Users/kefei/Documents/results/win60/future/VAE/non_residual/'
        model_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/future/VAE/non_residual/'
    else:
        hyperparameters = [1024, 128, 0.004875173292623599, 0.04085806061902107, 32, 0.00025378664002876164,
        861, 0.003950403393695316]
        model_path = '/Users/kefei/Documents/results/win60/future/VAE/residual/'
        result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/future/VAE/residual/'

    fitness(hyperparameters, residual_bool, result_path, model_path)

#already know for klanneal with classifier, just without reconstructed poses

collect_result(False)
