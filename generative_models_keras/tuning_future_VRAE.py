from sklearn.model_selection import StratifiedKFold
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
from VRAE_model import *
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
class BetaCallback(keras.callbacks.Callback):
    def __init__(self, beta):
        self.beta = beta
        self.min_beta = 0.01
        self.decay = 0.5

    def on_epoch_end (self, epoch, logs={}):
        new_beta = max(K.get_value(self.beta) * self.decay, self.min_beta)
        K.set_value(self.beta, new_beta)
        print ("Current beta is " + str(K.get_value(self.beta)))


def load_data(keep_3D = True):  # temporary.
    X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_aug.npy')
    Y_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_train_60_2_trimmed25.npy')
    Y_train_r = np.repeat(Y_train,4)
    Y_train = np.concatenate([Y_train, Y_train_r])
    print(X_train.shape)
    print(Y_train.shape)
    X_train = X_train[:650000]
    Y_train = Y_train[:650000]

    Y_train = np.asarray([1 if x==43 else 0 for x in Y_train])
    return X_train, Y_train

X_train, y_train = load_data()
kl_type = 'kl_anneal'
#model parameter
input_dim = 51
#hidden_dims = Integer(low=512, high=1024, name='hidden_dim')
hidden_dims = Categorical([256,512,768,1024,1280],name='hidden_dim')
#latent_dims = Integer(low=128, high=256, name='latent_dim')
latent_dims = Categorical([64,128,192,256],name='latent_dim')
W_regularizer_vals =Real(low=0.001, high=0.01, name='hidden_W_regularizer')
past_dropout_rates = Real(low=0.01, high=0.12, name='past_dropout_rate')
future_dropout_rates = Real(low=0.01, high=0.12, name='future_dropout_rate')
classifier_dense_dims = Categorical([32,64,96,128,256],name='classifier_dense_dim')
learning_rates = Real(low=0.00001, high=0.0005, name='learning_rate')
#teacher_force_bool = Categorical([True,False],name='teacher_force')
lstm_layers = Integer(low=1, high=3, name='lstm_layer')
#residual_bool = Categorical([True,False],name='residual')
#opt_types = Categorical(['adam','sgd','rmsprop'],name='opt_type')

param_space = [hidden_dims,latent_dims,W_regularizer_vals,past_dropout_rates, future_dropout_rates,
    classifier_dense_dims, learning_rates,lstm_layers]#,residual_bool]
#hyper Parameters

def conditional_parameters(param_space ,kl_type):
    kl_weights =Real(low=0.001, high=0.01, name='kl_weight') #only true when constant_kl
    Ms = Integer(low=2, high=10, name='num_cycles') #only true when cyclic_kl
    annealtimes = Integer(low=512, high=1024, name='annealtime') #only true when kl_anneal
    max_kl_weights = Real(low=0.0001, high=0.01, name='maximum_kl_weight')

    if kl_type== 'kl_anneal':
        param_space.append(annealtimes)
        param_space.append(max_kl_weights)

    elif kl_type=='kl_cyclic':
        param_space.append(Ms)
        param_space.append(max_kl_weights)

    elif kl_type == 'kl_constant':
        param_space.append(kl_weights)
    return param_space

param_space = conditional_parameters(param_space, kl_type='kl_anneal') #kl_anneal

#for simplicty sake, just do annealing first

@use_named_args(dimensions=param_space)
def fitness(hidden_dim, latent_dim, hidden_W_regularizer, past_dropout_rate, future_dropout_rate,
        classifier_dense_dim, learning_rate,lstm_layer,#residual,
        annealtime, maximum_kl_weight): #annealtime

    print([hidden_dim, latent_dim, hidden_W_regularizer, past_dropout_rate, future_dropout_rate,
            classifier_dense_dim, learning_rate,lstm_layer,
            annealtime, maximum_kl_weight])

    kFold = StratifiedKFold(n_splits=5)
    scores =  np.zeros(5)
    i = 0
    print('start cross validation')
    for train, val in kFold.split(X_train, y_train):
        while i<2:
            print('i=',i)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=0, min_lr=0.00005)
            early_stopping = keras.callbacks.EarlyStopping(patience=2)
            file_path = '/Users/kefei/Documents/results/win60/future/VAE/call_back_result_klanneal_classifier.hdf5'
            checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=False,
            save_weights_only=True, mode='auto', period=1)
            call_backs = [reduce_lr,early_stopping, checkpoint]

            X_train_k = X_train[train]
            y_train_k = y_train[train]
            X_valid_k = X_train[val]
            y_valid_k = y_train[val]

            K.clear_session()
            K.clear_session()
            past_ae = None

            weight = K.variable(0.0)
            #call_backs.append(CyclicAnnealingCallback(weight,M=num_cycles,max_kl_weight=maximum_kl_weight))
            call_backs.append(AnnealingCallback(weight,annealtime,max_kl_weight=maximum_kl_weight))

            params = {'hidden_dim': hidden_dim,'latent_dim': latent_dim,'W_regularizer_val': hidden_W_regularizer,
            'past_dropout_rate': past_dropout_rate,'future_dropout_rate': future_dropout_rate,
            'classifier':True,'classifier_dense_dim': classifier_dense_dim,
            'kl_weight':weight, 'VAE':True, 'input_dim':51, 'joint_dropout':False,
            'GPU':True,'add_losses':True,'teacher_force':False, 'residual':True, 'layers':lstm_layer}
            print('params=',params)

            past_ae = future_LSTM_VAE(**params)

            beta = K.variable(1.) #initialised at 100, but decreased over time with decay rate 0.5. set min_beta = 0.01
            #call_backs.append(BetaCallback(beta))
            opt_type = 'adam'
            if opt_type == 'adam':
                optimizer = optimizers.Adam(learning_rate)
            elif opt_type == 'sgd':
                optimizer = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            elif opt_type == 'rmsprop':
                optimizer = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)

            kl_history = KLLossHistory()
            recon_history = MSELossHistory()
            call_backs.append(kl_history)
            call_backs.append(recon_history)
            past_ae.model.compile(optimizer,loss={'future_pred':past_ae.vae_loss,'classified_label':'binary_crossentropy'},
                            loss_weights = {'future_pred':beta,'classified_label':0.01},
                            metrics = {'classified_label':['accuracy']}
                            ) #with classifier
            print('model compiled')
            #print(past_ae.model.summary())
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
            history_callback = past_ae.model.fit(
                                x=[X_train_k[:,:15,:], X_train_k[:,14:,:]],#past pose include current pose
                                y={'future_pred': X_train_k[:,15:,:], 'classified_label':y_train_k},
                                epochs=15,
                                verbose = 2,
                                callbacks=call_backs,
                                validation_data=([X_valid_k[:,:15,:], X_valid_k[:,14:,:]],
                                {'future_pred':X_valid_k[:,15:,:],'classified_label':y_valid_k}),
                                batch_size=128
                            ) #class_weight = class_weight

            del X_train_k
            del X_valid_k
            del y_train_k
            del y_valid_k

            scores[i] = history_callback.history["val_loss"][-1]
            print(history_callback.history["val_loss"][-1])
            i+=1

    #pred = past_ae.classifier_model.predict(X_train[:100])
    #print(pred)
    print('scores=',scores)
    print('average score=', scores.mean()*(5/2))
    K.clear_session()
    tf.reset_default_graph()

    mean_score = scores.mean()*(5/2)
    return mean_score

#anneal
#default_parameters.append(4)
#default_parameters.append(0.0001)
default_parameters =   [1024, 128, 0.0028269470115122606, 0.07962675839794084, 0.04781778231696922, 96,
0.0004816813096902684, 2, 875, 0.0022991914320305137]

gp_result = gp_minimize(func=fitness,
                            dimensions=param_space,
                            n_calls=25,
                            verbose = True,
                            noise= 0.01,
                            n_jobs=-1,
                            kappa = 1.96,
                            x0=default_parameters)

#hidden_dim, latent_dim, hidden_W_regularizer, dropout_rate,
#classifier_dense_dim, learning_rate, annealtime, maximum_kl_weight

dump(gp_result, '/Users/kefei/Documents/results/win60/future/VAE/result_klcyclic_with_classifier.pkl', store_objective=False)
print(gp_result.x)
#print(gp_result.yi)
from skopt.plots import plot_convergence

plot_convergence(gp_result)
