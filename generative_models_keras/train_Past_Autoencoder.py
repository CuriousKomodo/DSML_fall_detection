#$ -l gpu=2
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
from keras.utils.vis_utils import plot_model
from keras.callbacks import LambdaCallback
from Past_Autoencoder_model import *
import pydotplus
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from create_past_train_valid_split import *


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


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

# total number of epochs
n_epochs = 20
# The number of epochs at which KL loss should be included
# number of epochs over which KL scaling is increased from 0 to 1
kl_annealtimes = [2,5,10]

class AnnealingCallback(keras.callbacks.Callback):
    def __init__(self, weight,kl_annealtime):
        self.klstart = 0
        self.kl_weight = weight
        self.kl_annealtime = kl_annealtime
    def on_epoch_end (self, epoch, logs={}):
        if epoch > self.klstart :#grows linearly towards one
            new_weight = min(K.get_value(self.kl_weight) + (1/ self.kl_annealtime), 1.)
            print(new_weight)
            K.set_value(self.kl_weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.kl_weight)))

class CyclicAnnealingCallback(keras.callbacks.Callback):
    def __init__(self, weight,M):
        self.kl_weight = weight
        self.R = 0.5 #keep this
        self.M = M
        self.T = 20

    def on_epoch_end (self, epoch, logs={}):
        tau = np.mod(epoch, int(self.T/self.M))*1.0 /int(self.T/self.M)
        if tau>=self.R:
            new_weight = 1.0/5000 #capped at 0.0002, because posterior collapose happens!
        else:
            new_weight = tau/5000

        K.set_value(self.kl_weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.kl_weight)))

#latent dim = [128, 256, 512, 1024]
latent_dims= [64, 128, 256, 512] #128
hidden_dims = [256,512,1024]
#latent_hidden_dims = [[64,512],[128,1024],[256,512]]
latent_hidden_dims = [[128,1024]]
dropout_rates=[0.05, 0.1, 0.2, 0.4]
dropout_rates = [0.1]
learning_rates = [0.0001, 0.00005, 0.00002, 0.00001]
kl_weights = [1, 0.01, 0.0001]
M_list = [2,4,8] #defines the number of cycles
kl_annealing = False
kl_cyclic = True
window=60
t =  1 #attempt
repeats = 1
VAE=True
keep_3D= True
joint_dropout=False
add_losses = True

if VAE:
    result_path = '/Users/kefei/Documents/results/win%s/past/VAE/kl_tuning/'%window
    model_path = '/Users/kefei/Documents/mm_fall_detection/models/win%s/past/VAE/'%window
else:
    result_path = '/Users/kefei/Documents/results/win%s/past/AE/hyperparameter_tuning/'%window
    model_path = '/Users/kefei/Documents/mm_fall_detection/models/win%s/past/AE/'%window

print('will save results at ', result_path)

#maybe keep a constant dropout rate, learning rate and kl_weight.
val_loss_matrix = np.zeros((repeats, len(kl_weights), len(latent_hidden_dims)))
train_loss_matrix = np.zeros((repeats, len(kl_weights), len(latent_hidden_dims)))
past_ae = None
print('val_loss_matrix shape=', val_loss_matrix.shape)

for r in range(repeats):
    if r>0:
        train_valid_split_past()
        print('complete split')
    valid_data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_valid.npy')
    valid_target = valid_data[:,::-1,:]
    print('finish loading valid dataset')

    #for i in range(len(latent_dims)):
    #    for j in range(len(hidden_dims)):
    for i in np.arange(0,1):
        for j in range(len(latent_hidden_dims)):
            print('repeat=',r)
            weight = None
            past_ae = None
            M = M_list[i]
            #kl_annealtime = kl_annealtimes[i]
            #dropout_rate = dropout_rates[i]
            #latent_dim = latent_dims[i]
            latent_dim = latent_hidden_dims[j][0]
            hidden_dim = latent_hidden_dims[j][1]
            #hidden_dim = hidden_dims[j]
            dropout_rate = 0.1

            if kl_annealing or kl_cyclic:
                weight = K.variable(0.)
            else:
                weight = kl_weights[i]


            params = {'hidden_dim': hidden_dim,'latent_dim': latent_dim, 'dropout_rate': dropout_rate,
            'kl_weight':weight, 'VAE':VAE, 'input_dim':51, 'joint_dropout':False,'GPU':True,'add_losses':add_losses}

            print(params)

            #params_valid = {'train': False}
            optimizer=optimizers.Adam(0.0001)
            #optimizer = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
            #optimizer = optimizers.RMSprop(lr=0.0005,decay=0.5)
            past_ae = past_LSTM_autoencoder(**params)
            print('finish setting the model')
            if past_ae.vae_loss:
                print('VAE approach')
                #if kl_annealing:
                past_ae.model.compile(optimizer ,loss=past_ae.vae_loss)
                #else:
                batch_vae_losses = []
                batch_kl_losses = []

                if add_losses:
                    past_ae.model.metrics_tensors.append(past_ae.kl_loss)
                    past_ae.model.metrics_names.append("kl_loss")
                    past_ae.model.metrics_tensors.append(past_ae.recon_loss)
                    past_ae.model.metrics_names.append("mse_loss")
            else:
                past_ae.model.compile(optimizer ,loss=losses.mean_squared_error)
                batch_mse_losses = []
            print('finish compile')
            #print(past_ae.model.summary())
            elbo_history = LossHistory()
            kl_history = KLLossHistory()
            recon_history = MSELossHistory()
            #plot_model(past_ae.model, to_file=result_path+'past_vae_plot_%s_%s.png'%(hidden_dim,latent_dim), show_shapes=True, show_layer_names=True)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=0, min_lr=0.00005)
            #batch_call_loss = LambdaCallback(on_batch_end= lambda batchs, logs: batch_vae_losses.append(past_ae.vae_loss))

            call_backs = [reduce_lr, elbo_history, kl_history, recon_history]
            earlystopping = False
            if earlystopping:
                call_backs.append(keras.callbacks.EarlyStopping(patience=2))
            if kl_annealing:
                print('applying annealing, annleaingtime=', kl_annealtime)
                call_backs.append(AnnealingCallback(weight, kl_annealtime))
            elif kl_cyclic:
                call_backs.append(CyclicAnnealingCallback(weight,M))


            history_callback = past_ae.model.fit_generator(generator=Past_DataGenerator(),verbose=1,
                                                        epochs=20, callbacks=call_backs,validation_data=(valid_data,valid_target)
                                                        )#
            K.clear_session()
            np.save(result_path+'batch_train_elbo_%s_%s_%s_klcyclic_%s.npy'%(hidden_dim,latent_dim,r,i), elbo_history.losses)
            np.save(result_path+'batch_train_klloss_%s_%s_%s_klcyclic_%s.npy'%(hidden_dim,latent_dim,r,i), kl_history.losses)
            np.save(result_path+'batch_train_reconloss_%s_%s_%s_klcyclic_%s.npy'%(hidden_dim,latent_dim,r,i), recon_history.losses)
            loss_history = history_callback.history["loss"]
            val_kl_loss_history = history_callback.history["val_kl_loss"]
            val_recon_loss_history = history_callback.history["val_mse_loss"]
            val_loss_history = history_callback.history['val_loss']
            np.save(result_path+'epoch_val_elbo_%s_%s_%s_klcyclic_%s.npy'%(hidden_dim,latent_dim,r,i), val_loss_history)
            np.save(result_path+'epoch_val_kl_%s_%s_%s_klcyclic_%s.npy'%(hidden_dim,latent_dim,r,i), val_kl_loss_history)
            np.save(result_path+'epoch_val_recon_%s_%s_%s_klcyclic_%s.npy'%(hidden_dim,latent_dim,r,i), val_recon_loss_history)
            #print(history_callback.history["kl_loss"])
            val_loss_matrix[r,i,j] = val_loss_history[-1] #final validation loss.
            train_loss_matrix[r,i,j]=loss_history[-1]


print('save losses')
print('train losses=',train_loss_matrix)
print('val losses=',val_loss_matrix)
np.save(result_path+'train_loss_history_cyclic%s.npy'%t, train_loss_matrix)
np.save(result_path+'val_loss_history_cyclic%s.npy'%t, val_loss_matrix)



#


#
