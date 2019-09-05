#This script loads the pre-trained LSTM-AE and missing_joints, then train them together.
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


class Past_DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 #train_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_60_2_untrimmed.npy',
                 train_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/X_train_60_2_untrimmed.npy',
                 missing_joints = True,
                 train=True,
                 batch_size=128,
                 shuffle=True,
                 keep_3D=True,
                 **kwargs):

        self.missing_joints = missing_joints
        self.train_data_dir = train_data_dir
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.keep_3D = keep_3D
        self.X_train= self.load_data()
        self.on_epoch_end()

    def __len__(self):
        return int((self.X_train.shape[0]) / self.batch_size)

    def __getitem__(self, idx):
        X = self.X_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.asarray(X, dtype=np.float32)
        X = 0.8 * X + 0.1
        Y = np.flip(X, axis=1)

        if self.missing_joints:
            X = X.reshape(X.shape[0]*X.shape[1], X.shape[-1], order='F') #unroll from the first dimension.
            X = remove_joints(X, max_num_remove=5) #Do it by batch, to cope with memory.
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.X_train)

    def load_data(self):
        X_train = np.load(self.train_data_dir)
        print('finish loading training set!')
        if not self.keep_3D:
            X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],3,int(X_train.shape[-1]/3))
            X_train = X_train[:,:,:2,:] #only extract x,y cooridnates
            X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],2*X_train.shape[-1])
        return X_train


#Define a model that: takes input from generator, uses pre-trained DAE, and trian DAE and past AE together.

class past_model(object):

    def __init__(self, input_dim=75, hidden_dim=1024, latent_dim=128,concat_h=True, W_regularizer_val=0.01, VAE=True, DAE_dir=None,
                 conditional=False, lr=0.0005, epochs=10, dropout_rate=0.1, GPU=True,**kwargs):
        self.valid_data_dir = None
        self.VAE = VAE
        self.GPU= GPU
        self.kl_weight = 0.01
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim =latent_dim
        self.concat_h = concat_h
        self.lr = lr
        self.epochs = epochs
        self.conditional = conditional
        self.W_regularizer_val = W_regularizer_val
        self.dropout_rate = dropout_rate
        self.DAE_dir = DAE_dir
        self.DAE_hidden_dim = DAE_hidden_dim
        self.DAE_model = load_pretrained_DAE_model()
        self.past_VAE_model = load_VAE_model()
        self.build_model()

        #build a model such that: it takes corrupted data as input:
        def load_pretrained_DAE_model(self):
            dae_weight_dir = self.DAE_dir
            DAE_params = {'hidden_dim': self.DAE_hidden_dim, 'input_dim':self.input_dim}
            dae = DAE(**DAE_params)
            model = dae.model
            model.load_weights(self.DAE_dir)

        def build_past_model(self):
            past_params = {'hidden_dim': self.hidden_dim, 'input_dim':self.input_dim, ...}
            self.past_VAE_model = past_LSTM_autoencoder(**past_params)

        def build_model(self):
            initializer = keras.initializers.glorot_normal(seed=None)
            corrupted_poses = Input(shape=(None, self.input_dim)) #sequential. 
            recon_poses = layers.TimeDistributed(self.DAE_model)(corrupted_poses)
            print(recon_poses.shape)
            poses = layers.Lambda(lambda x: K.dropout(x, level=self.dropout_rate))(poses)

            if self.GPU:
                encoder = CuDNNLSTM(self.hidden_dim, return_state=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val), kernel_initializer=initializer)
                decoder = CuDNNLSTM(self.hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val),
                       go_backwards=True, kernel_initializer=initializer)
            else:
                encoder = LSTM(self.hidden_dim, return_state=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val), kernel_initializer=initializer)
                decoder = LSTM(self.hidden_dim, return_sequences=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val),
                       go_backwards=True, kernel_initializer=initializer)

            encoder_outputs, state_h, state_c = encoder(poses)
            encoder_states = [state_h, state_c]  # the last state?
            en_out = layers.Lambda(lambda x: x[:, None, :])(encoder_outputs)  # reshape to (?,1,hidden_dim)

            if self.VAE:
                print('VAE approach indeed')
                z_mean = Dense(self.latent_dim, name='z_mean')(state_h)
                z_log_var = Dense(self.latent_dim, name='z_log_var')(state_h)
                z = layers.Lambda(sampling, name='z', output_shape=(self.latent_dim,))([z_mean, z_log_var])

                if self.latent_dim < self.hidden_dim:
                    h = Dense(self.hidden_dim, name='sampled_representation')(z)
                else:
                    h = state_h
                initial_state = [h,state_c]

            else:
                initial_state = encoder_states

            decoder_inputs = layers.Lambda(lambda x: x[:, 1:, :])(poses)
            decoder_outputs = decoder(decoder_inputs, initial_state=initial_state)

            if self.concat_h:
                ts_dense_inputs = layers.concatenate([en_out, decoder_outputs], 1)
            else:
                ts_dense_inputs = decoder_outputs

            ts_dense = layers.TimeDistributed(layers.Dense(self.input_dim, kernel_initializer=initializer))
            ts_dense_inputs = Activation('relu')(ts_dense_inputs)
            pred = ts_dense(ts_dense_inputs)

            if self.VAE:
                if self.kl_weight:
                    self.vae_loss = compute_vae_loss(z_log_var,z_mean,self.kl_weight)
                else:
                    print('Error: need to input a value for kl_weight in VAE')
            else:
                self.vae_loss = None

            self.model = Model(corrupted_poses, pred)

            if self.VAE:
                self.enc_model_pre_inference = Model(corrupted_poses, en_out)
                self.enc_model = Model(corrupted_poses, [z_mean, z_log_var])
            else:
                self.enc_model = Model(corrupted_poses, en_out)
