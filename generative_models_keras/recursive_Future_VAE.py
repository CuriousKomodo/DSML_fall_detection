
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/generative_models_keras/')
from keras.layers import Dense, Activation
from keras.layers import Input
from keras.models import Model
from keras.layers import Dropout
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras import callbacks,regularizers, layers, losses, models, optimizers
import Past_Autoencoder_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow import Graph
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

from Past_Autoencoder_model import *
import random
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


# Generates inputs for encoder: which is concat of future poses with past information.
# Also generates target, which is a few poses in future.
# Set to one step for now, while h_past is computed from t>10

class future_generator(keras.utils.Sequence):

    def __init__(self,
                 train_data_dir ='/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_fall_60_2.npy' ,
                 past_hidden_dim = 1024,
                 train=True,
                 elem_dim=75,
                 batch_size=128,
                 shuffle=True,
                 n_poses_predict=1,
                 future_steps= 10,
                 past_steps = 20,
                 **kwargs):

        self.train_data_dir = train_data_dir
        self.train = train
        self.past_hidden_dim = past_hidden_dim
        self.future_steps = future_steps
        self.past_steps = past_steps
        self.elem_dim = elem_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.X_train = self.load_data()
        self.H = self.obtain_past_representation()

        self.on_epoch_end()

    def obtain_past_representation(self):
        enc_model = MyModel()
        representation = enc_model.predict(self.X_train[:,:self.past_steps,:])
        return representation

    def __len__(self):
        return int(len(self.X_train) / self.batch_size)

    def __getitem__(self, idx):
        data=self.X_train
        X = data[idx * self.batch_size:(idx + 1) * self.batch_size,:,:]  # load by batch.

        #Take the first 15 frames as past, try to predict the next 15 time_frames
        past_poses = X[:,:self.past_steps,:]
        h_past = self.H[idx * self.batch_size:(idx + 1) * self.batch_size,:]
        current_pose = X[:,self.past_steps,:]
        future_poses = X[:,self.past_steps:self.past_steps + self.future_steps,:]
        print('returned inputs for the model')

        return [future_poses, h_past, current_pose], future_poses

    def on_epoch_end(self):
        if self.train:
            random.shuffle(self.X_train)

    def load_data(self):
        X_train = np.load(self.train_data_dir)
        #X_train, X_val = train_test_split(data_list, test_size = 0.2)
        print('finish loading the dataset')
        return X_train

def compute_vae_loss(z_log_var, z_mean, kl_weight=0.01):
    """" Wrapper function which calculates auxiliary values for the complete loss function.
     Returns a *function* which calculates the complete loss given only the input and target output """

    # KL loss
    def compute_kl(z_mean, z_log_var, kl_weight):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * kl_weight * K.sum(kl_loss, axis=-1)
        return kl_loss

    kl_loss = compute_kl(z_mean, z_log_var, kl_weight)

    def vae_loss(y_true, y_pred):
        md_loss = mse(y_true, y_pred)
        model_loss = kl_loss + md_loss
        return model_loss

    return vae_loss

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    z = z_mean + K.exp(0.5 * z_log_var) * epsilon
    return z


def mse(X, Y, axis=None):
    SSE = np.square(Y - X)
    MSE = K.mean(SSE, axis=axis)
    return MSE


class recursive_future_VAE(object):

    def __init__(self, hidden_dim=1024,  # from past autoencoder
                 elem_dim=75,
                 timestep=5,
                 latent_dim=128,
                 epochs=5,
                 enc_intermediate_dim=512,
                 dec_intermediate_dim=512,
                 lr=0.0001,
                 kl_weight=0.01

                 ):
        self.hidden_dim = hidden_dim
        self.elem_dim = elem_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.enc_intermediate_dim = enc_intermediate_dim
        self.dec_intermediate_dim = dec_intermediate_dim
        self.future_steps = future_steps
        self.past_steps =  past_steps
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.build_model()
        self.past_ae_enc = MyModel()


    def build_model(self):
        future_poses = Input(shape=(self.future_steps, self.elem_dim), name='future_poses')  # fixed time step ahead.
        h_past = Input(shape=(self.hidden_dim,), name='h_past')
        current_pose = Input(shape=(self.elem_dim,), name='current_pose')
        future_poses_r = layers.Reshape((self.future_steps*self.elem_dim,), name='reshaped_future')(future_poses)

        # future encoder, but discarded in testing.
        enc_inputs = layers.concatenate([h_past, future_poses_r, current_pose],-1)  # so dimension = (?, hidden_layer + timesteps * 75 )
        enc_l1 = Dense(self.enc_intermediate_dim, activation='relu')(enc_inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(enc_l1)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(enc_l1)
        z = layers.Lambda(sampling, name='z', output_shape=(self.latent_dim,))([z_mean, z_log_var])
        # instantiate encoder model
        encoder = Model([future_poses, h_past, current_pose], [z_mean, z_log_var, z], name='encoder')
        # future decoder - training
        latent = Input(shape=(self.latent_dim,), name='z_sampling')
        latent_inputs = layers.concatenate([latent, h_past, current_pose], 1)
        dec_l1 = Dense(self.dec_intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(self.elem_dim, activation='sigmoid',name='decoder_outputs')(dec_l1)
        # instantiate decoder model
        decoder = Model([latent, h_past, current_pose], outputs, name='decoder')
        #print(decoder.summary())
        predictions = decoder([encoder([future_poses, h_past, current_pose])[2], h_past, current_pose]) #one pose

        for i in range(self.future_steps):
            h_past = self.past_ae_enc.predict(past_poses)
            predictions = decoder([encoder([future_poses, h_past, current_pose])[2], h_past, current_pose])

        self.vae = Model([future_poses, h_past, current_pose], predictions, name='vae')
        self.vae_loss = compute_vae_loss(z_log_var,z_mean,self.kl_weight)

        '''
        #future decoder - testing
        latent_test = Input(shape=(self.latent_dim,), name='z_test')
        latent_inputs_test = layers.concatenate([latent_test, h_past], 1)
        dec_l1_test = Dense(self.dec_intermediate_dim, activation='relu')(latent_inputs_test)
        outputs_test = Dense(self.elem_dim, activation='sigmoid', name='decoder_outputs_test')(dec_l1_test)
        predictions_test = decoder([latent_test, h_past], name = 'decoder_pred_test')

        self.vae_test = Model([latent_test,h_past], outputs_test, name='vae_test')
        '''
    def train(self,optimizer):
        self.vae.compile(optimizer=optimizer, loss=self.vae_loss)
        #print(self.vae.summary())
        #self.vae.compile(optimizer=optimizers.Adam(self.lr), loss=losses.mean_squared_error)

        #
    '''
    def test(data_test):
        self.vae_test.predict()
'''



#We know the future.. still in a training environment.
#now predict poses recursively for the next...
'''
n_pred = 10
n_start = 10
short_target_obs = X_test[:10,n_start,:]
long_target_obs = X_test[:10,:n_start+n_pred,:]
long_prediction = np.zeros_like(X_test[:10,: n_start + n_pred, :])
print(long_prediction.shape)
#fill the prediction sequence with past observation up to starting point.
long_prediction[:,:n_start,:] = X_test[:10,:n_start,:]

for t in np.arange(n_start,n_start+n_pred):
  past_obs = X_test[:10,:t,:]
  future_obs = X_test[:10,t:t+10,:]
  h_test = past_ae_lstm.predict(past_obs)
  pred =vae.predict([future_obs,h_test[:,0,:]])
  past_obs = np.concatenate([past_obs,pred.reshape(10,1,75)], axis=1)
  long_prediction[:,t,:] = pred
'''
