
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Dense, Activation
from keras.layers import Input
from keras.models import Model
from keras.layers import Dropout
from keras import backend as K
from keras import callbacks,regularizers, layers, losses, models, optimizers
from sklearn.model_selection import train_test_split
import keras
import random
import numpy as np
import os
from keras import backend as K
os.environ['KERAS_BACKEND'] = 'tensorflow'
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
import functions
from functions import *
from keras.models import Sequential

def remove_joints(data, max_num_remove=2,num_joints=17,dim=2, noise_density=0.05):
    incomplete_data = data.reshape(data.shape[0],data.shape[1], dim, num_joints)
    for i in range(data.shape[0]):
        #pick a random number of joints to remove
        r = np.random.randint(max_num_remove+1)
        joints_to_remove = np.random.randint(num_joints, size=r) #pick random joint to remove for each data
        for x in joints_to_remove:
            incomplete_data[i,:,:,x] = 0.0 #or let be a random number??
            #if noise_density>0:
            #    incomplete_data[i] = noisy_joints(data, num_noisy_joints=2, num_joints=17,density=0.05)
    incomplete_data = incomplete_data.reshape(data.shape[0],data.shape[1], dim*num_joints)
    return incomplete_data


class Past_DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 #train_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_60_2_untrimmed.npy',
                 train_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_train.npy',
                 train=True,
                 batch_size=128,
                 shuffle=True,
                 keep_3D=True,
                 joint_dropout=False,
                 classifier = False,
                 trim=True,
                 **kwargs):

        self.train_data_dir = train_data_dir
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.keep_3D = keep_3D
        self.joint_dropout = joint_dropout
        self.classifier = classifier
        self.X_train= self.load_data()
        if self.classifier:
            self.Y_train = self.load_label()
            if trim:
                self.X_train, self.Y_train = trim_dataset(self.X_train, self.Y_train,threshold=10)
        self.binary = True
        self.on_epoch_end()

    def __len__(self):
        return int((self.X_train.shape[0]) / self.batch_size)

    def __getitem__(self, idx):
        data = self.X_train[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.asarray(data, dtype=np.float32)
        X_target = np.flip(X, axis=1)

        if self.joint_dropout:
            X = remove_joints(X) #generated for each batch to prevent OOM

        if self.classifier: #if classifier is trained simultaneously as encoder
            y_target = self.Y_train[idx * self.batch_size:(idx + 1) * self.batch_size]
            if self.binary:
                y_target = [1 if x==43 else 0 for x in y_target] #convert
            return X,[X_target,y_target]
        else:
            return X, X_target

    def on_epoch_end(self):
        if self.shuffle:
            if self.classifier:
                ind = list(range(self.X_train.shape[0]))
                random_shuffle(ind)
                self.X_train = self.X_train[ind]
                self.Y_train = self.Y_train[ind]
            else:
                random.shuffle(self.X_train)

    def load_data(self):
        X_train = np.load(self.train_data_dir )
        #X_train=X_train[:1000]
        print('finish loading training set!')
        if not self.keep_3D:
            X_train = crop_2D(X_train)
        return X_train

    def load_label(self):
        Y_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/y_train.npy')
        Y_train = np.tile(Y_train, 24)
        return Y_train

class baseline_LSTM(object):
    def __init__(self, input_dim=75, hidden_dim=1024, concat_h=True, W_regularizer_val=0.01,
                 dropout_rate = 0.1,  GPU=True, classifier_dense_dim = 32,
                 joint_dropout=False, **kwargs):
        self.valid_data_dir = None
        self.GPU= GPU
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.concat_h = concat_h
        self.joint_dropout = joint_dropout
        self.W_regularizer_val = W_regularizer_val
        self.classifier_dense_dim = classifier_dense_dim
        self.dropout_rate = dropout_rate
        self.layers = 1
        self.build_model()


    def build_model(self):
        original_poses = Input(shape=(None,self.input_dim),name='observed_poses')
        poses = layers.Lambda(lambda x: K.dropout(x, level=self.dropout_rate),name='dropout_poses')(original_poses) #this is just a dropout layers

        initializer = keras.initializers.glorot_normal(seed=None)
        bs_model = Sequential()
        if self.layers ==1:
            bs_model.add(CuDNNLSTM(self.hidden_dim, return_state=False, kernel_regularizer=regularizers.l2(self.W_regularizer_val),
                    kernel_initializer=initializer))
        else:

            bs_model.add(CuDNNLSTM(self.hidden_dim,return_sequences=True,kernel_regularizer=regularizers.l2(self.W_regularizer_val),
                    kernel_initializer=initializer, batch_input_shape=(None, 30, self.input_dim)))
            bs_model.add(CuDNNLSTM(self.hidden_dim,return_state=False,return_sequences=False, kernel_regularizer=regularizers.l2(self.W_regularizer_val),
                    kernel_initializer=initializer))
            '''
            outputs = LSTM(self.hidden_dim,return_sequence=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val),
                    kernel_initializer=initializer)(original_poses)
            print('SHAPE=',outputs)
            outputs2 = LSTM(self.hidden_dim,return_state=False,return_sequence=False,kernel_regularizer=regularizers.l2(self.W_regularizer_val),
                    kernel_initializer=initializer)(outputs)

            dense_outputs = Dense(self.classifier_dense_dim, input_dim=self.hidden_dim, kernel_initializer='normal', activation='relu')(outputs2)
            pred = Dense(1, kernel_initializer='normal', activation='sigmoid',name='pred')(dense_outputs)
            '''
        bs_model.add(Dense(self.classifier_dense_dim, input_dim=self.hidden_dim, kernel_initializer='normal', activation='relu'))
        bs_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid',name='pred'))
        pred = bs_model(poses)
        self.model = Model(original_poses, pred)



class past_LSTM_autoencoder(object):
    def __init__(self, input_dim=75, hidden_dim=1024, latent_dim=128,concat_h=True, W_regularizer_val=0.01, VAE=True,
                  lr=0.0005, epochs=10, dropout_rate=0.1, kl_weight=0.01,GPU=True,
                 joint_dropout=False, add_losses=False, classifier = False, classifier_dense_dim = 64, **kwargs):
        self.valid_data_dir = None
        self.VAE = VAE
        self.GPU= GPU
        self.kl_weight = kl_weight
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim =latent_dim
        self.concat_h = concat_h
        self.joint_dropout = joint_dropout
        self.epochs = epochs

        self.W_regularizer_val = W_regularizer_val
        self.dropout_rate = dropout_rate
        self.add_losses = add_losses
        self.classifier = classifier
        self.classifier_dense_dim = classifier_dense_dim
        self.classifier_network = self.build_classifier()
        self.build_model()

    def build_classifier(self):
        model = Sequential()
        model.add(Dense(self.classifier_dense_dim, input_dim=self.latent_dim *2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid',name='classfied_label'))
        return model

    def build_model(self):
        initializer = keras.initializers.glorot_normal(seed=None)
        # else let be'random_uniform'
        original_poses = Input(shape=(None, self.input_dim),name='observed_poses')
        poses = layers.Lambda(lambda x: K.dropout(x, level=self.dropout_rate),name='dropout_poses')(original_poses) #this is just a dropout layers

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
        en_out = layers.Lambda(lambda x: x[:,None, :],name='encoder_h_T')(state_h)  # reshape to (?,1,hidden_dim)
        h_T = layers.Lambda(lambda x: x, name='h_T')(state_h)
        c_T = layers.Lambda(lambda x: x, name='c_T')(state_c)

        if self.VAE:
            print('VAE approach indeed')
            z_mean = Dense(self.latent_dim, name='z_mean')(h_T)
            z_log_var = Dense(self.latent_dim, name='z_log_var')(h_T)
            z = layers.Lambda(sampling, name='z', output_shape=(self.latent_dim,))([z_mean, z_log_var])

            if self.latent_dim < self.hidden_dim:
                sampled_representation = Dense(self.hidden_dim, name='sampled_representation')(z)
            else:
                sampled_representation = h_T
            initial_state = [sampled_representation,c_T]
        else:
            print('this is AE approach')
            #initial_state = encoder_states
            initial_state = [h_T,c_T]

        decoder_inputs = layers.Lambda(lambda x: x[:, 1:, :],name='observed_poses_shifted')(poses)
        decoder_outputs = decoder(decoder_inputs, initial_state=initial_state)

        if self.concat_h:
            ts_dense_inputs = layers.concatenate([en_out, decoder_outputs], 1)
        else:
            ts_dense_inputs = decoder_outputs

        ts_dense = layers.TimeDistributed(layers.Dense(self.input_dim, kernel_initializer=initializer))
        #ts_dense_inputs = Dropout(self.dropout_rate)(ts_dense_inputs)
        ts_dense_inputs = Activation('relu')(ts_dense_inputs)
        #recon = ts_dense(ts_dense_inputs)
        recon = layers.Lambda(lambda x:ts_dense(x),name='reconstructed_poses')(ts_dense_inputs)

        if self.VAE:
            if self.kl_weight:
                self.vae_loss = compute_vae_loss(z_log_var,z_mean,self.kl_weight)
            else:
                print('Error: need to input a value for kl_weight in VAE')
        else:
            self.vae_loss = None

        if self.add_losses:
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = -0.5 * K.mean(K.sum(kl_loss, axis=None))
            #kl_loss = K.mean(- 0.5 * 1/self.latent_dim * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
            reconstruction_loss = mse(K.flatten(original_poses[:,::-1,:]), K.flatten(recon))
            self.recon_loss = reconstruction_loss
            self.kl_loss = kl_loss

        if self.classifier: #only if trained simulatneously as autoencoder
            if self.VAE:
                latent_features = layers.concatenate([z_mean, z_log_var], 1, name='latent_features')
                classify_hidden = Dense(self.classifier_dense_dim, input_dim=self.latent_dim *2, kernel_initializer='normal',
                            activation='relu')(latent_features)
            else:
                classify_hidden = Dense(self.classifier_dense_dim, input_dim=self.hidden_dim, kernel_initializer='normal',
                            activation='relu')(h_T)

            pred_label = Dense(1, kernel_initializer='normal', activation='sigmoid',name='classified_label')(classify_hidden)
            #pred_label = self.classifier_network(latent_features) #use z_mean and z_log_var as the feature vecor?
            self.classifier_model = Model(original_poses,pred_label) #the loss will be weighted and calculated outside
            self.recon_model = Model(original_poses, recon)
            self.model = Model(original_poses,[recon, pred_label])
        else:
            self.model = Model(original_poses, recon)

        if self.VAE:
            self.enc_model_pre_inference = Model(original_poses, en_out)
            self.enc_model = Model(original_poses, [z_mean, z_log_var])
            self.z_model = Model(original_poses, z)
        else:
            self.enc_model = Model(original_poses, en_out)
