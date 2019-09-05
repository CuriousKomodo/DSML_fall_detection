
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
from recurrentshop import *
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
import functions
from functions import *

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
'''
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
'''

class future_LSTM_VAE(object):
    def __init__(self, input_dim=75, hidden_dim=1024, latent_dim=128,concat_h=True, W_regularizer_val=0.01, VAE=True,
                  lr=0.0005, past_dropout_rate=0.1, future_dropout_rate=0.1, kl_weight=0.01, GPU=True,
                 joint_dropout=False, add_losses=True, classifier = False, classifier_dense_dim = 64,
                 read_out = True, teacher_force = True,layers=3, residual=False,**kwargs):
        self.valid_data_dir = None
        self.VAE = VAE
        self.GPU= GPU
        self.kl_weight = kl_weight
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim =latent_dim
        self.concat_h = concat_h
        self.joint_dropout = joint_dropout

        self.W_regularizer_val = W_regularizer_val
        self.past_dropout_rate = past_dropout_rate
        self.future_dropout_rate = future_dropout_rate
        self.add_losses = add_losses
        self.classifier = classifier
        self.classifier_dense_dim = classifier_dense_dim
        self.classifier_network = self.build_classifier()
        self.layers = layers
        self.residual = residual
        self.teacher_force = teacher_force
        self.build_model()

    def build_classifier(self):
        model = Sequential()
        model.add(Dense(self.classifier_dense_dim, input_dim=self.latent_dim *2, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid',name='classfied_label'))
        return model

    def build_model(self):
        initializer = keras.initializers.glorot_normal(seed=None)
        # else let be'random_uniform'
        past_poses = Input(shape=(None, self.input_dim),name='past_poses')#include current pose
        future_poses = Input(shape=(None, self.input_dim),name='future_poses')#include current pose
        past_poses_DO = layers.Lambda(lambda x: K.dropout(x, level=self.past_dropout_rate),name='dropout_past_poses')(past_poses) #this is just a dropout layers
        future_poses_DO = layers.Lambda(lambda x: K.dropout(x, level=self.future_dropout_rate),name='dropout_future_poses')(future_poses)

        if self.GPU:
            encoder = CuDNNLSTM(self.hidden_dim, return_state=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val), kernel_initializer=initializer)
        else:
            encoder = LSTM(self.hidden_dim, return_state=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val), kernel_initializer=initializer)

        encoder_outputs, state_h, state_c = encoder(past_poses_DO)
        encoder_states = [state_h, state_c]  # the last state?
        en_out = layers.Lambda(lambda x: x[:,None, :],name='encoder_h_T')(state_h)  # reshape to (?,1,hidden_dim)
        h_T = layers.Lambda(lambda x: x, name='h_T')(state_h)
        c_T = layers.Lambda(lambda x: x, name='c_T')(state_c)

        z_mean = Dense(self.latent_dim, name='z_mean')(h_T)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(h_T)
        z = layers.Lambda(sampling, name='z', output_shape=(self.latent_dim,))([z_mean, z_log_var])

        if self.latent_dim < self.hidden_dim:
            sampled_representation = Dense(self.hidden_dim, name='sampled_representation')(z)
        else:
            sampled_representation = h_T

        initial_state = [sampled_representation,c_T]
        future_decoder_inputs = layers.Lambda(lambda x: x[:, 1:, :])(future_poses_DO) #input: t+1 pose till T-1 pose.
        future_ground_truth = layers.Lambda(lambda x: x[:,1:, :])(future_poses)

        if not self.residual:
            print('normal model with teacherforce=', self.teacher_force)
            if self.layers>1:
                x = Input((self.input_dim,))
                h_tm1 = Input((self.hidden_dim,))
                c_tm1 = Input((self.hidden_dim,))
                readout_input = Input((self.input_dim,))
                lstms_input = readout_input
                cells = [LSTMCell(self.hidden_dim) for _ in range(self.layers)]
                lstms_output, h, c = lstms_input, h_tm1, c_tm1

                for cell in cells:
                    lstms_output, h, c = cell([lstms_output, h, c])
                    #lstms_output = Activation('relu')(lstms_output)
                output = Dense(self.input_dim, kernel_initializer=initializer)(lstms_output)
                #output = Activation('relu')(y)

                future_decoder = RecurrentModel(input=x, initial_states=[h_tm1,c_tm1], teacher_force=self.teacher_force,
                            output=output, final_states=[h, c], readout_input=readout_input, return_sequences=True,
                            name='future_pred')

                if self.teacher_force:
                    future_pred = future_decoder(future_decoder_inputs,initial_state = initial_state,ground_truth = future_ground_truth)
                else:
                    future_pred = future_decoder(future_decoder_inputs,initial_state = initial_state)

            else:
                future_decoder = RecurrentContainer(readout='readout_only',teacher_force=self.teacher_force,
                    return_sequences=True,name='future_pred') # previous output will be added to input
                future_decoder.add(LSTMCell(self.hidden_dim, input_dim=self.input_dim))
                future_decoder.add(Dense(self.input_dim,input_dim=self.hidden_dim))

                        #now we only take the first element from future pose, which is the current pose
                if self.teacher_force:
                    future_pred = future_decoder(future_decoder_inputs, initial_state = initial_state,
                    ground_truth = future_ground_truth)
                else:
                    future_pred = future_decoder(future_decoder_inputs, initial_state = initial_state)

        else:#train residue, difficult and not working as well...?
            print('residual model with teacherforce=', self.teacher_force)
            x = Input((self.input_dim,))
            h_tm1 = Input((self.hidden_dim,))
            c_tm1 = Input((self.hidden_dim,))
            readout_input = Input((self.input_dim,))
            lstms_input = readout_input
            cells = [LSTMCell(self.hidden_dim) for _ in range(self.layers)]
            lstms_output, h, c = lstms_input, h_tm1, c_tm1

            for cell in cells:
                lstms_output, h, c = cell([lstms_output, h, c])
                    #lstms_output = Activation('relu')(lstms_output)
                velocity = Dense(self.input_dim, kernel_initializer=initializer)(lstms_output)
                output = Add()([velocity, lstms_input]) #residue addition to input
                future_decoder = RecurrentModel(input=x, initial_states=[h_tm1,c_tm1], teacher_force=self.teacher_force,
                            output=output, final_states=[h, c], readout_input=readout_input, return_sequences=True,
                            name='future_pred')

            if self.teacher_force:
                future_pred = future_decoder(future_decoder_inputs,initial_state = initial_state,ground_truth = future_ground_truth)
            else:
                future_pred = future_decoder(future_decoder_inputs,initial_state = initial_state)

        self.vae_loss = compute_vae_loss(z_log_var,z_mean,self.kl_weight)

        if self.add_losses:
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = -0.5 * K.mean(K.sum(kl_loss, axis=None))
            #kl_loss = K.mean(- 0.5 * 1/self.latent_dim * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
            reconstruction_loss = mse(K.flatten(future_pred), K.flatten(future_poses[:,1:,:])) #predicts future
            self.recon_loss = reconstruction_loss
            self.kl_loss = kl_loss

        if self.classifier: #only if trained simulatneously as autoencoder
            latent_features = layers.concatenate([z_mean, z_log_var], 1, name='latent_features')
            classify_hidden = Dense(self.classifier_dense_dim, input_dim=self.latent_dim *2, kernel_initializer='normal',
                            activation='relu')(latent_features)

            pred_label = Dense(1, kernel_initializer='normal', activation='sigmoid',name='classified_label')(classify_hidden)
            #pred_label = self.classifier_network(latent_features) #use z_mean and z_log_var as the feature vecor?
            self.classifier_model = Model([past_poses, future_poses],pred_label) #the loss will be weighted and calculated outside
            self.recon_model = Model([past_poses, future_poses], future_pred)
            self.model = Model([past_poses, future_poses],[future_pred, pred_label])
        else:
            self.model = Model([past_poses, future_poses], future_pred)

        self.enc_model_pre_inference = Model(past_poses, en_out)
        self.enc_model = Model(past_poses, [z_mean, z_log_var])
