#Future Predictor

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
os.environ['KERAS_BACKEND'] = 'tensorflow'

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
#config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from recurrentshop import *
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
import functions
from functions import *
import create_future_train_valid_split
from create_future_train_valid_split import *


class Future_DataGenerator_train(keras.utils.Sequence):

    def __init__(self,
                 train_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_train_train.npy',
                 encoder_result_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/',
                 train=True,
                 batch_size=128,
                 shuffle=True,
                 **kwargs):
        self.train_data_dir = train_data_dir
        self.encoder_result_dir = encoder_result_dir
        self.load_data()
        self.train = train
        self.batch_size = batch_size
        self.shuffle = True
        self.dataset_size = (self.X_train).shape[0]
        #self.data_list = self.load_data()
        self.on_epoch_end()

    def __len__(self):
        return int(self.X_train.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        future_poses= self.X_train[idx *self.batch_size:(idx + 1) * self.batch_size,:,:] #contains all the future poses including the current one.
        future_poses=0.8 * future_poses + 0.1
        z_mean = self.z_mean_train[idx *self.batch_size:(idx + 1) * self.batch_size,:]
        z_log_var = self.z_log_var_train[idx *self.batch_size:(idx + 1) * self.batch_size,:]

        future_target = future_poses[:,1:,:] #does not include current pose
        return [z_mean,z_log_var,future_poses], future_target

    def on_epoch_end(self):
        ind = np.arange(self.dataset_size)
        if self.shuffle:
            np.random.shuffle(ind) #shuffle the indices
            self.X_train = self.X_train[ind]
            self.z_log_var_train =self.z_log_var_train[ind]
            self.z_mean_train =self.z_mean_train[ind]

    def load_data(self):
        self.X_train = np.load(self.train_data_dir)
        self.z_log_var_train = np.load(self.encoder_result_dir +'z_log_var_train_train.npy')
        self.z_mean_train = np.load(self.encoder_result_dir + 'z_mean_train_train.npy')



class Future_DataGenerator_valid(keras.utils.Sequence):

    def __init__(self,
                 valid_data_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_train_valid.npy',
                 encoder_result_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/',
                 **kwargs):
        self.valid_data_dir = valid_data_dir
        self.encoder_result_dir = encoder_result_dir
        self.load_data()
        self.batch_size = self.X_valid.shape[0]

    def __len__(self):
        return 1 #int(len(self.data_list) / self.batch_size)

    def __getitem__(self, idx):
        future_poses = self.X_valid[idx *self.batch_size:(idx + 1) * self.batch_size] #contains all the future poses including the current one.
        z_mean = self.z_mean_valid[idx *self.batch_size:(idx + 1) * self.batch_size]
        z_log_var = self.z_log_var_valid[idx *self.batch_size:(idx + 1) * self.batch_size]
        future_poses = future_poses*0.8+0.1
        future_target = future_poses[:,1:,:] #does not include current pose
        return [z_mean, z_log_var, future_poses], future_target

    def load_data(self):
        self.X_valid= np.load(self.valid_data_dir)
        self.z_log_var_valid= np.load(self.encoder_result_dir +'z_log_var_train_valid.npy' )
        self.z_mean_valid = np.load(self.encoder_result_dir + 'z_mean_train_valid.npy')



class LSTM_predictor(object):
    def __init__(self, input_dim=75, latent_dim = 128, hidden_dim=1024, concat_h=True, W_regularizer_val=0.01, read_out = True, teacher_force = True,
                 conditional=False, lr=0.0005, epochs=10, dropout_rate=0.1, layers=3, residual=False, **kwargs):
        self.valid_data_dir = None
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.concat_h = concat_h
        self.lr = lr
        self.read_out = read_out
        self.teacher_force = teacher_force
        self.epochs = epochs
        self.layers = layers
        self.conditional = conditional
        self.residual = residual
        self.W_regularizer_val = W_regularizer_val
        self.dropout_rate = dropout_rate
        self.build_model()

    def build_model(self):
        initializer = keras.initializers.glorot_normal(seed=None)
        z_mean =  Input(shape=(self.latent_dim,))
        z_log_var =  Input(shape=(self.latent_dim,))
        z = layers.Lambda(sampling, name='z', output_shape=(self.latent_dim,))([z_mean, z_log_var])

        if self.hidden_dim==self.latent_dim:
            initial_states = [z, z]
        else:
            recovered_state = Dense(self.hidden_dim, kernel_initializer = initializer)(z)
            initial_states = [recovered_state, recovered_state]

        future_poses = Input(shape=(None, self.input_dim))
        future_poses_DO = layers.Lambda(lambda x: K.dropout(x, level=self.dropout_rate))(future_poses)
        future_decoder_inputs = layers.Lambda(lambda x: x[:, :-1, :])(future_poses_DO) #all future poses except for the last one.

        #Read out
        future_ground_truth = layers.Lambda(lambda x: x[:, 1:, :], name = 'ground_truth')(future_poses_DO) #ground truth of the future poses to be predicted

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
                y = Dense(self.input_dim, kernel_initializer=initializer)(lstms_output)
                output = Activation('relu')(y)
                future_decoder = RecurrentModel(input=x, initial_states=[h_tm1,c_tm1], teacher_force=self.teacher_force,
                        output=output, final_states=[h, c], readout_input=readout_input, return_sequences=True)
                if self.teacher_force:
                    future_pred = future_decoder(future_decoder_inputs,initial_state = initial_states,ground_truth = future_ground_truth)
                else:
                    future_pred = future_decoder(future_decoder_inputs,initial_state = initial_states)

            else:
                future_decoder = RecurrentContainer(readout='readout_only',teacher_force=self.teacher_force, return_sequences=True) # previous output will be added to input
                future_decoder.add(LSTMCell(self.hidden_dim, input_dim=self.input_dim))
                future_decoder.add(Dense(self.input_dim,input_dim=self.hidden_dim))
                future_decoder.add(Activation('relu'))

                #now we only take the first element from future pose, which is the current pose
                if self.teacher_force:
                    future_pred = future_decoder(future_decoder_inputs, initial_state = initial_states, ground_truth = future_ground_truth)
                else:
                    future_pred = future_decoder(future_decoder_inputs, initial_state = initial_states)

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
            y = Add()([velocity, lstms_input]) #residue addition to input
            output = Activation('relu')(y)
            future_decoder = RecurrentModel(input=x, initial_states=[h_tm1,c_tm1], teacher_force=self.teacher_force,
                    output=output, final_states=[h, c], readout_input=readout_input, return_sequences=True)
            if self.teacher_force:
                future_pred = future_decoder(future_decoder_inputs,initial_state = initial_states,ground_truth = future_ground_truth)
            else:
                future_pred = future_decoder(future_decoder_inputs,initial_state = initial_states)

        self.model = Model([z_mean,z_log_var,future_poses], future_pred)


hidden_dim=1024
latent_dim = 128
dropout_rate=0.05
window = 60
t =  2 #attempt
repeats = 3
layers_list = [1,2,3]
booleans = [True, False]


ave_val_errors = np.zeros((3,2,2))
ave_train_errors = np.zeros((3,2,2))

for r in range(repeats):
    train_valid_split()
    #load validation data
    valid_data_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_train_valid.npy'
    encoder_result_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/'
    future_poses_valid = np.load(valid_data_dir)
    z_log_var_valid= np.load(encoder_result_dir +'z_log_var_train_valid.npy' )
    z_mean_valid = np.load(encoder_result_dir + 'z_mean_train_valid.npy')
    future_poses_valid = future_poses_valid*0.8+0.1
    future_target_valid = future_poses_valid[:,1:,:] #does not include current pose


    val_errors = np.zeros((3,2,2))
    train_errors = np.zeros((3,2,2))
    for l in range(len(layers_list)):
        layer = layers_list[l]
        for i in range(2):
            bool1 = booleans[i]
            for j in range(2):
                bool2 = booleans[j]
                #train_valid_split()
                print('COMPLETE SPLIT!')
                print('residual=', bool1)
                print('teacher_force=',bool2)
                print('layers=', layer)
                result_path = '/Users/kefei/Documents/results/win%s/future/'%window
                model_path = '/Users/kefei/Documents/mm_fall_detection/models/win%s/future/'%window
                print('will save results at ', result_path)


                params = {'hidden_dim': hidden_dim, 'latent_dim':latent_dim,'dropout_rate': dropout_rate,'residual':bool1,'teacher_force':bool2, 'layers':layer}

                optimizer=optimizers.Adam(0.00005)
                predictor = None
                K.clear_session()
                predictor = LSTM_predictor()
                predictor.model.compile(optimizer ,loss=losses.mean_squared_error)
                #print(predictor.model.summary())
                print('finish compile')

                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=2, min_lr=0.00001)

                history_callback = predictor.model.fit_generator(generator=Future_DataGenerator_train(),
                                                            epochs=15, callbacks=[keras.callbacks.EarlyStopping(patience=3),reduce_lr],
                                                            validation_data=([z_mean_valid, z_log_var_valid, future_poses_valid], future_target_valid))


                loss_history = history_callback.history["loss"]
                val_loss_history = history_callback.history['val_loss']
                print('val loss =', val_loss_history[-1])
                train_errors[l,i,j] = loss_history[-1]
                val_errors[l,i,j] = val_loss_history[-1]
                ave_train_errors += train_errors
                ave_val_errors += val_errors

ave_train_errors = ave_train_errors*1.0/repeats
ave_val_errors = ave_val_errors*1.0/repeats

print(ave_train_errors)
print(ave_val_errors)
np.save(result_path+'train_CV.npy',ave_train_errors)
np.save(result_path+'val_CV.npy',ave_val_errors)

'''
        print('save losses')
        np.save(result_path+'train_loss_history%s.npy'%t, loss_history)
        np.save(result_path+'val_loss_history%s.npy'%t, val_loss_history)



        #make prediction
        test_data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_test.npy')
        future_test = test_data*0.8+0.1
        z_mean_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/z_mean_test.npy')
        z_log_var_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/z_log_var_test.npy')

        test_recon_pred = predictor.model.predict([z_mean_test, z_log_var_test,future_test])
        np.save(result_path+'past_ae_recon%s.npy'%t,test_recon_pred)
        print('save prediction')

        predictor.model.save_weights(model_path+"LSTM3LR_model%s_weights.h5"%t)
        print('save weights')
'''
