
from keras.layers import LSTM, CuDNNLSTM
from keras.layers import Dense, Activation
from keras.layers import Input
from keras.models import Model
from keras.layers import Dropout
from keras import backend as K
from keras import callbacks,regularizers, layers, losses, models, optimizers
import keras
import random
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from recurrentshop import *
print('recurrentshop')
# CuDNNLSTM
#Non conditional

class Composite_DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 #train_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_30_untrimmed.npy',
                 #valid_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_30_untrimmed.npy',
                 train_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_60_2_untrimmed.npy',
                 valid_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_60_2_untrimmed.npy',

                 train=True,
                 batch_size=128,
                 shuffle=True,
                 past_steps = 15,
                 future_steps = 15,
                 # can only be set to true when shuffle=False.
                 sort_batch_lengths=False,
                 **kwargs):

        self.train_data_dir = train_data_dir
        self.valid_data_dir = valid_data_dir
        self.train = train
        self.sort_batch_lengths = sort_batch_lengths
        self.batch_size = batch_size
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.shuffle = shuffle
        self.data_list = self.load_data()
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.data_list) / self.batch_size)

    def __getitem__(self, idx):
        if self.train:
            # load by batch.
            data = self.data_list[idx *
                                  self.batch_size:(idx + 1) * self.batch_size]

            if self.sort_batch_lengths:
                lengths_batch = self.lengths[idx *
                                             self.batch_size:(idx + 1) * self.batch_size]
                # print(lengths_batch)
                max_length = lengths_batch.max()
                data = data[:, :int(max_length), :]

        else:
            data = self.data_list

        #The input: current+future poses, past_representation. The output is future_poses.
        data = 0.8 * data + 0.1
        future_poses = np.asarray(data[:,self.past_steps:,:], dtype=np.float32) #include current pose
        past_poses = data[:,:self.past_steps+1,:] #also include current pose.
        #past_target = np.flip(past_poses[:,:-1,:],axis=1) #flipped, doesn't include current pose
        past_target = past_poses[:,:-1,:]
        future_target = future_poses[:,1:,:] #does not include current pose
        full_target = np.concatenate([past_target, future_target], axis=1)
        return [past_poses,future_poses], full_target

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data_list)

    def load_data(self):
        if self.train:
            data_list = np.load(self.train_data_dir)
            print('finish loading training set!')

        else:
            data_list = np.load(self.valid_data_dir)

        return data_list


def get_gradients(model):
    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    weights = [tensor for tensor in model.trainable_weights if model.get_layer(
        tensor.name[:-2]).trainable]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)


class Composite_LSTM_autoencoder(object):

    def __init__(self, input_dim=75, latent_dim=1024, concat_h=True, W_regularizer_val=0.01,
                 conditional=False, lr=0.0005, epochs=10, dropout_rate=0.1, **kwargs):
        self.valid_data_dir = None
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.concat_h = concat_h
        self.lr = lr
        self.read_out = True
        self.epochs = epochs
        self.conditional = conditional
        self.W_regularizer_val = W_regularizer_val
        self.dropout_rate = dropout_rate
        self.build_model()
        #if self.valid_data_dir:
        #    self.x_val = self.load_validation_data()
        #    self.target_val = np.flip(self.x_val, axis=1)

    def build_model(self):
        initializer = keras.initializers.glorot_normal(seed=None)
        # else let be'random_uniform'

        future_poses = Input(shape=(None, self.input_dim))
        past_poses = Input(shape=(None, self.input_dim))

        past_poses_DO = layers.Lambda(lambda x: K.dropout(x, level=self.dropout_rate))(past_poses)
        future_poses_DO = layers.Lambda(lambda x: K.dropout(x, level=self.dropout_rate))(future_poses)

        encoder = CuDNNLSTM(self.latent_dim, return_state=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val), kernel_initializer=initializer)
        past_decoder = CuDNNLSTM(self.latent_dim, return_sequences=True, kernel_regularizer=regularizers.l2(self.W_regularizer_val),
                       go_backwards=True,kernel_initializer=initializer)

        encoder_outputs, state_h, state_c = encoder(past_poses_DO)
        encoder_states = [state_h, state_c]  # the last state?
        en_out = layers.Lambda(lambda x: x[:, None, :])(encoder_outputs)  # reshape to (?,1,latent_dim)

        #Reconstructing the past poses in reverse, need x_1, ... x_t
        past_decoder_inputs = layers.Lambda(lambda x: x[:, 1:, :])(past_poses_DO)
        past_decoder_outputs = past_decoder(past_decoder_inputs, initial_state=encoder_states)
        past_ts_dense_inputs = past_decoder_outputs
        #past_ts_dense_inputs = layers.concatenate([en_out, past_decoder_outputs], 1)
        past_ts_dense = layers.TimeDistributed(layers.Dense(self.input_dim, kernel_initializer=initializer))
        past_ts_dense_inputs = Activation('relu')(past_ts_dense_inputs)
        past_recon =past_ts_dense(past_ts_dense_inputs)
        future_decoder_inputs = layers.Lambda(lambda x: x[:, :-1, :])(future_poses_DO)

        #Map the encoder representation into a latent space.

        future_ground_truth = layers.Lambda(lambda x: x[:, 1:, :])(future_poses_DO)
        future_decoder = RecurrentSequential(readout='readout_only',return_sequences=True,
                teacher_force=True) # previous output will be added to input
        future_decoder.add(LSTMCell(self.latent_dim, input_shape= (self.input_dim,)))
        future_decoder.add(Activation('relu'))
        future_decoder.add(Dense(self.input_dim))
            #future_decoder.add(Activation('relu'))
            #now we only take the first element from future pose, which is the current pose
        future_pred = future_decoder(future_decoder_inputs,
        initial_state = encoder_states, ground_truth = future_ground_truth)

        past_recon_reverse = layers.Lambda(lambda x: x[:, ::-1, :])(past_recon) #reverse past reconstruction so it is on the forward order again!
        full_pred = layers.concatenate([past_recon_reverse, future_pred], 1)
        self.enc_model = Model(past_poses, en_out)
        self.recon_model = Model(past_poses, past_recon)
        #might need to define two losses? or not.
        self.composite_model = Model([past_poses, future_poses], full_pred)


latent_dim=1024
dropout_rate=0.05
window = 60
t =  10 #attempt

result_path = '/Users/kefei/Documents/mm_fall_detection/results/win%s/'%window
model_path = '/Users/kefei/Documents/mm_fall_detection/models/win%s/'%window
print('will save results at ', result_path)


params = {'latent_dim': latent_dim, 'dropout_rate': dropout_rate}
params_train = {'train': True}
#valid_data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_%s_untrimmed.npy'%window)
valid_data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_60_2_untrimmed.npy')
valid_data = valid_data*0.8+0.1

past_steps = 15
future_steps = 15
future_poses = np.asarray(valid_data[:,past_steps:,:], dtype=np.float32) #include current pose
past_poses = valid_data[:,:past_steps+1,:] #also include current pose.
#past_target = np.flip(past_poses[:,:-1,:],axis=1) #flipped, doesn't include current pose
past_target = past_poses[:,:-1,:]
future_target = future_poses[:,1:,:] #does not include current pose
valid_target = np.concatenate([past_target, future_target], axis=1)

optimizer=optimizers.Adam(0.0001)
composite_ae = Composite_LSTM_autoencoder()
composite_ae.composite_model.compile(optimizer ,loss=losses.mean_squared_error)
print(composite_ae.composite_model.summary())
print('finish compile')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=6, min_lr=0.00005)
history_callback = composite_ae.composite_model.fit_generator(generator=Composite_DataGenerator(**params_train),
                                            epochs=20, callbacks=[keras.callbacks.EarlyStopping(patience=10),reduce_lr],
                                            validation_data=([past_poses, future_poses],valid_target))#Past_DataGenerator(**params_valid)


loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history['val_loss']


print('save losses')
np.save(result_path+'train_loss_history%s.npy'%t, loss_history)
np.save(result_path+'val_loss_history%s.npy'%t, val_loss_history)

#make prediction
valid_recon_pred = composite_ae.composite_model.predict([past_poses, future_poses])
np.save(result_path+'past_ae_recon%s.npy'%t,valid_recon_pred)
print('save reconstruction and prediction')


composite_ae.composite_model.save_weights(model_path+"composite_AE_model%s_weights.h5"%t)
print('save weights')
composite_ae.enc_model.save_weights(model_path+"composite_AE_enc_model%s_weights.h5"%t)
print('save hidden model weights')
'''
composite_ae.enc_model.save( model_path+"/composite_AE_enc_model%s.h5"%t)
print('save hidden model')
composite_ae.composite_model.save(model_path+"composite_AE_model%s.h5"%t)
print('save model')
'''
