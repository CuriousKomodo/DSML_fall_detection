import os
import random
import numpy as np
import keras
from keras import callbacks, layers, losses, models, optimizers,regularizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
import scipy.sparse as sparse
import scipy.stats as stats
import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
import functions
from functions import *
import gc

# https://bitbucket.org/parthaEth/humanposeprediction/src/ERD/python_models/EvaluateAutoEncoder/DropOutJointsExpt.py

#This function randomly adds noise to joints of one example, a little fiddly.
def noisy_joints(data, num_noisy_joints=2, num_joints=17,density=0.05):
    np.random.seed(42)
    sparse_noise = sparse.random(data.shape[0],data.shape[1], density=density)
    sparse_noise = sparse_noise.toarray()
    noisy_data = data+ sparse_noise
    return noisy_data

#This function is called at the beginning of every epoch.
#takes input in the form of [N, 51]
def remove_joints(data, max_num_remove=5,num_joints=17,dim=2, noise_density=0.05):
    incomplete_data = data.reshape(data.shape[0], dim, num_joints)
    for i in range(data.shape[0]):
        #pick a random number of joints to remove
        r = np.random.randint(max_num_remove+1)
        joints_to_remove = np.random.randint(num_joints, size=r) #pick random joint to remove for each data
        for x in joints_to_remove:
            incomplete_data[i,:,x] = 0.0 #or let be a random number??
            #if noise_density>0:
            #    incomplete_data[i] = noisy_joints(data, num_noisy_joints=2, num_joints=17,density=0.05)
    incomplete_data = incomplete_data.reshape(data.shape[0], dim*num_joints)
    return incomplete_data



# Generates inputs for encoder: which is concat of future poses with past information.
# Also generates target, which is a few poses in future.
# Set to one step for now, while h_past is computed from t>10

class DAE_generator(keras.utils.Sequence):

    def __init__(self,
                 train_data_dir='/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/2D_DAE/X_train_train.npy',
                 batch_size=128, shuffle=True):

        self.train_data_dir = train_data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.change_joint_removal=False
        self.original, self.corrupted = self.load_data()
        self.on_epoch_end()

    def __len__(self):
        return int(self.original.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        #example is already reshaped into NT * 34.
        target  = self.original[idx * self.batch_size:(idx + 1) * self.batch_size]  # load by batch.
        corrupted = self.corrupted[idx * self.batch_size:(idx + 1) * self.batch_size]
        return corrupted, target

    def on_epoch_end(self):
        if self.shuffle:
            if self.change_joint_removal: #the the random removal is done every epoch.
                random.shuffle(self.original)
                self.corrupted = remove_joints(self.original, max_num_remove=5)
            else:
                ind = list(range(self.original.shape[0]))
                random.shuffle(ind)
                self.original = self.original[ind]
                self.corrupted =self.corrupted[ind]

    def load_data(self):
        X = np.load(self.train_data_dir)
        X = standardize_data(X,time_sequence=False)
        #print(X.max())
        #print(X.min())
        #X = X.reshape(X.shape[0]*X.shape[1], X.shape[-1], order='F') #unroll from the first dimension.
        missing_X = remove_joints(X, max_num_remove=5)
        return X, missing_X

#as suggested by the literature, LSTM3LR
class DAE(object):

    def __init__(self, hidden_dim=256,  # from past autoencoder
                 input_dim=34,lamda=0.0,dropout=0.05,
                 **kwargs):
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lamda = lamda
        self.dropout = dropout
        self.build_model()

    def build_model(self):
        dropout_poses = Input(shape=(self.input_dim,), name='dropout_poses')
        l1 = Dense(self.hidden_dim, activation='relu',kernel_regularizer=regularizers.l2(self.lamda))(dropout_poses)
        l1 = layers.Dropout(self.dropout)(l1)
        l2 = Dense(self.hidden_dim, activation='relu',kernel_regularizer=regularizers.l2(self.lamda))(l1)
        l2 = layers.Dropout(self.dropout)(l2)
        l3 = Dense(self.hidden_dim, activation='relu',kernel_regularizer=regularizers.l2(self.lamda))(l2)
        recon_poses = Dense(self.input_dim)(l3)
        self.model = Model(dropout_poses, recon_poses, name='dae')


#This implements the paper: Filling in the Gaps.
class DVAE():
    def __init__(self, hidden_dim1=64, hidden_dim2=128, latent_dim=64,# from past autoencoder
                input_dim=34,lamda=0.00, **kwargs):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.lamda = lamda
        self.kl_weight = 0.001
        #self.dropout = dropout
        self.build_model()

    def build_model(self):
        initializer = keras.initializers.glorot_normal(seed=None)
        dropout_poses = Input(shape=(self.input_dim,), name='dropout_poses')
        enc_l1 = Dense(self.hidden_dim1, name ='enc_l1',activation='relu',kernel_regularizer=regularizers.l2(self.lamda), kernel_initializer=initializer)(dropout_poses)
        enc_l2 = Dense(self.hidden_dim2, name ='enc_l2',activation='relu',kernel_regularizer=regularizers.l2(self.lamda),kernel_initializer=initializer)(enc_l1)
        z_mean = Dense(self.latent_dim, name='z_mean',kernel_initializer=initializer)(enc_l2)
        z_log_var = Dense(self.latent_dim, name='z_log_var',kernel_initializer=initializer)(enc_l2)
        z = layers.Lambda(sampling, name='z', output_shape=(self.latent_dim,))([z_mean, z_log_var])
        #symmetrical.
        dec_l1 = Dense(self.hidden_dim2, name ='dec_l1' ,activation='relu',kernel_regularizer=regularizers.l2(self.lamda),kernel_initializer=initializer)(z)
        dec_l2 = Dense(self.hidden_dim1, name = 'dec_l2',activation='relu',kernel_regularizer=regularizers.l2(self.lamda),kernel_initializer=initializer)(dec_l1)
        filled_poses = Dense(self.input_dim, activation='relu',kernel_regularizer=regularizers.l2(self.lamda),kernel_initializer=initializer)(dec_l2)

        self.enc_model = Model(dropout_poses, [z_mean, z_log_var])
        self.model = Model(dropout_poses, filled_poses)
        self.vae_loss = compute_vae_loss(z_log_var,z_mean,self.kl_weight)



'''
def expandToAll3AxisIndices(index_list):
#3 x 25!
    expanded_list = []
    for idx in index_list:
        for i in range(3):
            expanded_list.append((i * 25) + idx)
    return expanded_list

def random_select_2D(batch_num, dropout_rate):
    joint_num = 25
    num_to_drop = int(dropout_rate * joint_num)
    index_array = np.random.randint(joint_num, size=(batch_num, num_to_drop))
    return index_array


def random_select(dropout_rate):
    joint_num = 25
    num_to_drop = int(dropout_rate * joint_num)
    index_array = np.random.randint(joint_num, size=(num_to_drop))
    return index_arrayß
'''
