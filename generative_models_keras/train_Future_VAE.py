
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
from tensorflow import Graph
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

from Past_Autoencoder_model import *

from Future_VAE import *
import random
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


gen_params = {'past_latent_dim': 1024}
future_vae = future_VAE()
future_vae.train(optimizers.Adam(lr=0.0001))
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.00005)
future_vae.vae.fit_generator(generator=future_generator(),
                        callbacks=[keras.callbacks.EarlyStopping(patience=3),reduce_lr],
                         epochs=10)#validation_data=future_generator_valid()

vae_model = future_vae.vae
