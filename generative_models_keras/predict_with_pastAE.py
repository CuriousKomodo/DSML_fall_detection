#Predict with pre-traine model

import numpy as np
import os
import sys
sys.setrecursionlimit(10000)
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend.tensorflow_backend as K
import keras
from keras.backend.tensorflow_backend import set_session
from keras import callbacks, layers, losses, models, optimizers
from keras.models import load_model
from Past_Autoencoder_model import *

# load model
weights_path = "/Users/kefei/Documents/mm_fall_detection/models/past_AE_model0.h5"
model = load_model(weights_path)
latent_dim=128
dropout_rate=0.05
params = {'latent_dim': latent_dim, 'dropout_rate': dropout_rate}
past_ae = past_LSTM_autoencoder(**params)
model = past_ae.model
## summarize model.
#model.summary()

test_data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/X_test_30.npy')
test_data = test_data*0.8+0.1
test_target = test_data[:,::-1,:]

loss = model.evaluate(test_data, test_target, verbose=0)
print('Reconstruction MSE=',loss)
recon = model.predict(test_data)

np.save('/Users/kefei/Documents/mm_fall_detection/results/past_ae_result1.npy', (recon,loss))
