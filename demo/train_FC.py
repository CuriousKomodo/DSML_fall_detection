#Use a single FC with softmax/log loss for classification
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend.tensorflow_backend as K
import keras
from keras import callbacks, layers, losses, models, optimizers
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


z_mean_train = np.load('/Users/kefei/Documents/mm_fall_detection/demo/data/z_mean_train.npy')
z_log_var_train = np.load('/Users/kefei/Documents/mm_fall_detection/demo/data/z_log_var_train.npy')
z_mean_test = np.load('/Users/kefei/Documents/mm_fall_detection/demo/data/z_mean_test.npy')
z_log_var_test = np.load('/Users/kefei/Documents/mm_fall_detection/demo/data/z_log_var_test.npy')
target_train = np.load('/Users/kefei/Documents/mm_fall_detection/demo/data/target_train.npy')
target_test = np.load('/Users/kefei/Documents/mm_fall_detection/demo/data/target_test.npy')
features_train = np.concatenate([z_mean_train,z_log_var_train],axis=-1)
features_test = np.concatenate([z_mean_test,z_log_var_test],axis=-1)
del z_mean_train
del z_log_var_train
del z_mean_test
del z_log_var_test

print('finished loading datasets')

features_train,features_valid,target_train,target_valid = train_test_split(features_train,target_train,0.15)
#need to tune this
FC = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,kernel_initializer='random_normal'),
    keras.layers.Dense(32, activation=tf.nn.relu,kernel_initializer='random_normal'),
    keras.layers.Dense(1, activation=tf.nn.sigmoid,kernel_initializer='random_normal')
])

optimizer=optimizers.Adam(0.0002)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=3, min_lr=0.00005)
FC.compile(optimizer =optimizer, loss='binary_crossentropy', metrics =['accuracy'])
FC.fit(features_train,target_train, batch_size=128, epochs=100,verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(patience=3),reduce_lr],validation_data=(features_valid,target_valid))
