
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
import keras
import numpy as np
#os.environ['KERAS_BACKEND'] = 'tensorflow'


'''
import numpy as np
paths = ['/Users/kefei/Documents/Dataset/NTU/single_poses/X_train_30.npy','/Users/kefei/Documents/Dataset/NTU/single_poses/X_test_30.npy']
save_paths = ['/Users/kefei/Documents/Dataset/NTU/single_poses/train/dataset_%s.npy','/Users/kefei/Documents/Dataset/NTU/single_poses/test/dataset_%s.npy']

for i in range(len(paths)):
    path = paths[i]
    save_path = save_paths[i]
    data = np.load(path)
    data_list = np.split(data,15,axis=0)
    i=0
    #data_list = range(15)
    for dataset in data_list:
        print(save_path%i)
        np.save(save_path%i, dataset)
        i+=1
'''

datagen = keras.preprocessing.image.ImageDataGenerator()
# load and iterate training dataset
train_it = datagen.flow_from_directory('/Users/kefei/Documents/Dataset/NTU/single_poses/train/', class_mode=None, batch_size=64)
# load and iterate validation dataset
#val_it = datagen.flow_from_directory('data/validation/', class_mode='binary', batch_size=64)
# load and iterate test dataset
#test_it = datagen.flow_from_directory('/Users/kefei/Documents/Dataset/NTU/single_poses/test/', class_mode='binary', batch_size=64)
print(len(train_it))

#Generate target on the fly?
