import keras.backend.tensorflow_backend as K
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
series = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_60_2_untrimmed.npy')
series = series[:10,:,:2]
n_input = 5
generator = TimeseriesGenerator(series, series,length=n_input, batch_size=1)
print(len(generator))
for i in range(10):
	x, y = generator[i]
	print(x.shape)
	print(y.shape)
