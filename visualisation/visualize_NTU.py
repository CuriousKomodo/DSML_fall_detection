import numpy as np
import matplotlib.pyplot as plt

data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/single_labels.npy')
data = data[data<50]
print(data)
hist = np.histogram(data)
print(hist)
#print hist(data)
plt.hist(data, bins=np.arange(data.min(), data.max()+2,1))
plt.ylabel('single-person action class')
plt.xlabel('Number of examples')
plt.title('Number of examples per single-person action class in NTU RGB+D')
plt.show()
