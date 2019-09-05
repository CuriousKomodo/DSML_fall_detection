import numpy as np
from sklearn.model_selection import train_test_split

#Splits the dataset with even number of fall examples!
centre = False
data = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/single_dataset.npy')
label = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/single_labels.npy')

#centre with the location of hip. Then normalize...?
def center_by_joint_location(joint_num, dataset):
    #if joint num = 0, then for x axis, ind = 0, for y: ind = 25, for z: ind = 50
    dataset[:,:,0:25] = dataset[:,:,0:25] - dataset[:,:,joint_num,None]
    dataset[:,:,25:50] = dataset[:,:,25:50] - dataset[:,:,joint_num+25,None]
    dataset[:,:,50:] = dataset[:,:,50:] - dataset[:,:,joint_num+50,None]
    return dataset

if centre:
    data = center_by_joint_location(joint_num, data)


#v_min = data.min(axis=(1,2), keepdims=True)
#v_max = data.max(axis=(1,2), keepdims=True)
#data_n = (data - v_min)/(v_max - v_min)
mask = np.all(np.isnan(data) | np.equal(data, 0), axis=(1,2))
data_n = data[~mask,:,:]
label_n = label[~mask]
#length_nn  = length[~mask]
used = np.sign(np.max(abs(data_n),axis=-1))
length  = np.sum(used, axis=1)
print('length=',length)

#Make sure the values are cleanly padded with 0
'''
for i in range(data_nn.shape[0]):
    data_nn[i,int(length_nn[i]):,:] = 0
'''

print('Number of fall examples',len(label_n[label_n==43]))
#Evenly split?

unique, counts = np.unique(label_n, return_counts=True)
count_examples = dict(zip(unique, counts))

##
fall_data = data_n[label_n==43]
fall_label = label_n[label_n==43]
non_fall_data = data_n[label_n!=43]
non_fall_label = label_n[label_n!=43]

#Dunno if evenly split?
X_train_non_fall, X_test_non_fall, y_train_non_fall, y_test_non_fall = train_test_split(non_fall_data, non_fall_label, test_size=0.2, random_state=1)
X_train_fall,X_test_fall, y_train_fall, y_test_fall = train_test_split(fall_data, fall_label, test_size=0.2, random_state=1)

#Then concat:
X_train = np.concatenate([X_train_non_fall,X_train_fall],axis=0)
y_train = np.concatenate([y_train_non_fall,y_train_fall],axis=0)
X_test = np.concatenate([X_test_non_fall,X_test_fall],axis=0)
y_test = np.concatenate([y_test_non_fall,y_test_fall],axis=0)


np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train.npy',X_train )
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test.npy',X_test)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_train.npy',y_train)
np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_test.npy',y_test)
