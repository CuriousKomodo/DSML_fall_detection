import numpy as np

X_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_fall_aug_int.npy')
X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_fall_aug_int.npy')
window = 60
time_start = 60
step_size = 10
sample_freq = 2
trim = False
integrated = True

#24 cycle if 60,2. 27 cycle if 30,1
max_length = X_train.shape[1]
time_frames = np.arange(time_start, max_length, step_size)
timesteps_to_loop = len(time_frames)

train_mini = []
test_mini = []

for i in np.arange(timesteps_to_loop):
    t = time_frames[i]
    print('t=',t)
    past_train_observation = X_train[:,t-window:t:sample_freq,:]
    past_test_observation = X_test[:,t-window:t:sample_freq,:]
    print(past_train_observation.shape)
    #print(train_mini[i*X_train.shape[0]:(i+1)*X_train.shape[0],:,:].shape)
    #train_mini[i*X_train.shape[0]:(i+1)*X_train.shape[0],:,:] = past_train_observation
    train_mini.append(past_train_observation)
    test_mini.append(past_test_observation)
    #test_mini[i*X_test.shape[0]:(i+1)*X_test.shape[0],:,:] = past_test_observation

def concat(data_list):
    dataset = data_list[0]
    for i in range(len(data_list)-1):
        print('i=',i)
        print('dataset shape',dataset.shape)
        dataset = np.concatenate([dataset,data_list[i+1]], axis=0)
    return dataset


train_mini = np.concatenate(train_mini)
test_mini = np.concatenate(test_mini)

def clean_data(data,threshold=20):
    used = np.sign(np.max(abs(data),axis=-1))
    length  = np.sum(used, axis=1)
    clean_data = data[length>=threshold]
    return clean_data


if trim:
    train_mini = clean_data(data,threshold=15)
    test_mini = clean_data(data,threshold=15)

    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_30.npy',train_mini )
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_30.npy',test_mini)

else:
    if not integrated:
        print('save untrimmed version, original')
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_train_60_2_untrimmed.npy',train_mini )
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_60_2_untrimmed.npy',test_mini)
    else:
        print('save untrimmed version, integrated')
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_train_60_2_untrimmed.npy',train_mini )
        np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_untrimmed.npy',test_mini)
