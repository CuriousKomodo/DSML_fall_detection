import numpy as np
from sklearn.model_selection import train_test_split

def train_valid_split():
    encoder_result_dir = '/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/'
    future_train = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_train.npy')
    z_log_var = np.load(encoder_result_dir+'z_log_var_train.npy')
    z_mean = np.load(encoder_result_dir+'z_mean_train.npy')
    print('future train shape',future_train.shape)

    data_list = train_test_split(future_train, z_log_var, z_mean, test_size=0.15, random_state=42)
    train = data_list[0]
    z_log_var_train = data_list[2]
    z_mean_train = data_list[4]
    print('finish allocating the train and validation set')
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_train_train.npy', train)
    np.save(encoder_result_dir+'z_log_var_train_train.npy', z_log_var_train)
    np.save(encoder_result_dir+'z_mean_train_train.npy', z_mean_train)

    valid = data_list[1]
    z_log_var_valid = data_list[3]
    z_mean_valid = data_list[5]
    np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/prediction/future_poses_trimmed_15_train_valid.npy', valid)
    np.save(encoder_result_dir+'z_log_var_train_valid.npy', z_log_var_valid)
    np.save(encoder_result_dir+'z_mean_train_valid.npy', z_mean_valid)
    print('finish saving all data')
