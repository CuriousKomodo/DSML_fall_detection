import numpy as np
from keras import backend as K


def mse(X, Y, axis=None):
    SSE = np.square(Y - X)
    MSE = K.mean(SSE, axis=axis)
    return MSE

def recall_m(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


#clean the dataset.
def trim_dataset(data,label,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    data = data[length>=threshold,:,:]
    label =label[length>=threshold]
    return data, label


def compute_vae_loss(z_log_var, z_mean, kl_weight=0.01):
    """" Wrapper function which calculates auxiliary values for the complete loss function.
     Returns a *function* which calculates the complete loss given only the input and target output """

    # KL loss
    def compute_kl(z_mean, z_log_var, kl_weight):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = -0.5 * kl_weight * K.sum(kl_loss, axis=-1)
        return kl_loss

    kl_loss = compute_kl(z_mean, z_log_var, kl_weight)

    def vae_loss(y_true, y_pred):
        md_loss = mse(y_true, y_pred)
        model_loss = kl_loss + md_loss
        return model_loss

    return vae_loss

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    z = z_mean + K.exp(0.5 * z_log_var) * epsilon
    return z

def standardize_data(data, time_sequence=True):
    if time_sequence:
        v_min = data.min(axis=(1,2), keepdims=True)
        v_max = data.max(axis=(1,2), keepdims=True)
        data_n = (data - v_min)/(v_max - v_min)
        mask = np.all(np.isnan(data_n) | np.equal(data_n, 0), axis=(1,2)) #remove all nan data
    else:
        v_min = data.min(axis=1, keepdims=True)
        v_max = data.max(axis=1, keepdims=True)
        data_n = (data - v_min)/(v_max - v_min)
        mask = np.all(np.isnan(data_n) | np.equal(data_n, 0), axis=(1))
    data_n = data_n[~mask]
    return data_n

def crop_2D(X):#takes 3D skeleton
    X = X.reshape(X.shape[0],X.shape[1],3,int(X.shape[-1]/3))
    X = X[:,:,:2,:] #only extract x,y cooridnates
    X = X.reshape(X.shape[0],X.shape[1],2*X.shape[-1])
    return X

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


def select_the_fall_frames(fall_df):
     #since we move in a step size of 10, we assume fall only happens after the 20th frame
     #so we only extract the examples after 10th frame.
     #but typically, the number of frames for falling is smaller than 100. This means we can exclude the 7/8th batch
     cycle = int(fall_df.shape[0]/24)
     fall_df_crop = fall_df[cycle*0:cycle*5]
     return fall_df_crop

'''
def get_gradients(model):
    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    weights = [tensor for tensor in model.trainable_weights if model.get_layer(
        tensor.name[:-2]).trainable]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)
'''
