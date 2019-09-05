


# Generates inputs for encoder: which is concat of future poses with past information.
# Also generates target, which is a few poses in future.
# Set to one step for now, while h_past is computed from t>10

class future_generator(keras.utils.Sequence):

    def __init__(self,
                 train_data_dir=dataset_path + 'PKUMMD/X_train.npy',
                 # past_ae_dir = dataset_path+'models/keras_past_model_hidden.h5',
                 past_ae_model=None,
                 past_hidden_dim=256,
                 elem_dim=75,
                 step_size=2,
                 batch_size=128,
                 shuffle=True,
                 n_poses_predict=1,
                 timestep=10,
                 time_start=10,
                 **kwargs):

        self.train_data_dir = train_data_dir
        self.past_hidden_dim = past_hidden_dim
        self.timestep = timestep
        self.elem_dim = elem_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.time_start = time_start
        self.step_size = step_size
        self.data_list = self.load_data()
        self.past_ae = past_ae_model
        self.on_epoch_end()

    def load_past_model(self):
        past_model = load_model(self.past_ae_dir)
        return past_model

    def __len__(self):
        return int(len(self.data_list) / self.batch_size)

    def __getitem__(self, idx):
        '''
        data = self.data_list[idx * self.batch_size:(idx + 1) * self.batch_size]  # load by batch.
        X = np.asarray(data, dtype=np.float32)
        max_length = X.shape[1]
        time_frames = np.arange(self.time_start, max_length - self.timestep, self.step_size)
        timesteps_to_loop = len(time_frames)

        # treat each time step as a separate example
        h_past_array = np.zeros((self.batch_size * (timesteps_to_loop + 1), self.past_hidden_dim))
        future_poses_array = np.zeros((self.batch_size * (timesteps_to_loop + 1), self.timestep, self.elem_dim))
        target_array = np.zeros((self.batch_size * (timesteps_to_loop + 1), self.elem_dim))
        '''


        for i in range(len(time_frames)):  # timestep
            # always predict one time step ahead during training
            t = time_frames[i]
            # print('t=',t)
            future_poses = X[:, t:t + self.timestep, :]
            past_poses = X[:, :t, :]
            if self.past_ae:
                h_past = self.past_ae.predict(past_poses)
            else:
                print('Need a past autoencoder model')
            target = X[:, t, :]

            h_past_array[self.batch_size * i: self.batch_size * (i + 1), :] = h_past[:, 0, :]
            future_poses_array[self.batch_size * i:self.batch_size * (i + 1), :] = future_poses
            target_array[self.batch_size * i:self.batch_size * (i + 1), :] = target

        return [future_poses_array, h_past_array], target_array

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data_list)

    def load_data(self):
        data_list = np.load(self.train_data_dir)
        return data_list


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
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def mse(X, Y, axis=None):
    SSE = np.square(Y - X)
    MSE = K.mean(SSE, axis=axis)
    return MSE


class future_VAE(object):

    def __init__(self, hidden_dim=256,  # from past autoencoder
                 elem_dim=75,
                 timestep=5,
                 latent_dim=128,
                 epochs=5,
                 enc_intermediate_dim=256,
                 dec_intermediate_dim=256,
                 lr=0.001,
                 kl_weight=0.01
                 ):
        self.hidden_dim = hidden_dim
        self.elem_dim = elem_dim
        self.epochs = epochs
        self.enc_intermediate_dim = enc_intermediate_dim
        self.dec_intermediate_dim = dec_intermediate_dim
        self.past_ae = None
        self.timestep = timestep
        self.lr = lr
        self.epochs = epochs
        self.kl_weight = kl_weight
        self.build_model()

    def build_model(self):
        future_poses = Input(shape=(self.timestep, self.elem_dim), name='future_poses')  # fixed time step ahead.
        h_past = Input(shape=(self.hidden_dim,), name='h_past')
        future_poses_r = layers.Lambda(lambda x: x[:, :, :])(future_poses)
        future_poses_r = layers.Reshape((self.timestep * self.elem_dim,))(future_poses_r)

        # future encoder, but discarded in testing.
        enc_inputs = layers.concatenate([h_past, future_poses_r],
                                        1)  # so dimension = (?, hidden_layer + timesteps * 75 )
        enc_l1 = Dense(enc_intermediate_dim, activation='relu')(enc_inputs)
        self.z_mean = Dense(latent_dim, name='z_mean')(enc_l1)
        self.z_log_var = Dense(latent_dim, name='z_log_var')(enc_l1)
        z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])

        # instantiate encoder model
        encoder = Model([future_poses, h_past], [self.z_mean, self.z_log_var, z], name='encoder')

        # future decoder
        latent = Input(shape=(latent_dim,), name='z_sampling')
        latent_inputs = layers.concatenate([latent, h_past], 1)

        dec_l1 = Dense(dec_intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(elem_dim, activation='sigmoid')(dec_l1)

        # instantiate decoder model
        decoder = Model([latent, h_past], outputs, name='decoder')
        outputs = decoder([encoder([future_poses, h_past])[2], h_past])
        outputs_test = decoder([latent, h_past])

        vae = Model([future_poses, h_past], outputs, name='vae')
        vae_test = Model(z, outputs_test, name='vae_test')

    def train(self):
        vae_loss = compute_vae_loss(self.z_log_var, self.z_mean, self.kl_weight)
        vae.compile(optimizer=optimizers.Adam(lr=lr), loss=vae_loss)
        self.model.compile(optimizer=optimizers.Adam(self.lr), loss=losses.mean_squared_error)
        self.model.fit_generator(generator=future_generator(past_ae_model=past_ae),
                                 epochs=self.epochs, validation_data=([future_val, h_val[:, 0, :]], target_val))


#We know the future.. still in a training environment.
#now predict poses recursively for the nex...
n_pred = 50
n_start = 10
short_target_obs = X_test[:10,n_start,:]
long_target_obs = X_test[:10,:n_start+n_pred,:]
long_prediction = np.zeros_like(X_test[:10,: n_start + n_pred, :])
print(long_prediction.shape)
#fill the prediction sequence with past observation up to starting point.
long_prediction[:,:n_start,:] = X_test[:10,:n_start,:]

for t in np.arange(n_start,n_start+n_pred):
  past_obs = X_test[:10,:t,:]
  future_obs = X_test[:10,t:t+10,:]
  h_test = past_ae_lstm.predict(past_obs)
  pred =vae.predict([future_obs,h_test[:,0,:]])
  past_obs = np.concatenate([past_obs,pred.reshape(10,1,75)], axis=1)
  long_prediction[:,t,:] = pred
