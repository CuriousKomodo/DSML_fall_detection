import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})


latent_dims= [64, 128, 256, 512] #128
hidden_dims = [256,512,1024]
latent_hidden_dims = [[64,512],[128,1024],[256,512]]
dropout_rates=[0.05, 0.1, 0.2, 0.4]

repeats = 5
batch_num = 6182

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

plt.figure(1)
#latent_dims = [64]
latent_hidden_dims = [[128,1024]]
result_path = '/Users/kefei/Documents/results/win60/past/VAE/hyperparameter_tuning/'
'''
for i in range(len(latent_hidden_dims)):
    hidden_dim = latent_hidden_dims[i][1]
    latent_dim = latent_hidden_dims[i][0]
    for j in range(len(dropout_rates)):
        r = 0
        dropout_rate = dropout_rates[j]
        #train_loss = np.load(result_path+'batch_train_loss_512_64_0_dropout_0')
        train_loss = np.load(result_path+'batch_train_loss_%s_%s_%s_dropout_%s.npy'%(hidden_dim,latent_dim,r,j))

        valid_loss = np.load(result_path+'epoch_val_loss_%s_%s_%s_dropout_%s.npy'%(hidden_dim,latent_dim,r,j))
        train_loss_ave = moving_average(train_loss[1000:],n=500)
        range_epoch = np.arange(0,len(train_loss_ave),batch_num)
        plt.subplot(2,2,j+1)
        plt.plot(train_loss_ave)
        plt.plot(range_epoch, valid_loss)
        plt.title('L=%s, H=%s, dropout rate = %s'%(latent_dim, hidden_dim, dropout_rate))
        plt.legend([
            'training loss',
            'validation loss'
            #'hidden_dim=256, latent_dim=64'
            ], loc='upper right')
plt.show()


result_path =  '/Users/kefei/Documents/results/win60/past/VAE/'

train_losses = np.load(result_path+'train_loss_history1.npy')
print('average training losses',train_losses)
valid_losses = np.load(result_path+'val_loss_history1.npy')
print('average validation losses',valid_losses)
'''
for i in range(len(latent_dims)):
    for j in range(len(hidden_dims)):
        latent_dim = latent_dims[i]
        hidden_dim = hidden_dims[j]

        #for r in range(repeats):
        r=0
        train_loss = np.load(result_path+'batch_train_loss_%s_%s_%s_dropout_0.npy'%(hidden_dim,latent_dim,r))
        train_loss = train_loss[1000:]
        train_loss_mv = moving_average(train_loss, n=200)
        plt.subplot(2,2,i+1)
        plt.plot(train_loss_mv)
        plt.title('Ave.training loss, fixed latent dim = %s'%latent_dim)
        plt.legend([
        'hidden_dim=256',
        'hidden_dim=512',
        'hidden_dim=1024'
        #'hidden_dim=256, latent_dim=64'
        ], loc='upper right')

plt.show()
t =1
