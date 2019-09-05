import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

kl_type = 'klcyclic'
legend = [
    'M = 2',
    'M = 4',
    'M = 8']
legend2 = ['annealtime = 2',
    'annealtime = 5',
    'annealtime = 10']

hidden_dim = 1024
latent_dim = 128
r=0
i=0
plt.figure(1)
result_path = '/Users/kefei/Documents/results/win60/past/VAE/kl_tuning/'
kl_weight = []
view_train=True

if view_train:
    for i in range(3):
        elbo_loss = np.load(result_path+'batch_train_elbo_%s_%s_%s_%s_%s.npy'%(hidden_dim,latent_dim,r,kl_type,i))
        kl_loss = np.load(result_path+'batch_train_klloss_%s_%s_%s_%s_%s.npy'%(hidden_dim,latent_dim,r,kl_type,i))
        recon_loss = np.load(result_path+'batch_train_reconloss_%s_%s_%s_%s_%s.npy'%(hidden_dim,latent_dim,r,kl_type,i))

        plt.subplot(1,3,1)
        plt.plot(moving_average(elbo_loss[500:],n=500))
        plt.title('ELBO Loss')
        plt.legend(legend,loc='upper right')
        plt.xlabel('batches')

        plt.subplot(1,3,2)
        plt.plot(moving_average(kl_loss[500:],n=500))
        plt.title('KL Divergence')
        plt.legend(legend, loc='upper right')
        plt.xlabel('batches')

        plt.subplot(1,3,3)
        plt.plot(moving_average(recon_loss[500:],n=500))
        plt.title('Reconstruction Loss')
        plt.legend(legend, loc='upper right')
        plt.xlabel('batches')

else:
    for i in range(3):
        elbo_loss = np.load(result_path+'epoch_val_elbo_%s_%s_%s_%s_%s.npy'%(hidden_dim,latent_dim,r,kl_type,i))
        kl_loss = np.load(result_path+'epoch_val_kl_%s_%s_%s_%s_%s.npy'%(hidden_dim,latent_dim,r,kl_type,i))
        recon_loss = np.load(result_path+'epoch_val_recon_%s_%s_%s_%s_%s.npy'%(hidden_dim,latent_dim,r,kl_type,i))

        plt.subplot(1,3,1)
        plt.plot(elbo_loss)
        plt.title('ELBO Loss')
        plt.legend(legend,loc='upper right')
        plt.xlabel('batches')

        plt.subplot(1,3,2)
        plt.plot(kl_loss/10000)
        plt.title('KL Divergence')
        plt.legend(legend, loc='upper right')
        plt.xlabel('batches')

        plt.subplot(1,3,3)
        plt.plot(recon_loss)
        plt.title('Reconstruction Loss')
        plt.legend(legend, loc='upper right')
        plt.xlabel('batches')

plt.show()
