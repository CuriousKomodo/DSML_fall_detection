import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':14})

'''
result_path = '/Users/kefei/Documents/results/win60/future/VAE/non_residual/'
ts = np.load(result_path + 'recon_loss_overtime.npy')
plt.plot(ts)
plt.show()


'''
#MSE_over_time(truth, recon, labels_r,43,30,False)
for kl_type in ['klanneal','klcyclic']:
    result_path1 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
    result_path2 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type
    ts_1 = np.load(result_path1 + 'recon_loss_overtime.npy')
    ts_2 = np.load(result_path2 + 'recon_loss_overtime.npy')
    print(ts_1.shape)
    plt.plot(ts_1[::-1][:-1])
    plt.plot(ts_2[::-1][:-1])

result_path0 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/AE/with_classifier/'
ts_0 = np.load(result_path0 + 'recon_loss_overtime.npy')
plt.plot(ts_0[::-1][:-1])

plt.legend([
        'VAE, KL Anneal + classifier',
        'VAE, KL Anneal',
        'VAE, KL Cyclic + classifier',
        'VAE, KL Cyclic',
        'AE'
        #'hidden_dim=256, latent_dim=64'
        ], loc='upper right')

plt.title('Reconstruction Loss over 30 frames')
plt.ylabel('Reconstruction MSE')
plt.xlabel('Frame')
plt.show()
