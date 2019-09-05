
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':14})

colors = ['green','blue']
k=0
for kl_type in ['klanneal','klcyclic']:

    classifier_bool = False
    if kl_type=='klanneal':
      if classifier_bool:
          model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/with_classifier/'%kl_type
          result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
      else:
          hyperparameters = [1024, 128, 0.004875173292623599, 0.04085806061902107, 32, 0.00025378664002876164, 861, 0.003950403393695316]
          model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/without_classifier/'%kl_type
          result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type

    elif kl_type=='klcyclic':
      if classifier_bool:
          hyperparameters = [768, 128, 0.001, 0.01, 128, 0.0005, 2, 0.0001]
          model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/with_classifier/'%kl_type
          result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
      else:
          hyperparameters = [768, 256, 0.001, 0.12, 32, 0.0005, 10, 0.0001]
          model_path = '/Users/kefei/Documents/results/win60/past/VAE/%s/without_classifier/'%kl_type
          result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type

    loss = np.load(result_path + 'loss.npy')
    val_loss = np.load(result_path + 'val_loss.npy')
    kl_loss = np.load(result_path + 'kl_loss.npy')
    val_kl_loss = np.load(result_path + 'val_kl_loss.npy')
    mse_loss = np.load(result_path + 'mse_loss.npy')
    val_mse_loss = np.load(result_path + 'val_mse_loss.npy')

    legend = ['Monotonic','Cyclic']

    for i in range(3):
        plt.subplot(1,3,1)
        plt.plot(val_loss, color = colors[k])
        plt.title('-ELBO + L2 Regularization')
        #plt.legend(legend,loc='upper right')
        plt.xlabel('batches')

        plt.subplot(1,3,2)
        plt.plot(val_kl_loss, color =colors[k] )
        plt.title('KL Divergence')
        #plt.legend(legend,loc='upper right')
        plt.xlabel('batches')
        print('final val kl=',val_kl_loss[-1])
        plt.subplot(1,3,3)
        plt.plot(val_mse_loss, color = colors[k])
        plt.title('Reconstruction Loss')
        #plt.legend(legend,loc='upper right')
        plt.xlabel('epochs')
    k+=1
#plt.legend(legend,loc='upper right')
plt.show()
