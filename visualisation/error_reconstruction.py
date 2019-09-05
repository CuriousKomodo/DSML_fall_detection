import numpy as np
import matplotlib.pyplot as plt

truth = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_60_2_untrimmed.npy')
recon = np.load('/Users/kefei/Documents/mm_fall_detection/results/win60/composite/past_ae_recon9.npy')
labels = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_test_fall_aug.npy')
labels_r = np.tile(labels,24)
#Plot the reconstruction error over time
composite = True
if not composite:
    recon = recon[:,::-1,:]

print(recon.shape)


def trim_dataset_mask(data,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    print(length)
    mask = (length>=threshold)
    return mask

def MSE_over_time(truth, recon, labels, action_class, trim_threshold=30, compare=False, composite=True, current_pose_ind=14):

    if trim_threshold:
        mask = trim_dataset_mask(truth,threshold=30)
        truth = truth[mask]
        recon = recon[mask]
        labels = labels[mask]
    truth = truth * 0.8 + 0.1


    if not compare:
        if action_class==43:
            truth = truth[labels==action_class]
            recon = recon[labels==action_class]
            actions_to_see = 'Falling' #maybe work on building a dictionary...

        elif not action_class:
            actions_to_see = 'All action classes'

        if composite:
            past_recon = recon[:,:current_pose_ind,:]
            future_recon = recon[:,current_pose_ind+1:,:]
            past_MSE_TS = np.mean(np.square(truth[:,:current_pose_ind,:]-past_recon), axis=(0,2))
            future_MSE_TS = np.mean(np.square(truth[:,current_pose_ind+2:,:]-future_recon), axis=(0,2))
            plt.plot(range(current_pose_ind),past_MSE_TS,'b')
            plt.plot(np.arange(current_pose_ind+1,30),future_MSE_TS,'g')
            plt.title('Reconstruction + Prediction MSE per joint per dimension over time - %s'%actions_to_see)
        else:
            MSE_TS = np.mean(np.square(truth-recon), axis=(0,2))
            print('MSE error for all time=',MSE_TS.mean())
            plt.plot(MSE_TS)
            plt.title('Reconstruction MSE per joint per dimension over time - %s'%actions_to_see)



    else:
        if action_class==43:
            action_truth = truth[labels==action_class]
            action_recon = recon[labels==action_class]
            actions_to_see = 'Falling'
            non_action_truth = truth[labels!=action_class]
            non_action_recon = recon[labels!=action_class]
            MSE_TS_action = np.mean(np.square(action_truth-action_recon), axis=(0,2))
            MSE_TS_non_action = np.mean(np.square(non_action_truth-non_action_recon), axis=(0,2))
            print('MSE error for all time for action =',MSE_TS_action.mean())
            print('MSE error for all time for non action =',MSE_TS_non_action.mean())
            plt.plot(MSE_TS_action,'r')
            plt.plot(MSE_TS_non_action,'b')
            plt.gca().legend((actions_to_see,'non-'+actions_to_see))
            plt.title('Reconstruction MSE per joint per dimension over time - comparing %s and non-%s'%(actions_to_see,actions_to_see))

    plt.ylabel('Reconstruction MSE')
    plt.xlabel('Time frame (forward)')
    plt.show()


#MSE_over_time(truth, recon, labels_r,43,30,False)
for kl_type in ['klanneal','klcyclic']:
    result_path1 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
    result_path2 = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/without_classifier/'%kl_type
    ts_1 = np.load(result_path1 + 'recon_loss_overtime.npy')
    ts_2 = np.load(result_path2 + 'recon_loss_overtime.npy')
    plt.plot(ts1[0:][::-1])
    plt.plot(ts2[0:][::-1])

plt.legend([
        'klanneal + classifier',
        'klanneal',
        'klcyclic + classifier',
        'klcyclic'
        #'hidden_dim=256, latent_dim=64'
        ], loc='upper right')
plt.title('Reconstruction Loss over 30 frames')
plt.ylabel('Reconstruction MSE')
plt.xlabel('Frame')
plt.show()
