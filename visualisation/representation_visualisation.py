import os
import random
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
plt.rcParams.update({'font.size':14})


import sys
sys.path.insert(0, '/Users/kefei/Documents/mm_fall_detection/tool/')
from functions import *

def sampling(args):
    z_mean, z_log_var = args
    batch = z_mean.shape[0]
    dim = z_mean.shape[1]
    epsilon = np.random.normal(size=(batch, dim))
    z = z_mean + np.exp(0.5 * z_log_var) * epsilon
    return z

kl_type = 'klcyclic'
data_dir = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/VAE/%s/with_classifier/'%kl_type
#data_dir = '/Users/kefei/Documents/results/win60/future/VAE/non_residual/'
#X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_trimmed.npy')
y_test_original = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed.npy')
y_test = np.array([1 if x==43 else 0 for x in y_test_original])
'''
y_test_original = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed25.npy')
y_test_r = np.repeat(y_test_original,4)
y_test_original = np.concatenate([y_test_original, y_test_r])
y_test_original =y_test_original[:150000]
y_test = np.asarray([1 if x==43 else 0 for x in y_test_original])
'''
#y_test = np.tile(y_test, 24)
#X_test, y_test = trim_dataset(X_test, y_test,threshold=5)
#np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/X_test_60_2_trimmed.npy',X_test)
#np.save('/Users/kefei/Documents/Dataset/NTU/single_poses/integrated/3D/y_test_60_2_trimmed.npy',y_test)
#del X_test

VAE = True
if VAE:
    z_mean = np.load(data_dir+'z_mean_test.npy')
    z_log_var = np.load(data_dir+'z_log_var_test.npy')
    hidden = np.concatenate([z_mean,z_log_var],axis=-1)
    hidden = sampling([z_mean, z_log_var])
    #hidden = z_mean

    print('hidden',hidden.shape)
else:
    result_path = '/Users/kefei/Documents/mm_fall_detection/models/win60/past/AE/with_classifier/'
    hidden = np.load(result_path+'hidden.npy')
    hidden = hidden[:,0,:]


def visualize_cluster(hidden, labels_r, original_labels_r,method='PCA', binary=True, VAE=True):
    print(labels_r.shape)
    print(hidden.shape)

    feat_cols = ['layer'+str(i) for i in range(hidden.shape[-1])]
    df = pd.DataFrame(hidden,columns=feat_cols)
    rndperm = np.random.permutation(hidden.shape[0])

    action_dict = {'16':'put on a shoe','42':'staggering', '43':'fall',
                    '6':'pick up', '17':'take off a shoe', '48':'nausea/vomiting', '14':'put on jacket','0':'other'}
    top_examples = [43,16,42,6,17,48]

    df['y'] =labels_r
    df['original_y'] = original_labels_r
    df['original_y'] = [str(x) if x in top_examples else '0' for x in original_labels_r]
    df['Selected action class'] = [action_dict[x] for x in df['original_y']]
    df['Action class'] = ['Fall' if x == 1 else 'Non-fall' for x in labels_r]

    if binary:
        category = 'Action class'
    else:
        category = 'Selected action class'

    if VAE:
        autoencoder_type = 'VAE'
    else:
        autoencoder_type = 'AE'

    if binary:
        title = "Representation(z) from Past-VAE (CA), Fall vs. Non-fall"#%(autoencoder_type)
    else:
        title = "Representation from Past-VAE (MA + classifier), by action classes"#%autoencoder_type

    title = title + ' via %s'%method

    if method=='PCA':
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df[feat_cols].values)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1]
        df['pca-three'] = pca_result[:,2]

        if binary:
            non_fall_df = df[df['y']==0].sample(50000)
            fall_df = df[df['y']==1]
        else:
            fall_df = df[df['Selected action class']!='other']
            non_fall_df = df[df['Selected action class']=='other'].sample(1000)

        new_df= pd.concat([fall_df,non_fall_df],ignore_index=True)
        new_df['size']=[3 if x==1 else 1 for x in new_df['y']]

        if binary:
            colors = sns.color_palette("hls", 2)
        else:
            colors = None

        plt.figure(figsize=(12,8))
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue=category,size='size',
            palette=colors,
            data=new_df,
            legend="full",
            alpha=0.3
        )
        ax = plt.gca()
        ax.set_title(title)
        plt.show()

    else:
        pca = PCA(n_components=15)
        pca_result = pca.fit_transform(df[feat_cols].values)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        print('tsne')
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(pca_result )
        df['tsne-2d-one'] = tsne_results[:,0]
        df['tsne-2d-two'] = tsne_results[:,1]

        if binary:
            non_fall_df = df[df['y']==0].sample(50000)
            fall_df = df[df['y']==1]
        else:
            fall_df = df[df['Selected action class']!='other']
            non_fall_df = df[df['Selected action class']=='other'].sample(1000)

        new_df= pd.concat([fall_df,non_fall_df],ignore_index=True)
        new_df['size']=[3 if x==1 else 1 for x in new_df['y']]
        palette = sns.color_palette("bright",7 )
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue=category,
            palette=palette,
            data=new_df,
            legend="full",
            alpha=0.3,
            s=10)
        title = 't-SNE latent visualisation: fall vs. top false positive'
        plt.title(title)
        plt.show()

visualize_cluster(hidden, y_test, y_test_original,method='PCA', binary=True, VAE=False)
