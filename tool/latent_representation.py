import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

X_test = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/X_test_60_2_untrimmed.npy')
result_path = '/Users/kefei/Documents/results/win60/past/VAE/'
mean = np.load(result_path+'z_mean1.npy')
log_sigma = np.load(result_path+'z_log_var1.npy')
labels = np.load('/Users/kefei/Documents/Dataset/NTU/single_poses/clean/y_test_fall_aug.npy')
labels_r = np.tile(labels,24)

def trim_dataset(data,label,threshold=10):
    used = np.sign(np.max(abs(data),axis=-1))
    length = np.sum(used, axis=1)
    ind = [length>=threshold]
    return ind

inds = trim_dataset(X_test,labels_r,threshold=10)
labels_r = labels_r[inds]
mean = mean[inds]
log_sigma = log_sigma[inds]
latent_params = np.concatenate([mean,log_sigma], axis=-1)
print(latent_params.shape)

feat_cols = ['layer'+str(i) for i in range(latent_params.shape[-1])]
df = pd.DataFrame(latent_params,columns=feat_cols)
rndperm = np.random.permutation(latent_params.shape[0])
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

df['y'] = labels_r
df['Action class'] = ['Fall' if x == 43 else 'Non-fall' for x in labels_r]

'''
def select_the_fall_frames(fall_df):
  #since we move in a step size of 10, we assume fall only happens after the 20th frame
  #so we only extract the examples after 10th frame.
  #but typically, the number of frames for falling is smaller than 100. This means we can exclude the 7/8th batch
  cycle = int(fall_df.shape[0]/24)
  fall_df_crop = fall_df[cycle*0:cycle*5]
  return fall_df_crop
'''
print(df.head())
non_fall_df = df[df['y']!=43].sample(50000)
fall_df = df[df['y']==43]
#fall_df_crop = select_the_fall_frames(fall_df)
#fall_df_crop = fall_df
#random_ind = list(np.random.randint(len(non_fall_df), size=10000))
new_df= pd.concat([fall_df,non_fall_df],ignore_index=True)
new_df['size']=[3 if x==43 else 1 for x in new_df['y']]

import seaborn as sns
plt.figure(figsize=(12,8))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="Action class",size='size',
    palette=sns.color_palette("hls", 2),
    data=new_df,
    legend="full",
    alpha=0.3
)
ax = plt.gca()
ax.set_title("Visualising the Past Autoencoder representation: Fall vs. Non-fall")
plt.show()
