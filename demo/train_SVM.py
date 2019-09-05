## Building a SVM classifier for fall, based on VAE

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pickle
from sklearn.metrics import classification_report, confusion_matrix
#Create a svm Classifier - too slow!
#hyperparameter tuning required: kernel, C and gamma
# Set the parameters by cross-validation

dropout = False
if not dropout:
    data_path = '/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/'
else:
    data_path = '/Users/kefei/Documents/Dataset/NTU/single_poses/demo_data/classification/2D/dropout/'

z_mean_train = np.load(data_path+'z_mean_train.npy')
z_log_var_train = np.load(data_path+'z_log_var_train.npy')
z_mean_test = np.load(data_path+'z_mean_test.npy')
z_log_var_test = np.load(data_path+'z_log_var_test.npy')
target_train = np.load(data_path+'target_train.npy')
target_test = np.load(data_path+'target_test.npy')
features_train = np.concatenate([z_mean_train,z_log_var_train],axis=-1)
features_test = np.concatenate([z_mean_test,z_log_var_test],axis=-1)

print('finished loading datasets')
#Will need to clean the dataset. like save an index folder somewhere.

num_fall_train = sum(target_train==1)
print(num_fall_train)
kernel_list = ['rbf']
C_list = [10,100,1000]
gamma_list = ['scale']


n_list = [10,15,20,25,30]
repeat = 1
f1_repeat = np.zeros((len(n_list),len(C_list),repeat))
recall_repeat = np.zeros((len(n_list),len(C_list),repeat))
precision_repeat = np.zeros((len(n_list),len(C_list),repeat))

'''
for r in range(repeat):
    for i in range(len(n_list)):
        n = n_list[i]
        print('n components=',n)
        pca = PCA(n_components=n)

        num_fall_test = 0
        #want to make sure the fall examples are 'evenly' distributed
        while (num_fall_test< num_fall_train*0.1) or (num_fall_test > num_fall_train*0.3):
            features_train_cv,features_val, target_train_cv,target_val = train_test_split(features_train,target_train, test_size = 0.2)
            num_fall_test = sum(target_val==1)
            print('total number of fall examples in validation set =', num_fall_test)

        data_train = pca.fit_transform(features_train_cv)
        data_val = pca.transform(features_val)

        for j in range(len(C_list)):
            C = C_list[j]
            print('C=',C, 'n=',n, 'repeat=',r)
            clf = svm.SVC(C=C, gamma='scale',kernel = 'rbf',class_weight='balanced')
            clf.fit(data_train, target_train_cv)

            print('finish fitting')
            pred = clf.predict(data_val)
            print('pred=',pred)

            classification_report_model = classification_report(target_val,pred)
            print(confusion_matrix(target_val,pred))
            print(classification_report_model)

            f1 = f1_score(target_val, pred, average=None)[1]
            recall = recall_score(target_val, pred, average=None)[1]
            precision = precision_score(target_val, pred, average=None)[1]
            f1_repeat[i,j,r] = f1
            recall_repeat[i,j,r] = recall
            precision_repeat[i,j,r] = precision

model_path = 'Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/'
results_path = 'Users/kefei/Documents/mm_fall_detection/demo/results/2D/classification/'

np.save(results_path+'f1_cv.npy',f1_repeat)
np.save(results_path+'recall_cv.npy',recall_repeat)
np.save(results_path+'precision_cv.npy',precision_repeat)


#save the model with the best performing f1/ and recall?
f1_repeat_ave = np.mean(f1_repeat,axis=-1)
ind = np.unravel_index(np.argmax(f1_repeat_ave),f1_repeat_ave.shape)
n_optimal = n_list[ind[0]]
C_optimal = C_list[ind[1]]
print('optimal n=',n_optimal)
print('optimal n=',C_optimal)
print('optimal f1 score=',f1_repeat_ave[ind])


#retrain the model and save for testing?
pca = PCA(n_components=n_optimal)
data_train = pca.fit_transform(features_train)
data_test= pca.transform(features_test)
clf = svm.SVC(C=C_optimal, gamma='scale',kernel = 'rbf',class_weight='balanced')
clf.fit(
data_train, target_train)
pred = clf.predict(data_test)

classification_report_model = classification_report(target_test,pred)
confusion_matrix_model = confusion_matrix(target_test,pred)
print(confusion_matrix_model)
print(classification_report_model)
np.save('/Users/kefei/Documents/mm_fall_detection/demo/results/2D/classification/classification_report_test.npy', classification_report_model)
np.save('/Users/kefei/Documents/mm_fall_detection/demo/results/2D/classification/confusion_matrix_test.npy', confusion_matrix_model)

'''

pca = PCA(n_components=30)
data_train = pca.fit_transform(features_train)
data_test= pca.transform(features_test)

clf = svm.SVC(C=100, gamma='scale',kernel = 'rbf',class_weight='balanced')
clf.fit(data_train, target_train)
print('finish fitting')
pred = clf.predict(data_test)

classification_report_model = classification_report(target_test,pred)
confusion_matrix_model = confusion_matrix(target_test,pred)
print(classification_report_model)
print(confusion_matrix_model )

if dropout:
    pca_filename = "/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/pca_dropout.pkl"
else:
    pca_filename = "/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/pca.pkl"

with open(pca_filename, 'wb') as file:
    pickle.dump(pca, file)

if dropout:
    svm_filename = "/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/svm_dropout.pkl"
else:
    svm_filename = "/Users/kefei/Documents/mm_fall_detection/demo/models/2D/classification/svm.pkl"

with open(svm_filename, 'wb') as file:
    pickle.dump(clf, file)
