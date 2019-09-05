## Building a SVM classifier for fall

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
#Create a svm Classifier
#hyperparameter tuning required: kernel, C and gamma
# Set the parameters by cross-validation

hidden_train = np.load('Users/kefei/Documents/Dataset/NTU/single_poses/classification/hidden_train.npy')
#hidden_test =
target_train = np.load('Users/kefei/Documents/Dataset/NTU/single_poses/classification/target_train.npy')
#target_test =

#PERFORM CROSS VALIDATION OF SVM
#https://www.kaggle.com/udaysa/svm-with-scikit-learn-svm-with-parameter-tuning
svc = svm.SVC(gamma="scale")
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 1, 10, 25, 50, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(svc, tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(hidden_train, target_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


'''
def SVM_classifier(X_train, y_train,X_test,kernel=None):

    if not kernel:
        kernel='linear'

    clf = svm.SVC(kernel=kernel) # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    return clf, y_pred

'''

## Building a FC neural network for fall






## Building a lstm + FC neural network for fall? maybe consider {h_t, h_t+1,...}
