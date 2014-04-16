# -*- coding: utf-8 -*-
"""
Created on Wed Apr 02 11:47:58 2014

@author: gsubramanian
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split,StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import sklearn.preprocessing as pp


############################## Read data ###################################################

working_directory =r"C:\Gopi\Kaggle\skikit\\"

train_init = np.genfromtxt(open(working_directory + 'train.csv','rb'), delimiter=',')
target_init = np.genfromtxt(open(working_directory + 'trainLabels.csv','rb'), delimiter=',')
test_init = np.genfromtxt(open(working_directory + 'test.csv','rb'), delimiter=',')

############################################################################################


def dsplit(train_init,target_init):
    """Split data into train and test
    
    Args:
      train_init  : training data set
      target_init : class labels
    Returns:
      train and test split, 90% for training and 10% for testing
    
    """
    train,test,train_target,test_target = train_test_split(train_init,target_init,test_size=0.1,random_state=42)
    return train,test,train_target,test_target



def stratk(train_init,target_init):
    """Stratified split by class lables
    
    Args:
      train_init  : training data set
      target_init : class labels
    Returns:
      stratified class label split of train and test
    """
    sk = StratifiedKFold(target_init,n_folds=2)
    
    for train_index,test_index in sk:
        train = train_init[train_index]
        test  = train_init[test_index]
        train_target = target_init[train_index]
        test_target = target_init[test_index]
    return train,test,train_target,test_target


def classifier(train,test,train_target,test_target):
    """"Nearest neighbour classifier
    
    Args:
        train        : training data
        test         : test data
        train_target : training labels
        test_target  : test labels
    Returns:
        classifier object
    """
    kclass = KNeighborsClassifier(n_neighbors=10)
    kclass.fit(train,train_target)
    res = kclass.predict(train)
    
    print classification_report(train_target,res)
    
    res1 = kclass.predict(test)
    print classification_report(test_target,res1)
    return kclass
 
    
def optimalFeatures(train,target):
    """"Feature selection using svn
    
    Args:
        train        : training data
        target : training labels
    Returns:
        classifier object
    """

    sk = StratifiedKFold(target,n_folds=3)
    est = SVC(kernel='linear')
    rfecv = RFECV(est,cv=sk)
    rfecv.fit(train,target)
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    
    return rfecv

def dopca(train,train_target,test,test_init):
    """"Feature reduction principle comonent analysis
    
    Args:
        train        : training data
        test         : test data
        train_target : training labels
        test_target  : test labels
    Returns:
        train     : training data with reduced features
        test      : test data with reduced features
        test_init : Original test data, from competition
    """

    pca = PCA(n_components=13,whiten=True)

    train = pca.fit_transform(train,train_target)
    test = pca.transform(test)
    test_init =pca.transform(test_init)
    
    
    return train,test,test_init
    

def minmax(train,test):
    """"Min max normalization
    
    Args:
        train        : training data
        test         : test data
    Returns:
        train     : training data normalized
        test      : test data normalized
    """
    mmax= pp.MinMaxScaler()
    train = mmax.fit_transform(train)
    test = mmax.transform(test)
    return train,test
    
def norm(train,test,test_init):
    """"Mean vairance normalization
    
    Args:
        train        : training data
        test         : test data
    Returns:
        train     : training data normalized
        test      : test data normalized
    """

    norm = pp.Normalizer()
    train = norm.fit_transform(train)
    test = norm.transform(test)
    test_init = norm.transform(test_init)
    return train,test,test_init
 
        
   
# Split the training file to train and test.
train,test,train_target,test_target = dsplit(train_init,target_init)
#train = pp.scale(train)
#test = pp.scale(test)
#test_init = pp.scale(test_init)
#train,test ,test_init= norm(train,test,test_init)
# Feature reduction using PCA
train_pca,test_pca,test_init_pca = dopca(train,train_target,test,test_init)

# Perform classification
est = classifier(train,test,train_target,test_target)

res = est.predict(test_init)
idcol = np.arange(start=1,stop=9001)
res2 = np.column_stack((idcol,res))

# Save results of original test
np.savetxt(working_directory + 'prediction.csv',res2,fmt='%d',delimiter=",")


