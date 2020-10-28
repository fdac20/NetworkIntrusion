'''
I made this a separate file because we are probably going to use it in other parts of our project
'''

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def read_data(data):
    '''
    Will return arrays of these sizes:
        X_train : (752494, 42)
        y_train_label : (752494, )
        y_train_binary : ()

        X_test : (82333, 42)
        y_test_label : (82333, )
        y_train_binary : (82333, )
    '''

    # convert all but last column to list of lists for data
    X = np.array(data.iloc[:,:-2].values.tolist())
    X = StandardScaler().fit_transform(X)  # mean of ~0 and variance of 1
    # convert last column to list for labels
    y_binary = np.array(data.iloc[:,-1].values.tolist())
    y_labels = np.array(data.iloc[:,-2].values.tolist())
    return X, y_binary,y_labels

# only call this on training set
def PCA_train_test(X):
    X, pca = pca_data(X)

    np.save("../../data/X_train_PCA.npy", X)
    return X, pca

def pca_data(X):
    pca = PCA()
    X = pca.fit_transform(X)

    #get variance explained
    explained_variance = pca.explained_variance_ratio_
    
    '''
    #make first plot of just principal components
    fig1 = plt.figure()
    plt.plot(explained_variance)
    plt.title("Principal Components")
    plt.ylabel("Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("../../graphs/principal_comp_.png")
    '''

    #select what percent var to keep
    desired_var = 0.9
    #select how many eigenvalues to keep
    cumsum = np.cumsum(explained_variance)
    k = np.argwhere(cumsum > desired_var)[0]

    ''' 
    #make second plot of cum var explained
    fig2 = plt.figure()
    plt.plot(cumsum)
    plt.title("Variance Explained")
    plt.plot(k, cumsum[k], 'ro', label="Eigenvalue #%d with %.2f Variance" % (k, desired_var))
    plt.legend()
    plt.ylabel("Cumulative Percent of Variance Explained")
    plt.xlabel("Principal Component")
    plt.savefig("../../graphs/var_exp_.png")
    '''

    pca = PCA(n_components=int(k))
    X = pca.fit_transform(X)

    return X, pca
            

def main():

    # training data
    data = pd.read_csv('../../data/UNSW_train.csv')
    X, y_bin, y_lab = read_data(data)

    np.save("../../data/X_train_ORIG.npy",X)
    np.save("../../data/y_train_bin.npy",y_bin)
    np.save("../../data/y_train_labels.npy",y_lab)

    X, pca = PCA_train_test(X)
     
    # testing data
    data = pd.read_csv("../../data/UNSW_test.csv")
    X, y_bin, y_lab = read_data(data)

    np.save("../../data/X_test_ORIG.npy",X)
    np.save("../../data/y_test_bin.npy",y_bin)
    np.save("../../data/y_test_labels.npy",y_lab)

    X_test_pca = pca.fit_transform(X)
    np.save("../../data/X_test_PCA.npy", X_test_pca)

    

if __name__ == "__main__":
    main()


