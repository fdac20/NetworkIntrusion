import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import random as rand
import sys
import collections
import joblib
import seaborn as sn
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations, combinations_with_replacement
from sklearn.metrics import make_scorer, f1_score, roc_curve, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def best_models():
    model_dict = {  
                    "Naive-Bayes" : GaussianNB(),
                    "SGD_Classifier" : SGDClassifier(max_iter=1000, alpha=0.1),
                    "kNN" : KNeighborsClassifier(p=3, n_neighbors=1),
                    "MLP" : MLPClassifier(learning_rate_init=0.001, hidden_layer_sizes=(10, 10, 10), batch_size=30, alpha=0.01, activation='relu'),
                    "Random_Forest" : RandomForestClassifier(n_estimators=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', criterion='gini')
                  }

    return model_dict
 
# the x_train and x_test here are the actual train and test sets
# NOT the split train set
def get_data():
    X_train = np.load("../data/X_train_PCA.npy")
    y_train = np.load("../data/y_train_bin.npy")
    X_test = np.load("../data/X_test_PCA.npy")
    y_test = np.load("../data/y_test_bin.npy")
    return (X_train, y_train, X_test, y_test)

# test accuracy of model and plot ROC curve
def plot_ROC(data, model_dict):

    labels = ["Recon","Backdoor","DoS","Exploits","Analysis","Fuzzers","Worms","Shellcode","Generic","Normal"]

    # go through all models and plot ROC curves
    fpr_tpr_auc = []
    for k, v in model_dict.items():
        y_pred_proba = None
        model = v.fit(data[0],data[1])
        print("Predicting %s" % k)
        y_pred = model.predict(data[2])
        confusion_matrix_(data[3], y_pred, k)
        '''
        try:
            y_pred_proba = model.predict_proba(data[2])
            
        except AttributeError as e:
            print("Failed, predicting %s" % k)
            y_pred_proba = model.decision_function(data[2])
        
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]
        fpr, tpr, _ = roc_curve(labels, y_pred_proba)
        roc_auc = roc_auc_score(labels, y_pred_proba)
        fpr_tpr_auc.append([k, fpr, tpr, roc_auc])
    ROC_curve(fpr_tpr_auc)
    '''
def confusion_matrix_(y_test, y_pred, k):
    labels = np.unique(y_test)
    strlabels=[]
    attacklabels=["attack","normal"]
    '''
    for i in labels:
        if i == b'Reconnaissance':
            strlabels.append("Recon")
        else:
            strlabels.append(i.decode("utf-8"))
    '''
    # get confusion matrix
    label_weights = []
    normal_weight = 0.9
    '''   
    if clusterJobs[jobNo][2] == "bin":
        num_normal = (y_true == 0).sum() 
        for elem in y_true:
            if elem == 0:
                label_weights.append(num_normal / (len(y_true) * (1 - normal_weight)))
            else:
                label_weights.append(1)
    else:
    
    num_normal = (y_test == b"Normal").sum() 
    for elem in y_test:
        if elem == b"Normal":
            label_weights.append(num_normal / (len(y_test) * (1 - normal_weight)))
        else:
            label_weights.append(1) 
    '''
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 7))
    sn.heatmap(cm, annot=True, ax=ax, fmt='.2g', cmap='Blues', linewidths=0.25, linecolor='black')
    ax.set_xlabel('Predicted Labels', fontsize='large')
    ax.set_ylabel('True Labels', fontsize='large')
    ax.set_title("Confusion matrix for %s" % k, fontsize='large')
    ax.xaxis.set_ticklabels(attacklabels, rotation=45, fontsize='small')
    ax.yaxis.set_ticklabels(attacklabels, rotation='horizontal', fontsize='large')
    plt.savefig("../graphs/%s_confusion_matrix.png" % k)

    
def ROC_curve(fpr_tpr_auc):
    # plot ROC curve
    for d in fpr_tpr_auc:
        plt.plot(d[1], d[2], label="%s, AUC = %.4f" % (d[0], d[3]))
    plt.plot([0, 1], [0, 1], 'k--')  # random 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('Best Models ROC Curve')
    plt.legend()
    plt.savefig('../graphs/roc_curves.png')


def main():

    model_dict = best_models()
    data = get_data()
    plot_ROC(data, model_dict)


if __name__ == "__main__":
    main()