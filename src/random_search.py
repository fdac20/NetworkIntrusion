import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import random as rand
import sys
import collections
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from itertools import combinations, combinations_with_replacement
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler 



#Nasty lil dictionary used to determine which node will run which parmiganna combination
clusterJobs = {1:('kNN',"ORIG","bin"), 2:('kNN',"PCA","bin"), 3:('kNN',"ORIG","labels"), 4:('kNN',"PCA","labels"),
        5:('MLP',"ORIG","bin"), 6:('MLP',"PCA","bin"), 7:('MLP',"ORIG","labels"), 8:('MLP',"PCA","labels"),
        9:('SGD_Classifier',"ORIG","bin"), 10:('SGD_Classifier',"PCA","bin"), 11:('SGD_Classifier',"ORIG","labels"), 12:('SGD_Classifier',"PCA","labels"),
        13:('Random_Forest',"ORIG","bin"), 14:('Random_Forest',"PCA","bin"), 15:('Random_Forest',"ORIG","labels"), 16:('Random_Forest',"PCA","labels"),
        17:('Naive-Bayes',"ORIG","bin"), 18:('Naive-Bayes',"PCA","bin"), 19:('Naive-Bayes',"ORIG","labels"), 20:('Naive-Bayes',"PCA","labels")}

jobNo = 0

#function to read in the CSV files
def read_npy():
    '''
    I chose to preprocess the data and write it to .npy files so that I wouldn't have
    to do so while this program runs.
    '''

    #split data into predictions and predictors: 4 sets (normal,labels),(normal,binary), (PCA,labels),(PCA,binary)


    X = np.load("../data/X_train_%s.npy" % clusterJobs[jobNo][1]) 
    y = np.load("../data/y_train_%s.npy" % clusterJobs[jobNo][2])


     
    return X, y

def createSkeletonCSV():
    row_names = []

    for k,v in clusterJobs.items():
        name = "Best %s %s %s" % (v[0], v[1],v[2])
        row_names.append(name)

    col_names = ["Best Params", "F1-Score", "Refit Time"]

    df = pd.DataFrame(None, index=[row_names], columns=col_names)
    df.to_csv("../saved_models/Random_Search_Info.csv")

def get_params(algorithm):
    '''
    algorithm : name of the algorithm to get params for

    returns dict of params
    '''
    if algorithm == "kNN":
        return { 'n_neighbors' : [1,11,25,69],
                 'p' : [1, 2, 3] } #different orders of minkowski distance. 1=manhattan, 2=euclidean
    elif algorithm == "MLP": 
        hidden_layers = MLP_structure()
        return { 'hidden_layer_sizes' : hidden_layers,
                 'alpha' : [0.01, 1, 5, 10],
                 'learning_rate_init' : [0.001, 0.01, 0.1, 1, 5],
                 'batch_size' : [1, 10, 30, 200],
                 'activation' : ['logistic', 'relu', 'tanh']}
    elif algorithm == "SGD_Classifier":
        return { 'max_iter' : [100,1000,10000],
                 'alpha' : [0.0001,0.001,0.01,0.1] }
    elif algorithm == "Random_Forest":
        return { 'n_estimators' : [10, 50, 100, 200],
                 'criterion' : ["gini","entropy"],
                 'min_samples_split' : [2, 4, 6, 8],
                 'min_samples_leaf' : [1, 2, 3, 4],
                 'max_features' : ["auto", "sqrt", "log2"] }
    elif algorithm == "Naive-Bayes":
        return { }


#this is just to handle exceptions
def custom_scorer(y_true, y_pred):
    score = np.nan

    #keep this identical to original scoring method from sklearn
    #The following if-else allows us to weight each label; since there are more normal than attack samples, we weight the normal samples less than the attacks
    label_weights = []
    normal_weight = 0.9
    
    if clusterJobs[jobNo][2] == "bin":
        num_normal = (y_true == 0).sum() 
        for elem in y_true:
            if elem == 0:
                label_weights.append(num_normal / (len(y_true) * (1 - normal_weight)))
            else:
                label_weights.append(1)
    else:
        num_normal = (y_true == "Normal").sum() 
        for elem in y_true:
            if elem == "Normal":
                label_weights.append(num_normal / (len(y_true) * (1 - normal_weight)))
            else:
                label_weights.append(1) 
    
    try:
        score = f1_score(y_true, y_pred, average='weighted',sample_weight=label_weights)
    except Exception:
        pass
    return score

#for hidden layer/hidden neuron combos in MLP
def MLP_structure():

    hidden_layers = [1, 3, 5]
    hidden_neurons = [5, 10, 20]
    structure = []

    for layer in hidden_layers:
        neuron_layer = list(combinations_with_replacement(hidden_neurons, layer))
        structure += tuple(neuron_layer)
    return structure

def random_search_(algorithm, params, X, y, iters=20, jobs=5):
    '''
    Testing the following algs: 

        kNN, BPNN/MLP, Decision Tree, Random Forest, SVM
    '''
    clf = None
    if algorithm == "kNN":
        clf = KNeighborsClassifier()
    elif algorithm == "MLP":
        #closest to what we did in class
        clf = MLPClassifier(solver="sgd")
    elif algorithm == "Decision_Tree":
        clf = DecisionTreeClassifier()
    elif algorithm == "SGD_Classifier":
        clf = SGDClassifier()
    elif algorithm == "Random_Forest":
        clf = RandomForestClassifier()
    elif algorithm == "Naive-Bayes":
        clf = GaussianNB()
        
    custom_neg_MSLE = make_scorer(custom_scorer)
    random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=iters, n_jobs=jobs, 
                                       scoring=custom_neg_MSLE, refit=True, verbose=2,cv=10)

    random_search.fit(X, y)
#    report(random_search.cv_results_)
    
    #save the model
    best_estimator = random_search.best_estimator_
    joblib.dump(best_estimator, "../saved_models/best_%s_%s_%s.joblib" % (clusterJobs[jobNo][0],clusterJobs[jobNo][1],clusterJobs[jobNo][2]))

    #write info about the model
    info = pd.read_csv("../saved_models/Random_Search_Info.csv", index_col=0)
    best_params = random_search.best_params_
    fit_time = random_search.refit_time_ 
    best_score = random_search.best_score_
    info.loc["Best %s %s %s" % (clusterJobs[jobNo][0],clusterJobs[jobNo][1],clusterJobs[jobNo][2]), "Best Params"] = str(best_params)
    info.loc["Best %s %s %s" % (clusterJobs[jobNo][0],clusterJobs[jobNo][1],clusterJobs[jobNo][2]), "F1-Score"] = "%.4f" % best_score
    info.loc["Best %s %s %s" % (clusterJobs[jobNo][0],clusterJobs[jobNo][1],clusterJobs[jobNo][2]), "Refit Time"] = "%.6f" % fit_time 
    print(info)
    info.to_csv("../saved_models/Random_Search_Info.csv")


#stolen shamelessly off the internet
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))

            #except this part, this is the F1-Score score
            print("F1-Score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == "__main__":
 
   # print(sys.argv)
    jobNo = int(sys.argv[1])     #For cluster

    #read data from one of 6 datasets
    X, y = read_npy()

    if not os.path.isfile("../saved_models/Random_Search_Info.csv"):
        createSkeletonCSV()

    '''
    algorithm : string
                - kNN
                - MLP
                - SGD_Classifier  
                - Random_Forest
                - Naive-Bayes
    '''
    algorithm = clusterJobs[jobNo][0]
    
    #this is where the params to test are stored
    param_dict = get_params(algorithm)

    #this where the actual searching happens
    random_search_(algorithm, param_dict, X, y, iters=1, jobs=5) #100, 30

