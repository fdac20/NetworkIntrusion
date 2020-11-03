import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random as rand
import sys
import joblib
import matplotlib.pyplot as plt
import re
from sklearn.metrics import f1_score, classification_report
import time

labels = ["Reconnaissance","Backdoor","DoS","Exploits","Analysis","Fuzzers","Worms","Shellcode","Generic"]

def load_test_data(data_set,pred_set):

    X = np.load("../data/X_test_%s.npy" % data_set) 
    y = np.load("../data/y_test_%s.npy" % pred_set)

    return X, y

def eval(y_test, y_pred,pred_set):
    overall_f1_score = f1_score(y_test, y_pred, average='weighted')
    if pred_set == "labels":
        report = classification_report(y_test,y_pred,labels=labels,output_dict=True) #labels = test_labels
    else:
        report = classification_report(y_test,y_pred,labels=[0,1],output_dict=True) 
    if("accuracy" in report.keys()):
        report.pop("accuracy")

    return overall_f1_score, report

#to make the skeleton
def make_report():

    algorithms = ["kNN", "MLP", "SGD_Classifier", "Random_Forest","Naive-Bayes"]
    row_names = []
    for a in algorithms:
        for ds in ["ORIG","PCA"]:
            for ps in ["labels","bin"]:
                name = "Best %s %s %s" % (a,ds,ps)
                row_names.append(name)

    col_names = ["PREDICTION TIME","OVERALL F1-SCORE"]
    for attack in labels:
        for val in ["precision","recall","f1-score","support"]:
            col_names.append("Overall_%s_%s" %(attack,val))
    for avg in ["micro avg", "macro avg","weighted avg"]:
        for val in ["precision","recall","f1-score","support"]: 
            col_names.append("Overall_%s_%s" %(avg,val)) 

    df = pd.DataFrame(None, index=[row_names], columns=col_names)
    return(df)

#DO NOT RUN THIS AGAIN OR YOU'LL BE SORRY
def test_best_algs():
        
    algorithms = ["kNN", "MLP", "SGD_Classifier", "Random_Forest","Naive-Bayes"]
    report = make_report()


    models_path = "../saved_models/"
    for root, dirs, files in os.walk(models_path):
        for name in files:
            if name.endswith(".joblib"):
                fields = name.split('_')
                alg_name = fields[1]
                if alg_name == "SGD":
                    alg_name += "_Classifier"
                    data_set = fields[3]
                    pred_set = fields[4][:-7]   
                else:
                    data_set = fields[2]
                    pred_set = fields[3][:-7]

                clf = joblib.load(models_path + name) 
                X, y = load_test_data(data_set,pred_set)
                print("Loaded %s%s" %(models_path,name))

                start_time = time.time()
                y_pred = clf.predict(X)
                end_time = time.time() - start_time
                overall_f1_score, overall_report = eval(y, y_pred,pred_set)
                report.loc["Best %s %s %s" % (alg_name,data_set,pred_set), "OVERALL F1-SCORE"] = "%.6f" % overall_f1_score
                report.loc["Best %s %s %s" % (alg_name,data_set,pred_set), "PREDICTION TIME"] = "%.10f" % end_time
                for k1,v1 in overall_report.items():
                    for k2,v2 in v1.items():
                            report.loc["Best %s %s %s" % (alg_name,data_set,pred_set),"Overall_%s_%s" % (k1,k2)] = "%.6f" % v2

    report.to_csv("../saved_models/Best_Model_Info.csv")
#################################    

if __name__ == "__main__":
   
    test_best_algs()

