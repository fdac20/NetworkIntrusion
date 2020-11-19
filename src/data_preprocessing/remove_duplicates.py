import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter

# helper function to cast items to float where ever possible
def float_cast(item):
    try:
        return float(item)
    except ValueError:
        return item

def count_plot(labels,ds):
    label_dict = {}
    for l in labels:
        # if l == 'normal' or l == 'neptune':
        #    continue
        if l not in label_dict:
            label_dict[l] = 1
        else:
            label_dict[l] += 1
    label_dict = OrderedDict(sorted(label_dict.items(), key=itemgetter(1), reverse=True))
     
    fig = plt.figure(figsize=(100, 100))
    plt.bar(label_dict.keys(), label_dict.values())
    plt.xticks(rotation=45, fontsize=125,ha="right")
    plt.yticks(fontsize=125)
    # plt.title("Distribution of Attacks Without Neptune")
    # plt.savefig("../graphs/attacks_without_neptune.png")
    # plt.title("Distribution of Attacks Only")
    # plt.savefig("../graphs/attacks_only.png")
    plt.title("Distribution of Samples", fontsize=125)
    plt.savefig("../../graphs/label_distribution_%s.png" % ds, bbox='tight_layout')
    

def main():

    # converts every row in the train data to a string, then adds it to a set
    # if the same string row already exists in the set, then it will not be added
    dataset=["UNSW_NB15_testing-set.csv","UNSW_NB15_training-set.csv"]
    for i in dataset:
        f=open("../../data/"+i,"r")
        line_set={"0"}
        line=""
        while True:
            line=f.readline()
            if line =="":
                break
            if "attack_cat" in line:
                continue
            line=line[:-1]  #YO I THINK THIS SHOULD BE -1 NOT -2!!*******
            line_set.add(line)
        line_set.remove("0")
        
        # goes through the unduplicated string set now and splits the data by comma
        # converts to float where possible
        list_data = []
        for line in line_set:
            line = line.split(',')
            line = [float_cast(l) for l in line]
            list_data.append(line)

        # changes all of the categorical variables into ordinally encoded variables
        # saves this result into csv file to be used for predictions
        list_data = pd.DataFrame(list_data)
        list_data = list_data.iloc[:,1:]

        if "test" in i:
            count_plot(list_data.iloc[:,-2].values.tolist(),"test") #MAKE LABELS BIGGER!! TODO
        else:
            count_plot(list_data.iloc[:,-2].values.tolist(),"train") 

        enc = OrdinalEncoder()
        list_data.iloc[:,1:4] = enc.fit_transform(list_data.iloc[:,1:4])
        if "test" in i:
            list_data.to_csv("../../data/UNSW_test.csv",index=False)  
        elif "train" in i:
            list_data.to_csv("../../data/UNSW_train.csv",index=False)  

if __name__ == "__main__":
    exit(main())
