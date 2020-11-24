# Network Intrusion Detection Project

## Authors: Charles Rizzo, Nick Skuda, Austin Saporito

### Final Project Report

The final project reports is entitled CS545_Final_Report.pdf and is in the top level of this repo.

### Workflow

1. Add the data sets **UNSW_NB15_training-set.csv** and **UNSW_NB15_testing-set.csv** to the folder `data/`

2. Run `python3 remove_duplicates.py` for both the aformentioned data sets (manually go into main and change the argument -- yes, I know this could have been done better)

3. You should see **UNSW_test.csv** and **UNSW_train.csv** in the `data/` directory now

4. Run `python3 preprocess_data.py` to generate all of the numpy files in the `data/` dir 

5. All of those .npy files collectively represent 4 versions of the data set: normal_binary (attack vs. normal), normal_labeled (normal vs. 1 of 9 attack labels), 
   PCA_binary (attack vs. normal), and PCA_labeled (normal vs. 1 of 9 attack labels)

#### Notes
- So with PCA, we decided to retain 90% variance in the data set, which reduced our amount of features from 44 to 16, which massively reduces the complexity of the data set. It will be interesting to see what the performance hit for providing less information is.

- We used an OrdinalEncoder to transform the 3 features that were categorical to numbers instead of strings

- We also standardized each column in the data set such that the mean was 0 with a variance of 1. Just seemed like the right thing to do.
