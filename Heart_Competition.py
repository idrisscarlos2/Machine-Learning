#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:08:08 2019

@author: tsayem
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

'''
IMPORT AND CLEAN DATA
'''
X = pd.read_csv("/users/nbc/tsayem/Documents/Data_Science_Spe/DrivenData/Heart_Competition_2019/train_values.csv", index_col='patient_id')
y = pd.read_csv("/users/nbc/tsayem/Documents/Data_Science_Spe/DrivenData/Heart_Competition_2019/train_labels.csv", index_col='patient_id')
X_unknown = pd.read_csv("/users/nbc/tsayem/Documents/Data_Science_Spe/DrivenData/Heart_Competition_2019/test_values.csv", index_col='patient_id')
# keep the patient identity in the unknown data frame 
unknown_index = X_unknown.index
thalium_codes = {'thal': {'normal':0, 'fixed_defect':1, 'reversible_defect':2}}
X.replace(thalium_codes, inplace=True) #coding categorical variables; unique represents the coding scheme to be checked for interpretation
X_unknown.replace(thalium_codes, inplace=True)

X, y, X_unknown = np.asarray(X), np.asarray(y), np.asarray(X_unknown)   #convert data to arrays
X = preprocessing.StandardScaler().fit(X).transform(X)  #normalize data
X_unknown = preprocessing.StandardScaler().fit(X_unknown).transform(X_unknown)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=4) #split data into train and test 

'''
Building the model for different c values and solver values
'''
#c_values = np.linspace(0.42,0.48,500)
#solvers = ['newton-cg','lbfgs','liblinear', 'sag','saga']
#list_accuracies = []
#
#for sol in solvers:
#    accuracy = []
#    for cv in c_values:
#        LR = LogisticRegression(C=cv, solver=sol).fit(X_train, y_train)
#        yhat_proba = LR.predict_proba(X_test)
#        accuracy.append(log_loss(y_test, yhat_proba))
#    list_accuracies.append(accuracy)
#  
##    
##fig, ax = plt.subplots(5, figsize=(15,10))
##fig.suptitle("subplots according to C values and solvers")
##ax[0].plot(c_values, list_accuracies[0])
##ax[1].plot(c_values, list_accuracies[1])
##ax[2].plot(c_values, list_accuracies[2])
##ax[3].plot(c_values, list_accuracies[3])
##ax[4].plot(c_values, list_accuracies[4])
#
#'''
#LOOKING FOR THE MINIMUN IN ALL LIST_ACCURACIES
#'''
#min_indexes = []
#min_log_losses =[]
#for i in range(5):
#    min_log_losses.append(min(list_accuracies[i]))
#    min_indexes.append(list_accuracies[i].index(min(list_accuracies[i])))
#
#min_cv = []
#for i in min_indexes:
#    min_cv.append (c_values[i])
#    
#'''
#BEST PARAMETERS ARE C = 0.45192193192019076, AND solver = 'liblinear'
#'''

LR = LogisticRegression(C= 0.42, solver='liblinear').fit(X, y)
y_unknown_proba = LR.predict_proba(X_unknown)

df_final = pd.DataFrame({'patient_id':unknown_index.to_numpy(), 'heart_disease_present': y_unknown_proba[:,1]})
df_final.set_index('patient_id', inplace=True)
df_final.to_csv('/users/nbc/tsayem/Documents/Data_Science_Spe/DrivenData/Heart_Competition_2019/Heart_competion.csv')