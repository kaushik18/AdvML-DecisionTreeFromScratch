####################################################################################
# Name: Kaushik Nadimpalli
# Course: CS6375.002
# Assignment 2: SciKit-Learn Decision Tree Implementation
# Purpose of this file: In this file, we are utilizing the traiditional
# decision tree classifier from SciKit-Learn to build our trees.
####################################################################################

import numpy as np
import pandas as pd
import graphviz
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix

# Monk 1  Train
M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytrn = M[:, 0]
Xtrn = M[:, 1:]

# Monk 1 Test
M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
ytst = M[:, 0]
Xtst = M[:, 1:]

d_tree = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5)
d_tree = d_tree.fit(Xtrn, ytrn)
y_pred = d_tree.predict(Xtst)
print(confusion_matrix(ytst, y_pred))


"""
# Part E - Anotehr Dataset - Hayes Roth (5 attributes and 160 instances)
# https://archive.ics.uci.edu/ml/datasets/Hayes-Roth
M = pd.read_csv('hayes-roth.data',delimiter=',',header=None,dtype=int)
T = pd.read_csv('hayes-roth.test',delimiter=',',header=None,dtype=int)

Xtrn = M.iloc[:, 1:-1].values
ytrn = M.iloc[:, -1].values
Xtst = T.iloc[:,:-1].values
ytst = T.iloc[:,-1].values

d_tree = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5)
d_tree = d_tree.fit(Xtrn, ytrn)
y_pred = d_tree.predict(Xtst)
print(confusion_matrix(ytst, y_pred))
"""
