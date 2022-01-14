#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 09:37:15 2021

@author: moumen
"""

# PART 1: Basic steps

from sklearn import tree
from matplotlib import pyplot as plt # for a good visualization of the trees 

# The following is a basic example for binary classification

# X is the training set 
# Each example in X has 4 binary features
X = [[0, 0, 1, 0], [0, 1, 0, 1] , [1, 1, 0, 0] , [1, 0, 1, 1] , [0, 0, 0, 1] , [1, 1, 1, 0]]

# Y is the classes associated with the training set. 
# For instance the label of the first and second example is 1; of the third example is 0, etc
Y = [1, 1, 0, 0, 1, 1]

# We construct a decision tree using the default parameters:

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# Now we can ask the decision tree to predict the outcome for unknown examples. For instance we can ask a prediction for the three examples:

clf.predict([[1,1,1,1] , [0,1,0,0] , [1,1,0,1] ])

# The result is an array of the 3 predicted labels (one for each example): array([0, 1, 0])

# PART 2 : Visualization

print("Prediction",clf.predict([[0,0,1,1] , [1,1,1,1] , [0,1,1,1] ]))

# There are many ways to visualize a decision tree. The first one is very basic:

text_representation = tree.export_text(clf)
print(text_representation)

# We can use a more readable and visual way as follows:

fig = plt.figure(figsize=(10,7))
_ = tree.plot_tree(clf, 
                   feature_names= ("Big_Size","Carnivore" , "Reproduction", "Solitary"),
                   class_names= ("Not_Extinct", "Extinct" ), 
                   filled=True)

# Where:

#     figsize restrains the size of the plot,
#     feature_names gives the names of the different features,
#     class_names corresponds to human readable labels for each class,
#     filled is a boolean indicating a preference to show a colorful tree.




# # PART 3: The compass dataset


# # Tasks

# #     What are the features?  A: Person_ID,AssessmentID,Case_ID,Agency_Text,LastName
# #     How many examples in the dataset?
# #     What are your expectations regarding the most importat features?
# #     Propose (informally) a way to reduce the dataset
# #     There many ways to binarize the dataset. How to you propose to do so?


import csv
import numpy as np
from utils import load_from_csv

train_examples, train_labels, features, prediction = load_from_csv("./compass.csv")

# Build severals decision trees (different parameters) and visualize them

# Decision tree 1

clf = tree.DecisionTreeClassifier(splitter='best',max_depth=(6),min_samples_leaf=1)
clf = clf.fit(X, Y)

fig = plt.figure(figsize=(10,7))
_ = tree.plot_tree(clf, 
                    feature_names= ("Big_Size","Carnivore" , "Reproduction", "Solitary"),
                    class_names= ("Not_Extinct", "Extinct" ), 
                    filled=True)

# Decision tree 2

clf = tree.DecisionTreeClassifier(splitter='best',max_depth=(6),min_samples_leaf=30)
clf = clf.fit(X, Y)

fig = plt.figure(figsize=(10,7))
_ = tree.plot_tree(clf, 
                    feature_names= ("Big_Size","Carnivore" , "Reproduction", "Solitary"),
                    class_names= ("Not_Extinct", "Extinct" ), 
                    filled=True)


# Decision tree 3

clf = tree.DecisionTreeClassifier(splitter='best',max_depth=(6),min_samples_leaf=500)
clf = clf.fit(X, Y)

fig = plt.figure(figsize=(10,7))
_ = tree.plot_tree(clf, 
                    feature_names= ("Big_Size","Carnivore" , "Reproduction", "Solitary"),
                    class_names= ("Not_Extinct", "Extinct" ), 
                    filled=True)