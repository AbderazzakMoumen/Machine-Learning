#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:18:06 2021

@author: moumen
"""


import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix,ConfusionMatrixDisplay
# fetch dataset from openml (might take some time)
import timeit
mnist = fetch_openml('mnist_784', as_frame=False)

"""
Exercice 1 - One hidden layer classifier

"""

# Selection of 5000 images among the 70000 available.
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

#On a importé des données 
# print(mnist)
# print (mnist.data)
# print (mnist.target)
len(mnist.data)
# help(len)   
# print (mnist.data.shape)
# print (mnist.target.shape)
mnist.data[0]
mnist.data[0][1]
mnist.data[:,1]
mnist.data[:100]


#On visualise quelques données 
images = mnist.data.reshape((-1, 28, 28))
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")
# Splitting the data base:
# Split the data base in 80% for training and 20% for the test (see TP1).

x_train, x_test, y_train, y_test = train_test_split(data, target,random_state=0, test_size = 0.20)


# # Build a classification model with the following parameter: hidden_layer_sizes = (50), 
# # then calculate the precision of the classifier

# mlp= MLPClassifier(hidden_layer_sizes=(50), )#number of neurons in hidden layer, here one single hidden layer with 20 neurons,

# # activation='relu', # activation function start with logistic function

# # solver='adam', #solver

# # alpha=0.0001,# L2 penalty (regularization term) parameter

# # batch_size='auto', #Size of minibatches for stochastic optimizers.

# # learning_rate='adaptive', #Learning rate schedule for weight updates

# # learning_rate_init=1e-50, #The initial learning rate used

# # max_iter=300, #Maximum number of iterations

# # tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.

# # verbose=True, # To Print progress messages during learning step

# # warm_start=False, #When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution

# # early_stopping=True, #Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver=’sgd’ or ‘adam’.

# # validation_fraction=0.1, #The proportion of training data to set aside as validation set for early stopping

# # n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.


# start = timeit.default_timer() #calculate program run time in python
# mlp.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage

# ScoreTest = mlp.score(x_test, y_test)  #computes the global score for test data

# ScoreTrain = mlp.score(x_train, y_train) #computes the global score for training data



# print('Score test', ScoreTest)
# print('Score train', ScoreTrain)



# # Display the class of image 4 and its predicted class.

# Prediction_test = mlp.predict(x_test)  #is used to predict values for test data

# Prediction_train = mlp.predict(x_train)  #is used to predict values for test data

# print("Predicted class of image 4 :", Prediction_test[3])
# print("Actual class of image 4 :", y_test[3])



# #On visualise Actual class of image 4
# images = x_test.reshape((-1, 28, 28))
# plt.imshow(images[3],cmap=plt.cm.gray_r,interpolation="nearest")


# # Compute the precision for learning and test/predicted values using the package metrics 

# Precision_score=precision_score(y_test,Prediction_test,average='micro')
# print("Precision du score :", Precision_score)

# Accuracy_score=accuracy_score(y_test,Prediction_test)
# print("Accuracy du score :", Accuracy_score)

# stop = timeit.default_timer()

# print('Time: ', stop - start) 



# # Compute and print the confusion matrix using confusion_matrix(ytest,preds_test), and comment on the result.
# cm = confusion_matrix(y_test, Prediction_test)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
# disp.plot()



#Afin de ne pas refaire chaque fois le même code, pour la modification 
# du nombre d'échantillons, le nombre d'itérations maximal ainsi que le 
# nombre de neurones, j’ai modifié directement les paramètres dans le code 
# ci-dessus et relancé à chaque fois la simulation 


"""
Exercice 2 - Multiple hidden layer classifier

"""

# Model 1 : (50)
# Model 2 : (50,50)
# Model 3 : (50,50,50)
# Model 4 :(50,50,50,50)
# Model 5 : (50,50,50,50,50)

mlp1= MLPClassifier(hidden_layer_sizes=(50))
mlp1.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage

Prediction_test1 = mlp1.predict(x_test)  #is used to predict values for test data

# Compute the precision for learning and test/predicted values using the package metrics 

Precision_score1=precision_score(y_test,Prediction_test1,average='micro')
print("Precision du score :", Precision_score1)

Accuracy_score1=accuracy_score(y_test,Prediction_test1)
print("Accuracy du score :", Accuracy_score1)

# Compute and print the confusion matrix using confusion_matrix(ytest,preds_test), and comment on the result.
cm1 = confusion_matrix(y_test, Prediction_test1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=mlp1.classes_)
disp.plot()






mlp2= MLPClassifier(hidden_layer_sizes=(50,50))
mlp2.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage

Prediction_test2 = mlp2.predict(x_test)  #is used to predict values for test data

# Compute the precision for learning and test/predicted values using the package metrics 

Precision_score2=precision_score(y_test,Prediction_test2,average='micro')
print("Precision du score :", Precision_score2)

Accuracy_score2=accuracy_score(y_test,Prediction_test2)
print("Accuracy du score :", Accuracy_score2)

# Compute and print the confusion matrix using confusion_matrix(ytest,preds_test), and comment on the result.
cm2 = confusion_matrix(y_test, Prediction_test2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=mlp2.classes_)
disp.plot()


mlp3= MLPClassifier(hidden_layer_sizes=(50,50,50))
mlp3.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage

Prediction_test3 = mlp3.predict(x_test)  #is used to predict values for test data

# Compute the precision for learning and test/predicted values using the package metrics 

Precision_score3=precision_score(y_test,Prediction_test3,average='micro')
print("Precision du score :", Precision_score3)

Accuracy_score3=accuracy_score(y_test,Prediction_test3)
print("Accuracy du score :", Accuracy_score2)

# Compute and print the confusion matrix using confusion_matrix(ytest,preds_test), and comment on the result.
cm3 = confusion_matrix(y_test, Prediction_test3)
disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=mlp3.classes_)
disp.plot()


mlp4= MLPClassifier(hidden_layer_sizes=(50,50,50,50))
mlp4.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage

Prediction_test4= mlp4.predict(x_test)  #is used to predict values for test data

# Compute the precision for learning and test/predicted values using the package metrics 

Precision_score4=precision_score(y_test,Prediction_test4,average='micro')
print("Precision du score :", Precision_score4)

Accuracy_score4=accuracy_score(y_test,Prediction_test4)
print("Accuracy du score :", Accuracy_score4)

# Compute and print the confusion matrix using confusion_matrix(ytest,preds_test), and comment on the result.
cm4 = confusion_matrix(y_test, Prediction_test4)
disp = ConfusionMatrixDisplay(confusion_matrix=cm4, display_labels=mlp4.classes_)
disp.plot()



mlp5= MLPClassifier(hidden_layer_sizes=(50,50,50,50,50))
mlp5.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage

Prediction_test5= mlp5.predict(x_test)  #is used to predict values for test data

# Compute the precision for learning and test/predicted values using the package metrics 

Precision_score5=precision_score(y_test,Prediction_test5,average='micro')
print("Precision du score :", Precision_score5)

Accuracy_score5=accuracy_score(y_test,Prediction_test5)
print("Accuracy du score :", Accuracy_score4)

# Compute and print the confusion matrix using confusion_matrix(ytest,preds_test), and comment on the result.
cm5 = confusion_matrix(y_test, Prediction_test4)
disp = ConfusionMatrixDisplay(confusion_matrix=cm5, display_labels=mlp5.classes_)
disp.plot()

plt.figure(3)
plt.plot(mlp1.loss_curve_,'r', label='LOSS CURVE 1 hidden layer')
plt.plot(mlp2.loss_curve_, 'g', label='LOSS CURVE 2 hidden layers')
plt.plot(mlp3.loss_curve_, 'b', label='LOSS CURVE 3 hidden layers')
plt.plot(mlp4.loss_curve_, 'k', label='LOSS CURVE 4 hidden layers')
plt.plot(mlp5.loss_curve_, 'c', label='LOSS CURVE 5 hidden layers')
plt.legend()
plt.title('The loss curve as function of epochs (iteration)')



#Afin de ne pas refaire chaque fois le même code, pour la modification 
# des differentes parametres pour realiser la suite du tp
 # j’ai modifié directement les paramètres dans le code 
# ci-dessus et relancé à chaque fois la simulation 