
import numpy as np
from sklearn.neural_network import MLPRegressor
import timeit
import matplotlib.pyplot as plt

"""
#EXERCICE 1

"""

# Importing the dataset

dataset = np.genfromtxt("yacht_hydrodynamics.data", delimiter='')

X = dataset[:, :-1]

y = dataset[:, -1] 

# Splitting the dataset into the Training set and Test set (here 20% of data for test)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y,random_state=0, test_size = 0.20)


# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)


# MPL regressor for one single hidden layer with 5 neurons, with activation “logistics”

mlp= MLPRegressor(hidden_layer_sizes=(5), #number of neurons in hidden layer, here one single hidden layer with 20 neurons,

activation='logistic', # activation function start with logistic function

solver='adam', #solver

alpha=0.01,# L2 penalty (regularization term) parameter

batch_size='auto', #Size of minibatches for stochastic optimizers.

learning_rate='adaptive', #Learning rate schedule for weight updates

learning_rate_init=0.01, #The initial learning rate used

max_iter=1000, #Maximum number of iterations

tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.

verbose=True, # To Print progress messages during learning step

warm_start=False, #When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution

early_stopping=True, #Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver=’sgd’ or ‘adam’.

validation_fraction=0.1, #The proportion of training data to set aside as validation set for early stopping

n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.


#Training set score, the test set score and the run time for training

start = timeit.default_timer() #calculate program run time in python

mlp.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage

stop = timeit.default_timer()

print('Time: ', stop - start) 

Prediction_test = mlp.predict(x_test)  #is used to predict values for test data

Prediction_train = mlp.predict(x_train)  #is used to predict values for test data


a = mlp.score(x_test, y_test)  #computes the global score for test data

b = mlp.score(x_train, y_train) #computes the global score for training data



print('Score test', a)
print('Score train', b)

# Plot Predicted values versus Target values in both cases train and test.

plt.figure(1)
plt.subplot(211)

plt.plot(Prediction_test, 'rx', label='Predicted values')

plt.plot(y_test, 'gx', label='Target values test')

plt.title('Predicted test values versus Target values test case ')

plt.legend()

plt.subplot(212)

plt.plot(Prediction_train, 'rx', label='Predicted values')

plt.plot(y_train, 'gx', label='Target values train')

plt.title('Predicted train values versus Target values train case ')

plt.legend()

plt.show()

# Plot the loss curve as function of epochs (iteration)

plt.figure(2)

plt.plot(mlp.loss_curve_)

plt.title('The loss curve as function of epochs (iteration)')

plt.show()


# Increase the number of hidden neurons from 5 to 100 (5, 10, 20, 50, 70 and 100).

def HIDDEN_NEURONS(NUMBER):
    
    mlp= MLPRegressor(hidden_layer_sizes=(NUMBER), #number of neurons in hidden layer, here one single hidden layer with 20 neurons,

    activation='logistic', # activation function start with logistic function
    
    solver='adam', #solver
    
    alpha=0.01,# L2 penalty (regularization term) parameter
    
    batch_size='auto', #Size of minibatches for stochastic optimizers.
    
    learning_rate='adaptive', #Learning rate schedule for weight updates
    
    learning_rate_init=0.01, #The initial learning rate used
    
    max_iter=1000, #Maximum number of iterations
    
    tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
    
    verbose=True, # To Print progress messages during learning step
    
    warm_start=False, #When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution
    
    early_stopping=True, #Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver=’sgd’ or ‘adam’.
    
    validation_fraction=0.1, #The proportion of training data to set aside as validation set for early stopping
    
    n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.
    
    return mlp

Score_TEST = [] 
Score_TRAIN = [] 
NEURONS = [5,10,20,50,70,100]
RUN_TIME = [] 

for i in NEURONS: 
    c = HIDDEN_NEURONS(i)
    start = timeit.default_timer() #calculate program run time in python
    c.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage
    stop = timeit.default_timer()
    time = stop - start
    RUN_TIME.append(time)
    a = c.score(x_test, y_test)  #computes the global score for test data
    b = c.score(x_train, y_train) #computes the global score for training data
    Score_TEST.append(a)
    Score_TRAIN.append(b) 

# Plot on the same graph the training set score and the test set score versus number of hidden neurons.

plt.figure(1)
plt.plot(NEURONS,Score_TEST, 'r', label='SCORE TEST')
plt.plot(NEURONS,Score_TRAIN, 'g', label='SCORE TRAIN')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('SCORES Logistic')

plt.figure(2)
plt.plot(NEURONS,RUN_TIME, 'b', label='RUN TIME')
plt.xlabel('NEURONS')
plt.ylabel('RUN TIME Logistic')
    

plt.show()

# Using relu activation function


def HIDDEN_NEURONS2(NUMBER):
    
    mlp= MLPRegressor(hidden_layer_sizes=(NUMBER), #number of neurons in hidden layer, here one single hidden layer with 20 neurons,

    activation='relu', # activation function start with logistic function
    
    solver='adam', #solver
    
    alpha=0.01,# L2 penalty (regularization term) parameter
    
    batch_size='auto', #Size of minibatches for stochastic optimizers.
    
    learning_rate='adaptive', #Learning rate schedule for weight updates
    
    learning_rate_init=0.01, #The initial learning rate used
    
    max_iter=1000, #Maximum number of iterations
    
    tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
    
    verbose=True, # To Print progress messages during learning step
    
    warm_start=False, #When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution
    
    early_stopping=True, #Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver=’sgd’ or ‘adam’.
    
    validation_fraction=0.1, #The proportion of training data to set aside as validation set for early stopping
    
    n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.
    
    return mlp

Score_TEST1 = [] 
Score_TRAIN1 = [] 
RUN_TIME1 = [] 

for i in NEURONS: 
    d = HIDDEN_NEURONS2(i)
    start = timeit.default_timer() #calculate program run time in python
    d.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage
    stop = timeit.default_timer()
    time = stop - start
    RUN_TIME1.append(time)
    a = d.score(x_test, y_test)  #computes the global score for test data
    b = d.score(x_train, y_train) #computes the global score for training data
    Score_TEST1.append(a)
    Score_TRAIN1.append(b) 
    

plt.figure(1)
plt.plot(NEURONS,Score_TEST1, 'r', label='SCORE TEST')
plt.plot(NEURONS,Score_TRAIN1, 'g', label='SCORE TRAIN')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('SCORES Relu')

plt.figure(2)
plt.plot(NEURONS,RUN_TIME1, 'b', label='RUN TIME')
plt.xlabel('NEURONS')
plt.ylabel('RUN TIME Relu')

plt.figure(3)
plt.plot(NEURONS,Score_TEST, 'r', label='SCORE TEST Logistic')
plt.plot(NEURONS,Score_TEST1, 'g', label='SCORE TEST Relu')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('SCORES')

plt.figure(4)
plt.plot(NEURONS,RUN_TIME1, 'b', label='RUN TIME Relu')
plt.plot(NEURONS,RUN_TIME, 'g', label='RUN TIME Logistic')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('RUN TIME')

plt.show()

# Plot the loss curve as function of epochs (iteration)

plt.figure(5)

plt.plot(d.loss_curve_,'b', label='LOSS CURVE Relu')
plt.plot(c.loss_curve_, 'g', label='LOSS CURVE Logistic')
plt.legend()

plt.title('The loss curve as function of epochs (iteration)')

plt.show()

"""
#EXERCICE 2

"""

#  2 hidden layers and Increase the number of hidden neurons from 5 to 20 (5, 10, 20).

def HIDDEN_NEURONS3(NUMBER):
    
    mlp= MLPRegressor(hidden_layer_sizes=(NUMBER,NUMBER), #number of neurons in hidden layer, here one single hidden layer with 20 neurons,

    activation='logistic', # activation function start with logistic function
    
    solver='adam', #solver
    
    alpha=0.01,# L2 penalty (regularization term) parameter
    
    batch_size='auto', #Size of minibatches for stochastic optimizers.
    
    learning_rate='adaptive', #Learning rate schedule for weight updates
    
    learning_rate_init=0.01, #The initial learning rate used
    
    max_iter=1000, #Maximum number of iterations
    
    tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
    
    verbose=True, # To Print progress messages during learning step
    
    warm_start=False, #When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution
    
    early_stopping=True, #Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver=’sgd’ or ‘adam’.
    
    validation_fraction=0.1, #The proportion of training data to set aside as validation set for early stopping
    
    n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.
    
    return mlp

Score_TEST3 = [] 
Score_TRAIN3 = [] 
NEURONS3 = [5,10,20]
RUN_TIME3 = [] 

for i in NEURONS3: 
    e = HIDDEN_NEURONS3(i)
    start = timeit.default_timer() #calculate program run time in python
    e.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage
    stop = timeit.default_timer()
    time = stop - start
    RUN_TIME3.append(time)
    a = e.score(x_test, y_test)  #computes the global score for test data
    b = e.score(x_train, y_train) #computes the global score for training data
    Score_TEST3.append(a)
    Score_TRAIN3.append(b) 

# Plot on the same graph the training set score and the test set score versus number of hidden neurons.

plt.figure(1)
plt.plot(NEURONS3,Score_TEST[0:3], 'r', label='SCORE TEST 1 hidden layer')
plt.plot(NEURONS3,Score_TEST3, 'g', label='SCORE TEST 2 hidden layers')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('SCORES Logistic')

plt.figure(2)
plt.plot(NEURONS3,RUN_TIME[0:3], 'b', label='RUN TIME 1 hidden layer')
plt.plot(NEURONS3,RUN_TIME3, 'g', label='RUN TIME 2 hidden layers')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('RUN TIMELogistic')

plt.figure(3)
plt.plot(c.loss_curve_,'b', label='LOSS CURVE 1 hidden layer')
plt.plot(e.loss_curve_, 'g', label='LOSS CURVE 2 hidden layers')
plt.legend()
plt.title('The loss curve as function of epochs (iteration)')

plt.show()
    

# Using relu activation function


def HIDDEN_NEURONS4(NUMBER):
    
    mlp= MLPRegressor(hidden_layer_sizes=(NUMBER,NUMBER), #number of neurons in hidden layer, here one single hidden layer with 20 neurons,

    activation='relu', # activation function start with logistic function
    
    solver='adam', #solver
    
    alpha=0.01,# L2 penalty (regularization term) parameter
    
    batch_size='auto', #Size of minibatches for stochastic optimizers.
    
    learning_rate='adaptive', #Learning rate schedule for weight updates
    
    learning_rate_init=0.01, #The initial learning rate used
    
    max_iter=1000, #Maximum number of iterations
    
    tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
    
    verbose=True, # To Print progress messages during learning step
    
    warm_start=False, #When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution
    
    early_stopping=True, #Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver=’sgd’ or ‘adam’.
    
    validation_fraction=0.1, #The proportion of training data to set aside as validation set for early stopping
    
    n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.
    
    return mlp

Score_TEST4 = [] 
Score_TRAIN4 = [] 
RUN_TIME4 = [] 

for i in NEURONS3: 
    f = HIDDEN_NEURONS4(i)
    start = timeit.default_timer() #calculate program run time in python
    f.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage
    stop = timeit.default_timer()
    time = stop - start
    RUN_TIME4.append(time)
    a = f.score(x_test, y_test)  #computes the global score for test data
    b = f.score(x_train, y_train) #computes the global score for training data
    Score_TEST4.append(a)
    Score_TRAIN4.append(b) 
    
#HD : Hidde Layer

plt.figure(1)
plt.plot(NEURONS3,Score_TEST[0:3], 'r', label='SCORE TEST 1 HD Logistic')
plt.plot(NEURONS3,Score_TEST3, 'g', label='SCORE TEST 2 HD Logistic')
plt.plot(NEURONS3,Score_TEST1[0:3], 'b', label='SCORE TEST 1 HD Relu')
plt.plot(NEURONS3,Score_TEST4, 'k', label='SCORE TEST 2 HD Relu')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('SCORES')

plt.figure(2)
plt.plot(NEURONS3,RUN_TIME[0:3], 'r', label='RUN TIME 1 HD Logistic')
plt.plot(NEURONS3,RUN_TIME3, 'g', label='RUN TIME 2 HD Logistic')
plt.plot(NEURONS3,RUN_TIME1[0:3], 'b', label='RUN TIME 1 HD Relu')
plt.plot(NEURONS3,RUN_TIME4, 'k', label='RUN TIME 2 HD Relu')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('RUN TIMELogistic')

plt.figure(3)
plt.plot(c.loss_curve_,'r', label='LOSS CURVE 1 HD Logistic')
plt.plot(e.loss_curve_, 'g', label='LOSS CURVE 2 HD Logistic')
plt.plot(d.loss_curve_,'b', label='LOSS CURVE 1 HD Relu')
plt.plot(f.loss_curve_, 'k', label='LOSS CURVE 2 HD Relu')
plt.legend()
plt.title('The loss curve as function of epochs (iteration)')

plt.show()

#  3 hidden layers and Increase the number of hidden neurons from 5 to 20 (5, 10, 20).

def HIDDEN_NEURONS5(NUMBER):
    
    mlp= MLPRegressor(hidden_layer_sizes=(NUMBER,NUMBER,NUMBER), #number of neurons in hidden layer, here one single hidden layer with 20 neurons,

    activation='logistic', # activation function start with logistic function
    
    solver='adam', #solver
    
    alpha=0.01,# L2 penalty (regularization term) parameter
    
    batch_size='auto', #Size of minibatches for stochastic optimizers.
    
    learning_rate='adaptive', #Learning rate schedule for weight updates
    
    learning_rate_init=0.01, #The initial learning rate used
    
    max_iter=1000, #Maximum number of iterations
    
    tol=1e-4, #When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
    
    verbose=True, # To Print progress messages during learning step
    
    warm_start=False, #When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution
    
    early_stopping=True, #Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. Only effective when solver=’sgd’ or ‘adam’.
    
    validation_fraction=0.1, #The proportion of training data to set aside as validation set for early stopping
    
    n_iter_no_change=50) #Maximum number of epochs to not meet tol improvement.
    
    return mlp

Score_TEST5 = [] 
Score_TRAIN5 = [] 
NEURONS3 = [5,10,20]
RUN_TIME5 = [] 

for i in NEURONS3: 
    g = HIDDEN_NEURONS5(i)
    start = timeit.default_timer() #calculate program run time in python
    g.fit(x_train, y_train)  #uses the data to perform training of the MLP regressor, apprentissage
    stop = timeit.default_timer()
    time = stop - start
    RUN_TIME5.append(time)
    a = g.score(x_test, y_test)  #computes the global score for test data
    b = g.score(x_train, y_train) #computes the global score for training data
    Score_TEST5.append(a)
    Score_TRAIN5.append(b) 

# Plot on the same graph the training set score and the test set score versus number of hidden neurons.

plt.figure(1)
plt.plot(NEURONS3,Score_TEST[0:3], 'r', label='SCORE TEST 1 hidden layer')
plt.plot(NEURONS3,Score_TEST3, 'g', label='SCORE TEST 2 hidden layers')
plt.plot(NEURONS3,Score_TEST5, 'b', label='SCORE TEST 3 hidden layers')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('SCORES Logistic')

plt.figure(2)
plt.plot(NEURONS3,RUN_TIME[0:3], 'r', label='RUN TIME 1 hidden layer')
plt.plot(NEURONS3,RUN_TIME3, 'g', label='RUN TIME 2 hidden layers')
plt.plot(NEURONS3,RUN_TIME5, 'b', label='RUN TIME 3 hidden layers')
plt.legend()
plt.xlabel('NEURONS')
plt.ylabel('RUN TIMELogistic')

plt.figure(3)
plt.plot(c.loss_curve_,'r', label='LOSS CURVE 1 hidden layer')
plt.plot(e.loss_curve_, 'g', label='LOSS CURVE 2 hidden layers')
plt.plot(g.loss_curve_, 'b', label='LOSS CURVE 3 hidden layers')
plt.legend()
plt.title('The loss curve as function of epochs (iteration)')

plt.show()
    