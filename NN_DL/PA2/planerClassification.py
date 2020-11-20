"""
MalithJKD
planer data classification with one hidden layer 
17.11.2020
"""

#import packages
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

get_ipython().magic('matplotlib inline')

np.random.seed(1)

X, Y = load_planar_dataset()

plt.scatter(X[0,:],X[1,:], c=Y, s=10, cmap=plt.cm.Spectral);

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

print('X shape = ' +str(shape_X))
print('Y shape = ' +str(shape_Y))
print('m = ' +str(m))


# scikit learn to prect the output using leanear regression 

#clf = sklearn.linear_model.LogisticRegressionCV();
#clf.fit(X.T,Y.T)

#plot_decision_boundary(lambda x: clf.predict(x), X, Y)
#plt.title("Logistic Regression")

#n_x

def layer_sizes(X,Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

"""
# testcase
X_a, Y_a = layer_sizes_test_case()
n_x, n_h, n_y = layer_sizes(X_a, Y_a)
print(n_x)
print(n_h)
print(n_y)
"""

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))    
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    
    return parameters

"""
# test case for initialize parameters function

n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1")
print(parameters["W1"])
print("b1")
print(parameters["b1"])
print("W2")
print(parameters["W2"])
print("b2")
print(parameters["b2"])
"""

def forward_propergation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}
    
    return A2, cache

X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propergation(X_assess, parameters)

print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

def compute_cost(A2 , Y, parameters):
    m = X.shape[1]
    logprobs = np.multiply(Y, np.log(A2))+np.multiply((1-Y),np.log(1-A2))
    