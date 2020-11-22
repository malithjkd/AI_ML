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

# load defalt dataset
X, Y = load_planar_dataset()

"""
#comment out the defalt flower dataset before running this
#select differant dataset
noizy_circles, noisy_moons, blobs, gaussian_quantitles, no_structure = load_extra_datasets()

datasets = {"noizy_circles": noizy_circles,
            "noisy_moons":noisy_moons,
            "blobs":blobs,
            "gaussian_quantitles":gaussian_quantitles}

dataset_select ="noisy_moons"
print(dataset_select)
X,Y =  datasets[dataset_select]
X,Y = X.T, Y.reshape(1, Y.shape[0])

if dataset_select == "blobs":
    Y =Y%2
# loding dataset is finished
"""

# plot the dataset using matplotlib
plt.scatter(X[0,:], X[1,:],c=Y,s=5,cmap=plt.cm.Spectral)


#dataset shape identification
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
# testcase for layer size
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

"""
# test case for test funtion propergation 
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propergation(X_assess, parameters)

print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))
"""


def compute_cost(A2 , Y, parameters):
    m = X.shape[1]
    logprobs = np.multiply(Y, np.log(A2))+np.multiply((1-Y),np.log(1-A2))
    cost = -np.sum(logprobs)/m
    cost = float(np.squeeze(cost))
    
    assert(isinstance(cost, float))
    
    return cost

"""
A2, Y_assess, prameters = compute_cost_test_case()
print("cost")
print(compute_cost(A2, Y_assess, prameters))
"""

def backwork_propergation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache['A1']
    A2 = cache['A2']
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    
    grads = {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}

    return grads
"""
# back propegation test case
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backwork_propergation(parameters, cache, X_assess, Y_assess)

print("dW1: " + str(grads["dW1"]))
print("db1: " + str(grads["db1"]))
print("dW2: " + str(grads["dW2"]))
print("db2: " + str(grads["db2"]))
"""

def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}
    
    return parameters
"""
# update paremeters function test
parameters, grads = update_parameters_test_case()
print("W1: \n" + str(parameters["W1"]))
print("b1: \n" + str(parameters["b1"]))
print("W2: \n" + str(parameters["W2"]))
print("b2: \n" + str(parameters["b2"]))

print("dW1: \n" + str(grads["dW1"]))
print("db1: \n" + str(grads["db1"]))
print("dW2: \n" + str(grads["dW2"]))
print("db2: \n" + str(grads["db2"]))

paremeters = update_parameters(parameters, grads)
print("results form the testcase")
print("W1: \n" + str(parameters["W1"]))
print("b1: \n" + str(parameters["b1"]))
print("W2: \n" + str(parameters["W2"]))
print("b2: \n" + str(parameters["b2"]))
# results are not exsacly the same
"""  

def nn_moddel(X, Y, n_h, num_iteration = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    print("n_h = " + str(n_h))
    
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iteration):
        
        A2, cache = forward_propergation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backwork_propergation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
    
    return parameters
"""
# test case for nn model function 
X_assess, Y_assess = nn_model_test_case()
parameters = nn_moddel(X_assess, Y_assess, 4, num_iteration=10000, print_cost=True)
print("W1: \n" + str(parameters["W1"]))
print("b1: \n" + str(parameters["b1"]))
print("W2: \n" + str(parameters["W2"]))
print("b2: \n" + str(parameters["b2"]))
"""

def predict(parameters, X):
    A2, cache = forward_propergation(X, parameters)
    predictions = np.round(A2)
    
    return predictions

"""
# prediction fuction testcase
parameters, X_assess = predict_test_case()
predictions = predict(parameters, X_assess)
print("prediction mean = " + str(np.mean(predictions)))
"""

# Running fill examples
parameters = nn_moddel(X, Y, n_h=4, num_iteration=4500, print_cost=True)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("decisiton boudry for hidden layer size " + str(8))

# to print accuracy 
predictions = predict(parameters, X)
print("Accuracy = ")
print(float(np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)






