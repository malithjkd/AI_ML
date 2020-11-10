"""
MalithJKD
Logistric Regression with a Neural Networks Mindset
07.10.2020
"""

#import packages

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline

# load the dateset 
train_set_x_orig, train_set_y, test_set_x, test_set_y, classes = load_dataset()

"""
# to see the data
index = 200
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"' picture.")
"""

# idengify the train set and test set, no of pic, its shape
m_train = train_set_x_orig.shape[0]
m_test = test_set_x.shape[0]
num_px = train_set_x_orig.shape[1]
"""
print("Numbers of training examples: m_train = " +str(m_train))
print("Numbers of test examples: m_test = " +str(m_test))
print("Height/Width of each image: num_px = " +str(num_px))
print("Each image is of size:("+str(num_px) + "," + str(num_px) + ",3)")
print("train_set_x shape = " + str(train_set_x_orig.shape))
print("train_set_y shape = " + str(train_set_y.shape))
print("test_set_x shape = " + str(test_set_x.shape))
print("test_set_y shape = " + str(test_set_y.shape))
"""

# reshape the train and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0])
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[1]*test_set_x.shape[2]*test_set_x.shape[3],test_set_x.shape[0])
"""
print("train_set_x_flatten shape" + str(train_set_x_flatten.shape))
print("train_set_y shape" + str(train_set_y.shape))
print("test_set_x_flatten shape" + str(test_set_x_flatten.shape))
print("test_set_y shape" + str(test_set_y.shape))
# the internal values of the coloumn/row up to 5 values
print("sanity check after reshaping" + str(train_set_x_flatten[0:5,0]))
"""

# flatten the train set x values
train_set_x_flatten = train_set_x_flatten/225
test_set_x_flatten = test_set_x_flatten/225
#print("sanity check after reshaping" + str(train_set_x_flatten[0:5,0]))

# Greaded Function: segmoid function 
def segmoid(z):
    s = 1/(1+np.exp(-1*z))
    return s

#testing segmoid function
#print("Segmoid [0,2,3] = " + str(segmoid(np.array([0,2,3]))))

#creating initial metrix for w and b
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.0
    
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    
    return w,b
"""
#testing initialize_with_zeros function
w,b = initialize_with_zeros(5)
print("w = " + str(w))
print("b = " + str(b))
"""

def propagate(w,b,X,Y):
    m = X.shape[1]
    z = np.dot(w.T,X) + b
    A = segmoid(z)
    
    cost = (-1/m)*(np.sum( Y*np.log(A) + (1-Y)*np.log(1-A)))
        
    dw = (1/m)*(np.dot(X,np.transpose(A-Y)))
    db = (1/m)*(np.sum(A-Y))
    
    return cost, dw, db



w = np.array([[1],[2]])
b = 2.0
X = np.array([[1,2],[3,4]])
Y = np.array([[1,0]])

cost, dw, db = propagate(w, b, X, Y)

print("dw = " + str(dw))
print("db = " +str(db))
print("cost = " +str(cost))


# optimize function 
def optimize(w, b, X, Y, num_iteration, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iteration):
        cost, dw, db = propagate(w, b, X, Y)
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        
    
    return w, b, dw, db, costs

num_iteration = 100
learning_rate = 0.009
w, b, dw, db, costs = optimize(w, b, X, Y, num_iteration, learning_rate, print_cost=False)
"""
print("w")
print(w)
print("b")
print(b)
print("dw")
print(dw)
print("db")
print(db)
print("cost")
print(costs)
"""
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    z = np.dot(w.T,X) + b
    A = segmoid(z)
    
    for i in range(A.shape[1]):
        if A[0,i]<=0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

print("prediction")
print(predict(w, b, X))

def model(X_train, Y_train, X_test, Y_test, num_iteration, learning_rate, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    w, b, dw, db, costs = optimize(w, b, X, Y, num_iteration, learning_rate, print_cost = False)
    Y_prection_train = predict(w, b, X_train)
    Y_prection_test = predict(w, b, X_test)
    
    print("train accuracy = {}%".format(100 - np.mean(np.abs(Y_prection_train - Y_train))*100))
    print("test accuracy = {}%".format(100 - np.mean(np.abs(Y_prection_test - Y_test))*100))
    
    return 1

model(train , Y_train, X_test, Y_test, num_iteration, learning_rate)