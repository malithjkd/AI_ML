import numpy as np

#z = np.zeros((10,),dtype=int)
#print(z)

b = 0
#print(b)

X = np.random.randint(2,size=(10,10))
Y = np.random.randint(2,size=(1,10))
#print(X)
#print(Y)

W = np.random.rand(10)
WT = W.transpose()
#print(WT)

Z = np.dot(WT,X)+b
#Z = np.round_(Z,decimals=3)
#print(z)

A = 1/(1+np.exp(-1*Z))
#print(np.round_(A,decimals=3))

dZ = A-Y
#print(dZ)

dZT = dZ.transpose()
dW = (1/10)*np.dot(X,dZT)
dB = (1/10)*np.sum(dZ)
print(dW)
print(dB)

