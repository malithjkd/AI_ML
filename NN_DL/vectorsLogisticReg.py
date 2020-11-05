import numpy as np

#z = np.zeros((10,),dtype=int)
#print(z)

b = -0.01

#print(b)
X = np.random.randint(2,size=(10,10))

W = np.random.rand(10)
WT = W.transpose()
#print(WT)

z = np.dot(WT,X)+b
z = np.round_(z,decimals=3)
#print(z)

a = 1/(1+np.exp(-1*z))
print(np.round_(a,decimals=3))