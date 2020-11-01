"""
Created on Thu Oct 29 20:55:59 2020
For numpy buildin function testing 
@author: malithjkd
"""

import numpy as np
import math

"""
# Trigonometry maths operation
a = [0,math.pi/2,math.pi/3,math.pi]
sin_val = np.sin(a)
cos_val = np.cos(a)
tan_val = np.tan(a)
dig_val = np.rad2deg(a)

print("a :")
print(a)
print("sin val")
print(sin_val)
print("cos val")
print(cos_val)
print("tan val")
print(tan_val)
print("dig val")
print(dig_val)
"""

"""
# round off testing
b = np.random.rand(7)
c = np.random.rand(7)

d = np.round_(b,decimals = 2)
e = np.round_(c,decimals = 2)
print(d)
print(e)
"""

"""
# concatinate testing
f = np.concatenate((d,e))
print (f)
g = np.array([[1,2],[3,4]])
h = np.array([[5,6]])
i = np.concatenate((g,h),axis=0)
print(i)
"""

"""
# sort testing
sort_mx= np.sort(f)
print(sort_mx)
"""

"""
# array size, dimention, shape
array_example = np.array([[[0,1,2,3],[4,5,6,7]],
                          [[0,1,2,3],[4,5,6,7]],
                          [[0,1,2,3],[4,5,6,7]]])

print(array_example)
print(array_example.size)
print(array_example.ndim)
print(array_example.shape)
"""

"""
j = np.arange(8)    #to get 8 values to the array
reshape_array = j.reshape(4,2)

print(j)
print(reshape_array)

k = np.reshape(j, newshape=(2,4),order='C')
print(k)
"""

"""
# Basic array operation
data = np.array([1,2,3,4])
once = np.ones(4, dtype=int)
print(once)
total = once*3+data
print(total)

# Max, min and sum
print(total.sum())
"""

"""
# idengifying unique and count in metrix 
l = np.array([11,11,12,14,15,16,17,12,13,11,14,18,19,20])
print(l)

unique_valus = np.unique(l)
print(unique_valus)

# which nos are unique in the metrix
unique_valus, indices_list = np.unique(l, return_index=True)
print(indices_list)

# no of occurance of nos
unique_valus, occurance_count = np.unique(l, return_counts=True)
print(occurance_count)
"""
"""
# Transposing and reshaping metix
m = np.random.rand(6)
print(m)
#n = np.arange(6).reshape((2,3))
n = m.reshape((2,3))
print(n)

p = n.transpose()
print(p)
"""
"""
# reverse array
q = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(q)
fliped_array = np.flip(q)
print(fliped_array)
fliped_rows = np.flip(q, axis=0)
print(fliped_rows)
fliped_colomns = np.flip(q, axis=1)
print(fliped_colomns)
"""

"""
# flatten
r = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(r)
#flat_array = r.flatten() #do not chage the parent date 
flat_array = r.ravel()  # change the parent data it after flatten
flat_array[1] = 90
print(r)
print(flat_array)
"""


# working with mathematicl equations
# mean_square_error = 1/n(sum((y^i - yi)^2))

y = np.array([1,1,1,1,1,1,1,1])
y_hat = np.array([0,1,0,1,0,1,1,1])

error = (1/8)*np.sum(np.square(y_hat - y))
print(error)

# array save to file
np.save('y_hat',y_hat)
y_hat_file = np.load('y_hat.npy')
print(y_hat_file)


