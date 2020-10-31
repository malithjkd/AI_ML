"""
Created on Thu Oct 29 20:55:59 2020
For numpy buildin function testing 
@author: malithjkd
"""

import numpy as np
import math

"""
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

# round off testing
b = np.random.rand(7)
c = np.random.rand(7)

d = np.round_(b,decimals = 2)
e = np.round_(c,decimals = 2)
print(d)
print(e)

# concatinate testing
f = np.concatenate((d,e))
print (f)

# sort testing
e = np.sort