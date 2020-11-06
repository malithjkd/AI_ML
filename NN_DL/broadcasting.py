
"""
Created on Thu Nov  5 15:19:44 2020
Broadcasting example in cousera NN and DL
@author: malithjkd
"""

import numpy as np

A = np.array([[56,0,4.4,68],[1.2,104,52,8],[1.8,135,99,0.9]])
#print(A)

calc = A.sum(axis=0)
#print(calc)

populateCalc = calc.reshape(1,4)
#print(populateCalc)

presentage = 100*(A/calc)
presentage = np.round_(presentage,decimals=2)
print(presentage) 
