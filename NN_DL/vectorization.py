#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:52:17 2020
To test the speed differance of numpy dot product multoplicationn
vs For loop 
@author: malithjkd
"""

import numpy
import time

a = numpy.random.rand(1000000)
b = numpy.random.rand(1000000)

tic = time.time()

c = numpy.dot(a,b)

toc = time.time()

print(c)
print("Vector multiplication Time :" + str(1000*(toc-tic)) + "ms") 

c = 0
tic = time.time()

i = 0 
for i in range(1000000):
    c = c + a[i]*b[i]
    
toc = time.time()

print(c)
print("For loop time :" + str(1000*(toc - tic)) + "ms")