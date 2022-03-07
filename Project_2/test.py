#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:37:57 2022

@author: jacob
"""

from data_functions import *
from radius_angle_length import *

import numpy as np

n = 6 #number of points

mu = 10**(-3)

shrinkfactor = (2**(2/3))/2
r= 1
radius = new_radius(r)


length_int =  0.5*10**(-3)

G = np.zeros((n+1,n+1))


for i in range(0,n+1):
    if i % 2:
        G[i,i-1] = 1
        
G[0,0] = 0
G[n,n] = 0
G[0,1] = length_int


print(G)



#R = 8*mu/(np.pi*radius**4)

 

#print(radius)