#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:37:57 2022

@author: jacob
"""

from data_functions import *
from radius_angle_length import *

import numpy as np

n = 7 #number of points

mu = 10**(-3)

shrinkfactor = (2**(2/3))/2
r= 1
radius = new_radius(r)


length_int =  0.5*10**(-3)

G = np.zeros((n+1,n+1))


for i in range(2,n):
    for j in range(2,n):
        if j % 2 == 0:
            #if i % 2 == 0:
                G[int(j/2),j] = 1
                G[int(j/2),j+1] = 2
                
        if j % 2 == 0:
            #if i % 2 == 0:
                G[j,int(j/2)] = 1
                G[j+1,int(j/2)] = 2
        
G[0,0] = 0
G[n,n] = 0
G[0,1] = G[1,0] = length_int


p = pressure(G,1,2)

print(G)



#R = 8*mu/(np.pi*radius**4)

 

#print(radius)