#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 11:37:57 2022

@author: jacob
"""

from data_functions import *
from radius_angle_length import *

import numpy as np

n = 2**3#number of points

mu = 10**(-3)

shrinkfactor = (2**(2/3))/2
r= 1
radius = new_radius(r)


length_int =  0.6#0.5*10**(-3)

G1 = G2 = np.zeros((n,n))

G3 = G4 = np.zeros((n,n))


for i in range(2,n):
    for j in range(2,n):
        if j % 2 == 0:
            #if i % 2 == 0:
                G1[int(j/2),j] = 0.3
                G1[int(j/2),j+1] = 0.4
                
        if j % 2 == 0: 
            #if i % 2 == 0:
                G1[j,int(j/2)] = 0.3
                G1[j+1,int(j/2)] = 0.4
        
for i in range(0,n):
    G3[i,i] = 2
        

        
G = np.bmat([[G1, G3], [G4, G2]])
G2[0,0]
G1[0,0] = 0
G1[n-1,n-1] = 0
G1[0,1] = G1[1,0] = length_int

G = np.bmat([[G1, G3], [G4, G2]])

p,A,b = pressure(G,1,0.1)

print(G)



#R = 8*mu/(np.pi*radius**4)

 

#print(radius)