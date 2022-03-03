#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:11:38 2022

lag et mesh av blodårer og løs det

@author: lemmet
"""

import numpy as np
from matplotlib import pyplot as plt


 

viscosity = 10**-3


shrinkfactor = (2**(2/3)) / 2

D = 3 /shrinkfactor #means that the first radius in 3, it gets multiplied by shrinkfactor later
levels = 4  #the amount of times the cells split

startflux = 1 #the flux into the first point

startpressure = 0  #pressure in the first point

save = "False"
plot = "True"





listofpoints = []   #list all the points

listofcells = [] #list of cells, each entry is (cellnumber, first point, second point, diameter, length )



firstpoint = [0, 0]
listofpoints.append(firstpoint)

cellnumber = 0  #to be used later 




for i in range(1, levels): #going throug the levels from the first
    pointsinlevel = 2**i #amount of points in level
    toppoint = i  #y coordinate for the top point
    
    
    
    for j in range(0, pointsinlevel): #going through the points in the level from the top down
        
        
        x = i
        y = toppoint - j*2*toppoint/(pointsinlevel-1)
        point = [x, y] #l for scaling
        
        
        listofpoints.append(point)
        
        
        
        
        
        
        
        
for i in range (1, levels):
    
    D = D * shrinkfactor   #setting the diameter as the last diameter times the shrinkingfactor
        
    tubesinlevel = 2**i
    pointsalreadytaken = 0 #need this mark how many points I have gone through in a level
    for j in range(0, int(tubesinlevel/2)):    #because there is half as many startingpoints as endpoints
        
        firstpoint = int(tubesinlevel/2) + j -1
        
        for k in range(0, 2):
            
            secondpoint = tubesinlevel + k + pointsalreadytaken*2 - 1
            cell = [cellnumber, firstpoint, secondpoint, D, D*100]  #last entry is the length
            listofcells.append(cell)        
            cellnumber += 1
        
        pointsalreadytaken += 1
        
        
        
if save == "True":
    f = open("listofpoints.txt", "w")
    for element in listofpoints:
        
        f.write(str(element)+"\n")
        
        print(element)
        
        
    for element in listofcells:
        
        f.write(str(element)+"\n")   
       
        
    f.close()

    




print("list of cells (cellnumber, two points it touches, diameteter, length):", listofcells)

print()
print("list of points: ", listofpoints)





if plot == "True":

    listofpointsarray = np.array(listofpoints)
    listofcellsarray = np.array(listofcells)
    x, y = listofpointsarray.T
    

    plt.scatter(x,y)






#solver



flux = []

A = np.zeros((len(listofpoints), len(listofpoints))) #the coefficient matrix
B = np.zeros(len(listofpoints))  #The B in Ax = B


#Setting up the A matrix from the eq p2-p1 = -8*flux*viscosity*length/(pi*r**4)
for i in range(1, len(listofpoints)):
    A[i, i] = 1
    if i % 2 == 0: 
        A[i, int(i/2)-1] = -1
    else:
        A[i, int((i-1)/2)] = -1
        
A[0, 0] = 1 #makes it so the first entry in B is the pressure in first point


#finding the flux from the startflux, using same numbering system as cells;
for i in range (1, levels):
    
   
    
        
    tubesinlevel = 2**i
    for k in range(0, tubesinlevel):
        fluxincell = startflux*(1/2)**i 
        flux.append (fluxincell)
        
        
        
B[0] = 0 #first pressure

#calculating rest of B:
for i in range(1, len(listofpoints)):
    fluxincell = flux[i-1]
    length = listofcells[i-1][4]
    radius = listofcells[i-1][3]/2
    B[i] = -8*fluxincell*viscosity*length/(np.pi*radius**4)
                




pressure = np.linalg.solve(A, B)
    

print("This is the pressure:", pressure)













