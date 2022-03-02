#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:11:38 2022

lag et mesh av blodårer

@author: lemmet
"""

import numpy as np
from matplotlib import pyplot as plt
import csv

l = 1 #typical length 

r = 10**-2 # typical radius

levels = 4





save = "True"
plot = "True"





listofpoints = []   #list all the points

listofcells = [] #list all the points each cell touches



firstpoint = [0, 0]
listofpoints.append(firstpoint)

cellnumber = 0
pointsalreadytaken = 0




for i in range(1, levels):
    pointsinlevel = 2**i #amount of points in level
    toppoint = i  #y coordinate for the top point
    tubesinlevel = pointsinlevel #moving backwards from the points in the level
    
    
    
    
    for j in range(0, pointsinlevel):
        
        
        x = i
        y = toppoint - j*2*toppoint/(pointsinlevel-1)
        point = [l*x, l*y] #l for scaling
        
        
        listofpoints.append(point)
        
        
        
        
        
        
        
        
for i in range (1, levels):
        
    tubesinlevel = 2**i
    pointsalreadytaken = 0
    for j in range(0, int(tubesinlevel/2)):    
        
        firstpoint = int(tubesinlevel/2) + j -1
        
        for k in range(0, 2):
            
            secondpoint = tubesinlevel + k + pointsalreadytaken*2 - 1
            cell = [cellnumber, firstpoint, secondpoint, r]
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
"""with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(listofpoints)
 """   
    
    
    
# if save == "True":
    
#     file = open(f"mesh_med_{levels}_levels", "w")
    
#     for element in listofpoints:
#         file.write(element)
    
    
#     file.close




print("list of cells (cellnumber, two points it touches, length, radius):", listofcells)

print()
print("list of points: ", listofpoints)





if plot == "True":

    listofpointsarray = np.array(listofpoints)
    listofcellsarray = np.array(listofcells)
    x, y = listofpointsarray.T
    

    plt.scatter(x,y)
    # for i in listofcells:
    #     plt.plot(listofcellsarray[i:i+1, 1], listofcellsarray[i:i+1, 2])
    # plt.show()












