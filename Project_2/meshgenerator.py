#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:11:38 2022

lag et mesh av blodårer

@author: lemmet
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack, vstack, bmat
from scipy.sparse.linalg import spsolve
# from scipy import linal

#%% setting constants

leak_resistance = 1/100  #Factor needed for calculating the leaking
factoroutflow = 10**-2 #the factor for calculating boundary pressure from flux as p_out = flux*factoroutflow
viscosity = 10**-3


shrinkfactor = (2**(2/3)) / 2  #how much the diameter shrinks each split

D = 3 /shrinkfactor #means that the first radius is 3, it gets multiplied by shrinkfactor later
levels = 10 #the amount of times the cells split

# startflux = 1 #the flux into the first point(not used in current version)

startpressure = 100  #pressure in the first point, cannot be zero as we need a startflux to calculate boundary pressures. Is the "overpressure"



#%%
save = "False" #saves the mesh to a file
plot = "False"  #plots the points





listofpoints = []   #list all the points

listofcells = [] #list of cells, each entry is (cellnumber, first point, second point, diameter, length )



firstpoint = [0, 0]
listofpoints.append(firstpoint)

cellnumber = 0  #to be used later 

#%% creating mesh
#Creating the points (coordinates are not used and not important)

for i in range(1, levels): #going throug the levels from the first
    pointsinlevel = 2**i #amount of points in level
    toppoint = i  #y coordinate for the top point
    
    
    
    for j in range(0, pointsinlevel): #going through the points in the level from the top down
        
        
        x = i              
        y = toppoint - j*2*toppoint/(pointsinlevel-1) # y = y of toppoint at that level - j*(the y-distance between points at that level)
        point = [x, y] 
        
        
        listofpoints.append(point)  
        
        
        
        
        
        
        
        
for i in range (1, levels):
    
    D = D * shrinkfactor   #setting the diameter as the last diameter times the shrinkingfactor
        
    tubesinlevel = 2**i
    pointsalreadytaken = 0 #need this mark how many points I have gone through in a level
    for j in range(0, int(tubesinlevel/2)):    #because there is half as many startingpoints as endpoints
        
        firstpoint = int(tubesinlevel/2) + j -1 #i dont use this
        
        for k in range(0, 2): #for the two last poinst after a split
            
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

    




# print("list of cells (cellnumber, two points it touches, diameteter, length):", listofcells)

# print()
# print("list of points: ", listofpoints)





if plot == "True":

    listofpointsarray = np.array(listofpoints)
    listofcellsarray = np.array(listofcells)
    x, y = listofpointsarray.T
    

    plt.scatter(x,y)




#%%

#solver



flux = [] #dont need this really

A = csr_matrix((len(listofpoints), len(listofpoints))) #the coefficient matrix for just the blooddomain
B = np.zeros(len(listofpoints)*2)  #The B in Ax = B, is for both blood and CSF so has to be twice as long
A_topright = csr_matrix((len(listofpoints), len(listofpoints)))

A_downleft = csr_matrix((len(listofpoints), len(listofpoints)))
c = [] #this is all the coefficients needed, on the form r**4/(8*l*viscosity)

for i in range(0, len(listofpoints)-1):
    c_value = ((listofcells[i][3])/2)**4/(8*listofcells[i][4]*viscosity)
    c.append(c_value)
    
c = c+c  #this copies the array and adds it, so it can be used on the csf system as well    

#Setting up the A matrix from the eq p2-p1 = -8*flux*viscosity*length/(pi*r**4)
for i in range(1, int((len(listofpoints)-1)/2)): #all points exept firstpoint and last level
    # A[i, i] = 1
    if i % 2 == 0: #different for odd and even numbers, as the splitting affects the numbering
        A[i, int(i/2)-1] = -c[i-1]  #dependence on last point
        A[i, int(i*2)+1] = -c[int(i*2)] #dependence on first splitpoint
        A[i, int(i*2)+2] = -c[int(i*2)+1] #dependence on  second splitpoint
        A[i, i] = c[int(i*2)+1] + c[int(i*2)] + c[i-1] - c[i-1]*leak_resistance #dependence on it self, including the leak to csf
    else:
        
        #same thing but for odd numbers
        A[i, int((i-1)/2)] = -c[i-1]
        A[i, int(i*2)+1] = -c[int(i*2)]
        A[i, int(i*2)+2] = -c[int(i*2)+1]
        A[i, i] = c[int(i*2)] + c[int(i*2)+1] + c[i-1] - c[i-1]*leak_resistance
        
for i in range(int((len(listofpoints)-1)/2), len(listofpoints)): #the coefficiants on the boundary eqs
    
    A[i, i] = 1 #later in the B-array we set this relative to the flux at that point
        
A[0, 0] = 1 #makes it so the first entry in B is the pressure in first point

A_CSF = A  #the matrix for CSF, lower left corner of total matrix

for i in range(int((len(listofpoints)-1)/2), len(listofpoints)): #says that for the boundary in csf, the pressure = the entry in B
    
    A_CSF[i, i] = 1



for i in range(1, len(listofpoints)):
    
    # A_topright[i, i] = k*np.pi*D*L/(viscosity*L_radial) #darcy's law, this is more accurate, but we lack information on thickness of cellwall and k (permability)
    
    # A_downleft[i, i] = - k*np.pi*D*L/(viscosity*L_radial)
    A_topright[i, i] = c[i-1]*leak_resistance  #says that the leak resistance is the c of last cell, times some factor
    
    A_downleft[i, i] = c[i-1]*leak_resistance





# A_final = np.bmat('A, A_topright; A_downleft, A_CSF') #combining the matrixes to a final A-matrix
topp_part = hstack([A, A_topright])
bottom_part = hstack([A_downleft, A_CSF])



A_final = vstack([topp_part, bottom_part], format="csc") #') #combining the matrixes to a final A-matrix
   
    
B[0] = startpressure
# B[int((len(listofpoints)-1)/2):len(listofpoints)] = - 5  #if we want dirichlet instead
    

pressure = np.zeros(len(listofpoints)*2) #solution array
B[0] = startpressure


for g in range(0, 10): #the iteration to find pressure at boundary. Pressure is unknown, using last solved pressure

    for i in range(int((len(listofpoints)-1)/2), len(listofpoints)): #this is for blood, all the last points
        
        if i % 2 == 0:
        
            B[i] = factoroutflow * c[i-1] * (pressure[i]-pressure[int(i/2)-1]) #says that the pressure = flux * factoroutflow. Have not included the leaking flux here
        else:
            B[i] = factoroutflow * c[i-1] * (pressure[i]-pressure[int((i-1)/2)])
            
            
    for i in range(len(listofpoints)+int((len(listofpoints)-1)/2),   int(len(listofpoints)*2)): #this is for csf
        
        if i % 2 == 0:
        
            B[i] = factoroutflow * c[i-2] * (pressure[i]-pressure[int(i/2)+len(listofpoints)]) #same thing as over, just for CSF
        else:
            B[i] = factoroutflow * c[i-2] * (pressure[i]-pressure[int((i-1)/2)])
    
    #this is for csf's first point, where flux is the sum of the split
    B[len(listofpoints)] = factoroutflow * (c[0] * (pressure[len(listofpoints)+1]-pressure[len(listofpoints)])+c[1] * (pressure[len(listofpoints)+2]-pressure[len(listofpoints)]))
    
    
    
    
    pressure_new = spsolve(A_final, B)
    maxerror = abs(max(pressure_new-pressure))
    print(maxerror)
    pressure = pressure_new        

# print("This is the pressure:", pressure)





