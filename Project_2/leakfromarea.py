#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:12:37 2022

@author: lemmet
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:11:38 2022

lag et mesh av blodårer

@author: lemmet
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, hstack, vstack, bmat, lil_matrix
from scipy.sparse.linalg import spsolve
# from scipy import linal

#%% setting constants
"INPUTS"
leak_resistance = 10**-9  #1/Resisistance, so higher number means more leaking
viscosity = 10**-3   
factoroutflow = 2*133*60*10**6    #Needed on the boundary as P_out = factoroutflow*Q + P_end
lengthfactor = 100 #"cell length = diameter*lengthfactor"
shrinkfactor = (2**(2/3)) / 2  #how much the diameter shrinks each split

D = 0.03 /shrinkfactor #means that the first radius is 3, it gets multiplied by shrinkfactor later
levels = 20#10*-15 the amount of times the cells split

startpressure = 40*133  #pressure in the first point in blood
endpressure = 30*133   #The pressure at the end, not including the boundary condition from flux. 1 Pa = 133 mmHg
csfpressure = 5*133  # the pressure at csf boundaries, not including flux boundary condition

#%%
save = "False" #saves the mesh to a file
plot = "False"  #plots the points
"END IMPUTS"


listofpoints = []   #list all the points

listofcells = [] #list of cells, each entry is (cellnumber, first point, second point, diameter, length )



firstpoint = [0, 0]
listofpoints.append(firstpoint)

cellnumber = 0  #to be used later 

#%% creating mesh
#Creating the points (coordinates are not used and not important)
print("creating mesh")
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
            cell = [cellnumber, firstpoint, secondpoint, D, D*lengthfactor]  #last entry is the length
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
    plt.show()



#%%

#solver

Np = int(len(listofpoints))

# flux = [] #dont need this really
print(f"creating matrices, number of points = {Np}")
A = lil_matrix((Np, Np)) #the coefficient matrix for just the blooddomain
A_CSF = lil_matrix((Np, Np)) #for CSF
B = np.zeros(Np*2)  #The B in Ax = B, is for both blood and CSF so has to be twice as long
A_topright = lil_matrix((Np, Np))

A_downleft = lil_matrix((Np, Np))
c = [] #this is all the coefficients needed, on the form r**4/(8*l*viscosity)

for i in range(0, Np-1):
    c_value = np.pi*((listofcells[i][3])/2)**4/(8*listofcells[i][4]*viscosity)
    c.append(c_value)
    
c = c+c  #this copies the array and adds it, so it can be used on the csf system as well    

#Setting up the A matrix from the eq p2-p1 = -8*flux*viscosity*length/(pi*r**4)
for i in range(1, int((Np-1)/2)): #all points exept firstpoint and last level
    # A[i, i] = 1
    if i % 2 == 0: #different for odd and even numbers, as the splitting affects the numbering
        lastpoint = int(i/2-1)
        nextpoint1 = int(i*2+1)
        nextpoint2 =  nextpoint1 + 1
        A[i, lastpoint] = c[i-1]  #dependence on last point
        A[i, nextpoint1] = c[nextpoint1-1] #dependence on first splitpoint
        A[i, nextpoint2] = c[nextpoint2-1] #dependence on  second splitpoint
        A[i, i] = -c[int(i*2)+1] - c[int(i*2)] - c[i-1] - np.pi*listofcells[i-1][3]*listofcells[i-1][4]*leak_resistance #dependence on it self, including the leak to csf
    else:
        
        
        #same thing but for odd numbers
        lastpoint = int((i-1)/2)
        nextpoint1 = int(i*2+1)
        nextpoint2 =  nextpoint1 + 1
        
        A[i, lastpoint] = c[i-1]
        A[i, nextpoint1] = c[nextpoint1-1]
        A[i, nextpoint2] = c[nextpoint2-1]
        A[i, i] = -c[int(i*2)] - c[int(i*2)+1] - c[i-1] - np.pi*listofcells[i-1][3]*listofcells[i-1][4]*leak_resistance
        if i % 10000 == 0:
            print(f"creating A_blood, row {i} of {Np} points")
for i in range(int((Np-1)/2), Np): #the coefficiants on the boundary eqs
    
    A[i, i] = 1 + factoroutflow*c[i-1] + leak_resistance*np.pi*listofcells[i-1][3]*listofcells[i-1][4]
    if i % 2 == 0: #different for odd and even numbers, as the splitting affects the numbering
        lastpoint = int(i/2-1)   
        A[i, lastpoint] = -factoroutflow*c[i-1]
    else:
        
        lastpoint = int((i-1)/2)
        A[i, lastpoint] = -factoroutflow*c[i-1]

    if i % 10000 == 0:
        print(f"creating A_blood, row {i} of {Np}")
    
    
    
    
A[0, 0] = 1 #makes it so the first entry in B is the pressure in first point

print("Assembling A_final")
A_CSF = lil_matrix.copy(A)  #the matrix for CSF, lower left corner of total matrix

# for i in range(int((Np-1)/2), Np): #says that for the boundary in csf, the pressure = the entry in B
    
#     A_CSF[i, i] = 1 + factoroutflow*c[i-1]
"#for the first point in csf:"
A_CSF[0, 0] = 1+factoroutflow*c[0]+factoroutflow*c[1]

A_CSF[0, 1] = -factoroutflow*c[0]
A_CSF[0, 2] = -factoroutflow*c[1]


for i in range(1, int((Np-1)/2)):
    
    # A_topright[i, i] = k*np.pi*D*L/(viscosity*L_radial) #darcy's law, this is more accurate, but we lack information on thickness of cellwall and k (permability)
    
    
    A_topright[i, i] = np.pi*listofcells[i-1][3]*listofcells[i-1][4]*leak_resistance  #says that the leak resistance is the c of last cell, times some factor
    
    A_downleft[i, i] =  np.pi*listofcells[i-1][3]*listofcells[i-1][4]*leak_resistance
"for boundary in blood and csf"
for i in range(int((Np-1)/2), Np):
    
    # A_topright[i, i] = k*np.pi*D*L/(viscosity*L_radial) #darcy's law, this is more accurate, but we lack information on thickness of cellwall and k (permability)
    
    
    A_topright[i, i] = -np.pi*listofcells[i-1][3]*listofcells[i-1][4]*leak_resistance  #says that the leak resistance is the c of last cell, times some factor
    
    A_downleft[i, i] =  -np.pi*listofcells[i-1][3]*listofcells[i-1][4]*leak_resistance




# A_final = np.bmat('A, A_topright; A_downleft, A_CSF') #combining the matrixes to a final A-matrix
topp_part = hstack([A, A_topright])
bottom_part = hstack([A_downleft, A_CSF])



A_final = vstack([topp_part, bottom_part], format="csc") #') #combining the matrixes to a final A-matrix
   
    
B[0] = startpressure


print("Assembling B-array")
for i in range(Np+int((Np-1)/2),   int(Np*2)): #this is for csf
    
    B[i] = csfpressure
#this is for csf's first point, where flux is the sum of the split
B[Np] = csfpressure



for i in range(int((Np-1)/2), Np):
    B[i] = endpressure


print("Solving Ax=B")
pressure = spsolve(A_final, B)




avg_pressure_blood = np.zeros(levels)

avg_pressure_CSF = np.zeros(levels)


avg_pressure_blood[0] = pressure[0]
avg_pressure_CSF[0] = pressure[Np]

for i in range(1, levels):
    pointsinlevel = 2**i
    
    avg_pressure_blood[i] = np.average(pressure[pointsinlevel-1:2*pointsinlevel-2])

    avg_pressure_CSF[i] = np.average(pressure[Np+pointsinlevel-1:Np+2*pointsinlevel-2])


x = np.linspace(0, levels, levels)
plt.plot( x, avg_pressure_blood, "-r", label="Average Blood-Pressure")
plt.legend()
plt.xlabel("Amount of splits")
plt.ylabel("Average Pressure [Pa]")
plt.show()

plt.plot(x, avg_pressure_CSF,  "-b", label="Average CSF-Pressure")
plt.xlabel("Amount of splits")
plt.ylabel("Average Pressure [Pa]")
plt.legend()
plt.show()


