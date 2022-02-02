#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:20:25 2022

@author: jacob
"""
import numpy as np
from matplotlib import pyplot as plt
import datetime
import scipy.linalg

T = 48*3600
D1 = 10e-3
D2 = D1/3
L = 8
Nx = 1000
Nt = 50000
I1 = 1
I2 = 2/3

x = np.linspace(0, L, Nx+1)
dx = x[1] - x[0]
t = np.linspace(0, T, Nt+1)
dt = t[1] - t[0]
c = np.zeros(Nx+1)
c_n = np.zeros(Nx+1)
F1 = D1*dt/dx**2
F2 = D2*dt/dx**2


# Data structures for the linear system
A = np.zeros((Nx+1, Nx+1))
b = np.zeros(Nx+1)
for i in range(1, Nx//2+1):
    A[i,i-1] = -F1
    A[i,i+1] = -F1
    A[i,i] = 1 + 2*F1
for i in range(Nx//2+1, Nx):
    A[i,i-1] = -F2
    A[i,i+1] = -F2
    A[i,i] = 1 + 2*F2

A[Nx,Nx] = 1 + 2*F2
A[Nx,Nx-1]= -2*F2
A[0,0] = 1+2*F1
A[0,1] =  -2*F1
#A[0,1] = -F1
#A[Nx,Nx-1] = -F2
# Set initial condition u(x,0) = I(x)
c_n[0:Nx//2+1] = I1
c_n[Nx//2+1:Nx+1] = I2
    

#%%
for n in range(0, Nt):
    plt.figure(0)
    j = [4,4]
    k = [0,2]
    p = str(datetime.timedelta(seconds=t[n]))
    plt.plot(j,k, 'tab:gray', ls='--', lw='.8')
    plt.xlabel('Length [mm]')
    plt.ylabel('Concentration [C*y]')
    plt.title(f't = {datetime.timedelta(seconds=t[n])}')
    plt.plot(x,c_n,'k')
    plt.ylim([0.4,1.2])
    plt.show()
    for i in range(0, Nx+1):
        b[i] = c_n[i]
        #b[0] = I1
        #b[Nx] = I2
    c[:] = scipy.linalg.solve(A, b)
    c_n[:] = c
    
    if c_n[0] - c_n[Nx] < 0.005 :
        
        break
    

        
        
        
        
        


