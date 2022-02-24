#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 18:40:46 2022

@author: jacob
"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


'Get arguments from the commandline'
parser = argparse.ArgumentParser()
parser.add_argument("--quality", default='f', type=str)
parser.add_argument("--dt", default=200, type=float)
parser.add_argument("--days", default=1.0, type=float)
parser.add_argument("--plot", default="True", type=str)
args = parser.parse_args()

'Make directory'
directory = "tmp_" + "_dt" + str(args.dt) + 'quality' + str(args.quality)
if not os.path.exists(directory):
    os.mkdir(directory)

'Mesh'
if args.quality == 'f':
    mesh=Mesh()                                                             
    hdf=HDF5File(mesh.mpi_comm(), 'Mesh32.h5', 'r')                         
    hdf.read(mesh, '/mesh', False)                                                               


if args.quality == 'm':
    mesh=Mesh()                                                             
    hdf=HDF5File(mesh.mpi_comm(), 'Mesh16.h5', 'r')                         
    hdf.read(mesh, '/mesh', False)   
    

if args.quality == 'c':
    mesh=Mesh()                                                             
    hdf=HDF5File(mesh.mpi_comm(), 'Mesh8.h5', 'r')                         
    hdf.read(mesh, '/mesh', False)   



'Definitions'
V = FunctionSpace(mesh, "CG", 1) 
U = Function(V)
u = TrialFunction(V) 
v = TestFunction(V) 
u_n = Function(V) 



'Input data'
T_s = 86400 # Number of seconds in 24h
days = args.days # Number of days as an interger
T = T_s*days # Total time in seconds
dt = args.dt # Timestep size in seconds
D_ab = 0.0003726712994048381 # Diffusion coeff. for Amyloid-beta in mm^2/s
D_tau = 0.0001729919713074895 # Diffusion coeff. for Tau in mm^2/s
D_coeff = [D_ab, D_tau] 



'Boundary conditions'
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, 1, boundary)


'Initialize arrays used in the solver and post-proccesing'
night_ab = np.zeros(int(days))
day_ab = np.zeros(int(days))
night_t = np.zeros(int(days))
day_t = np.zeros(int(days))
solution_ab = np.zeros(int(T//dt) +1)
solution_t = np.zeros(int(T//dt) +1)
time = np.array(range(0,int(T),int(dt)))/(24*3600)
k = list(range(0,np.size(night_t)))
check = np.zeros(6)

for i,D in enumerate(D_coeff):
    'Initial condition'
    u_0 = Constant(2/3)
    u_n = Function(V) 
    u_n = interpolate(u_0, V)
    
    num_loop = 0 # counter for number of loops
    t = 0 # counter for time in a day
    t_n = 0 # counter for the total time for each molecule
    
    'initialize solution file'
    if args.quality == 'f' and dt==200:
        if i == 0:
            vtkfile = File(directory + '/Amyloid/solution.pvd')
        if i == 1:
            vtkfile = File(directory + '/Tau/solution.pvd')

    while t_n<T:
        'Sleep'
        if 0 <= t < 28800: 
            if check[0]==0: # Assembles the matrix only once within the time loop
                a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx
                A = assemble(a)
                bc.apply(A)
                check[0] = 1
            L = u_n*v*dx
            b = assemble(L)
            bc.apply(b)
            solve(A, U.vector(), b, "gmres", "amg")
            u_n.vector()[:] = U.vector()[:]
            
        'Sleep -> awake'
        if 28800 <= t < 32400 : 
            if check[1] ==0: # Assembles the matrix only once within the time loop
                a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx
                A = assemble(a)
                bc.apply(A)
                check[1] = 1      
            L = u_n*v*dx
            b = assemble(L)
            bc.apply(b)
            solve(A, U.vector(), b, "gmres", "amg")
            if check[2] == 0: # Saves the first solution in the transition phase
                UndFile = HDF5File(MPI.comm_world,"Und.h5","w")
                UndFile.write(U,"/Und")
                UndFile.close()
                int_nd = Function(V)
                UndFile = HDF5File(MPI.comm_world,"Und.h5","r")
                UndFile.read(int_nd,"/Und")
                UndFile.close()
                check[2] = 1  
            u_n.vector()[:] = U.vector() + (1/(2*(3600/dt)))*int_nd.vector() # Adjusts for the increase in concentration


        'Awake'
        if 32400 <= t < 85800:
            if check[3] == 0: # Assembles the matrix only once within the time loop
                a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx 
                A = assemble(a)
                bc.apply(A)
                check[3] = 1
            L = u_n*v*dx 
            b = assemble(L)
            bc.apply(b)
            solve(A, U.vector(), b, "gmres", "amg")
            u_n.assign(U)
            
            
        'Awake -> sleep'
        if 85800 <= t < 86400 :
            if check[4] ==0: # Assembles the matrix only once within the time loop
                a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx
                A = assemble(a)
                bc.apply(A)
                check[4] = 1
            L = u_n*v*dx
            b = assemble(L)
            bc.apply(b)
            solve(A, U.vector(), b, "gmres", "amg")
            if check[5] ==0: # Saves the first solution in the transition phase
                UdnFile = HDF5File(MPI.comm_world,"Udn.h5","w")
                UdnFile.write(U,"/Udn")
                UdnFile.close()
                int_dn = Function(V)
                UdnFile = HDF5File(MPI.comm_world,"Udn.h5","r")
                UdnFile.read(int_dn,"/Udn")
                UdnFile.close()
                check[5] = 1
            u_n.vector()[:] = U.vector() - (1/(3*(600/dt)))*int_dn.vector() # Adjusts for the decrease in concentration
            
        'Print solution every 10th timestep'
        if args.quality == 'f':
            if (t_n % 2000) == 0:
                vtkfile << (U, t_n)

        t+=dt
        if t > t % T_s: # Resets timecounter after 24 hours
            t = 0
            check[:] = 0 # Resets checks
        
        t_n += dt   # Advances total time counter 

        if i == 0:
            solution_ab[num_loop] = assemble(u_n*dx) # Saves the integral of the solution to an array each timestep
            
        if i == 1:
            solution_t[num_loop] = assemble(u_n*dx) # Saves the integral of the solution to an array each timestep
         
        num_loop += 1 # Advances counter for number of loops
        print(t) # Prints curent timestep within 24 hours 
        print(t_n) # Prints current timestep of the total time


'Calculate concentration of the molecules'
solution_abeta = (1.678e-9)*(75/(536.507))*solution_ab
solution_csftau = (3e-7)*(75/(536.507))*solution_t


'Create arrays for the solution at the first timestep after the transition periods'
for i in range(0, int(T), int(T_s)):
    night_ab[int(i/int(T_s))] = solution_ab[int(i/int(dt))]/(assemble(Constant(1)*dx(mesh)))
for i in range(32400, int(T), int(T_s)):
    day_ab[int(i/int(T_s))] = solution_ab[int(i/dt)]/(assemble(Constant(1)*dx(mesh))) 
                                      

for i in range(0, int(T), int(T_s)):
    night_t[int(i/int(T_s))] = solution_t[int(i/int(dt))]/(assemble(Constant(1)*dx(mesh)))
for i in range(32400, int(T), int(T_s)):
    day_t[int(i/int(T_s))] = solution_t[int(i/dt)]/(assemble(Constant(1)*dx(mesh))) 




'Plot'
if args.plot =='True':
    plt.plot(time, 1e10*solution_abeta[:-1]/(assemble(Constant(1)*dx(mesh))), 'c')
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12)
    plt.title('Concentration of Amyloid-beta')
    plt.ylabel('Concentration [1e-10 mg/ml]')
    plt.xlabel('time [days]')
    plt.savefig(directory + "/c_ab.png",dpi=300)
    plt.show()
    plt.plot(time, 1e8*solution_csftau[:-1]/(assemble(Constant(1)*dx(mesh))))
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12)
    plt.title('Concentration of Tau')
    plt.ylabel('Concentration [1e-8 mg/ml]')
    plt.xlabel('time [days]')
    plt.savefig(directory + "/c_tau.png",dpi=300)
    plt.show()
    
    
    
'Print solution arrays to file'  
np.savetxt(directory + "/solution_abeta.csv", solution_abeta, delimiter=",")
np.savetxt(directory + "/solution_csftau.csv", solution_csftau, delimiter=",")
np.savetxt(directory + "/night_ab.csv", night_ab, delimiter=",")
np.savetxt(directory + "/day_ab.csv", day_ab, delimiter=",")
np.savetxt(directory + "/night_t.csv", night_t, delimiter=",")
np.savetxt(directory + "/day_t.csv", day_t, delimiter=",")
np.savetxt(directory + "/k.csv", k, delimiter=",")
np.savetxt(directory + "/time.csv", time, delimiter=",")
