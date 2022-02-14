from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np


parameters["krylov_solver"]["report"] = True
parameters["krylov_solver"]["monitor_convergence"] = True
parameters["krylov_solver"]["absolute_tolerance"] = 1.0e-8
parameters["krylov_solver"]["relative_tolerance"] = 1.0e-16

'Mesh'
mesh=Mesh()                                                             
hdf=HDF5File(mesh.mpi_comm(), 'Mesh32.h5', 'r')                         
hdf.read(mesh, '/mesh', False)                                          


'Definitions'
V = FunctionSpace(mesh, "CG", 1) 
U = Function(V)
u = TrialFunction(V) 
v = TestFunction(V) 
u_n = Function(V) 


'Initial condition'
u_0 = Constant(2/3)
u_n = Function(V) 
u_n = interpolate(u_0, V)

'Input data'
T = 115800
dt = 200
D = 0.001 # Lage sammensetting med CSF-tau, Abeta og vann med prosent fordeling
m = 3600/dt


'Boundary conditions'
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, 1, boundary)


'initialize solution file'
vtkfile = File('brain_lang/solution.pvd')

solution = np.zeros(T//dt +1)
num_loop = 0
t = 0
#%%
while t < T:

    'Sleep 1 [0,8]t'
    if t < 28800:
        a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx 
        if True: A = assemble(a)
        bc.apply(A)
        L = u_n*v*dx
        b = assemble(L)
        bc.apply(b)
        print ("norm of b ", b.norm("l2"))
        solve(A, U.vector(), b, "gmres", "amg")
        
        u_n.assign(U)      
    

    'Sleep -> awake 1 [8,9]t'
    if 28800 <= t <= 32400:
        D_d = (3/2)**(1/m)*D
        a = u*v*dx + dt*D_d*dot(grad(u), grad(v))*dx 
        L = u_n*v*dx 
        solve(a == L, U, bc)
        u_n.vector()[:] = (4/3)**(1/m)*U.vector()[:]
    
 
    'Awake 1 [9, 24]'
    if 32400 < t <= 86400:
        D_d = (3/2)*D
        a = u*v*dx + dt*D_d*dot(grad(u), grad(v))*dx 
        L = u_n*v*dx 
        solve(a == L, U, bc)
        u_n.assign(U)    
        
        
    'Awake -> sleep 2 [24t, 24t + 10min]'
    if 86400 < t <= 87000:
        D_d = (3/2)*D
        D_n = (2/3)**(1/m)*D_d
        a = u*v*dx + dt*D_n*dot(grad(u), grad(v))*dx 
        L = u_n*v*dx 
        solve(a == L, U, bc)
        u_n.vector()[:] = (2/3)**(1/m)*U.vector()[:]
        
        
    'Sleep 2 [24t + 10min, 32t + 10min]'
    if 87000 < t <= 115800:
        a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx 
        L = u_n*v*dx 
        solve(a == L, U, bc)
        u_n.assign(U)   
    
    'Print solution every 100th timestep'
    if (t % 100) == 0:
        vtkfile << (U, t)


    
    t += dt
    
    solution[num_loop] = assemble(u_n*dx)
    
    num_loop += 1



