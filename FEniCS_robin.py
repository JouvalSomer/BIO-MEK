from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

mesh=Mesh()                                                             

hdf=HDF5File(mesh.mpi_comm(), 'Mesh32.h5', 'r')                         

hdf.read(mesh, '/mesh', False)                                          

V = FunctionSpace(mesh, "CG", 1) 

u_0 = Constant(2/3)
u_n = Function(V) 
U = Function(V)
u_n = interpolate(u_0, V)
u = TrialFunction(V) 
v = TestFunction(V) 



T = 40000
dt = 1000
D = 0.01 # Lage sammensetting med CSF-tau, Abeta og vann med prosent fordeling
beta = 0.01

t_v = np.linspace(0,T,T//dt)
sol = np.zeros(T//dt)

def boundary(x, on_boundary):
    return on_boundary

vtkfile = File('brain3/solution.pvd')

bc = DirichletBC(V, 1, boundary)
bc.apply(u_n.vector())



k = assemble(u_n*dx)

a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx + dt*D*beta*u*v*ds
L = u_n*v*dx + dt*D*beta*v*ds
b = assemble(L)
A = assemble(a)
solve(A, U.vector(), b)



u_n.assign(U)

t = 0

vtkfile << (U, t)



num_loop = 0 
#%%
while t < T:
    t += dt
    
    num_loop += 1
    
    
    a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx #+ dt*D*beta*u*v*ds
    L = u_n*v*dx #+ dt*D*beta*v*ds
    b = assemble(L)
    A = assemble(a)
    solve(A, U.vector(), b)
    
    if (t % 2) == 0:
        vtkfile << (U, t)
    #plot(U)
    
    
    # Update previous solution
    
    m = assemble(U*dx)
    
    sol[num_loop-1]=k/m
    
    u_n.assign(U)

# Plot solution and mesh
#plot(U)
#plot(mesh)

plt.plot(t_v, sol)
   
# Hold plot
plt.show()
