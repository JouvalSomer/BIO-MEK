
hei
from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt

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
D = 0.001
beta = 0.001

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
"""
a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx + dt*D*beta*u*v*ds
L = u_n*v*dx + dt*D*beta*v*ds
b = assemble(L)
A = assemble(a)
solve(A, U.vector(), b)
"""
#u_D = Constant(1)


#def boundary(x, on_boundary):
#    return on_boundary

#bc = DirichletBC(V, u_D, boundary)


vtkfile = File('brain3/solution.pvd')
"""
a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx
#A = assemble(a)
L = u_n*v*dx

solve(a == L, U, bc)

u_n.assign(U)

vtkfile << (U, t)
plot(U)
"""

vtkfile << (U, t)
#%%
while t < T:
    t += dt
    
    a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx + dt*D*beta*u*v*ds
    L = u_n*v*dx + dt*D*beta*v*ds
    b = assemble(L)
    A = assemble(a)
    solve(A, U.vector(), b)
    
    if (t % 2) == 0:
        vtkfile << (U, t)
    plot(U)

    # Update previous solution
    
    m = assemble(U*dx)
    plt.plot(t, m/k)
    
    u_n.assign(U)

# Plot solution and mesh
plot(U)
plot(mesh)

"""
# Compute error in L2 norm
error_L2 = errornorm(u_D, U, 'L2')

print('error_L2  =', error_L2)

"""
# Hold plot
plt.show()
