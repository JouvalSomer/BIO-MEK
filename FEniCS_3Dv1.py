from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

"""
parameters["krylov_solver"]["report"] = True
parameters["krylov_solver"]["monitor_convergence"] = True
parameters["krylov_solver"]["absolute_tolerance"] = 1.0e-8
parameters["krylov_solver"]["relative_tolerance"] = 1.0e-16
"""


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
#u_0 = Constant(2/3)
#u_n = Function(V) 
#u_n = interpolate(u_0, V)

'Input data'
T_s = 86400
days = 7
T = T_s*days
dt = 200
D_coeff = [1e-03, 0.623e-06 , ] 

m = 3600/dt


'Boundary conditions'
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, 1, boundary)

def D_nd(t):
    return  (-1/4800)*t + 7


def D_dn(t):
    return (1/800)*t-107


'initialize solution file'
vtkfile = File('brain_lang2/solution.pvd')

solution_w = np.zeros(T//dt +1)
solution_ab = np.zeros(T//dt +1)
solution_t = np.zeros(T//dt +1)
time = list(range(0,T,dt))#np.zeros(T//dt +1)
num_loop = 0
t = 0
t_n = 0
check = np.zeros(4)
#%%
for i,D in enumerate(D_coeff):
    'Initial condition'
    u_0 = Constant(2/3)
    u_n = Function(V) 
    u_n = interpolate(u_0, V)
    

    for t in range(0,T, dt):
        D_d = (0.5)**2*D
        'Sleep'
        if 0 <=t < 28800:
            if check[0]==0:
                a = u*v*dx + dt*D*dot(grad(u), grad(v))*dx
                A = assemble(a)
                bc.apply(A)
                check[0] = 1
            L = u_n*v*dx
            b = assemble(L)
            bc.apply(b)
            #print ("norm of b ", b.norm("l2"))
            solve(A, U.vector(), b, "gmres", "amg")
            u_n.assign(U)
            
    
        'Sleep -> awake'
        if 28800 <= t <= 32400 :
            
            D_nd1 = D_nd(t)*D
            a = u*v*dx + dt*D_nd1*dot(grad(u), grad(v))*dx 
            L = u_n*v*dx 
            solve(a == L, U, bc)
            if check[1] ==0:
                int_nd = U
                check[1] = 1
            u_n.assign(U + (1/(3*m))*int_nd)
            
    
        'Awake'
        if 32400 < t <= 85800:
            if check[2] == 0:
                a = u*v*dx + dt*D_d*dot(grad(u), grad(v))*dx 
                A = assemble(a)
                bc.apply(A)
                check[2] = 1
            L = u_n*v*dx 
            b = assemble(L)
            bc.apply(b)
            solve(A, U.vector(), b, "gmres", "amg")
            u_n.assign(U)
            
            
        'Awake -> sleep'
        if 85800 < t <= 86400 :
            D_dn1 = D_dn(t)*D
            a = u*v*dx + dt*D_dn1*dot(grad(u), grad(v))*dx 
            L = u_n*v*dx 
            solve(a == L, U, bc)
            if check[3] ==0:
                int_dn = U
                check[3] = 1
            u_n.assign(U - (1/(3*(600/dt)))*int_dn)
            
            
        'Print solution every 100th timestep'
        if (t_n % 300) == 0:
            vtkfile << (U, t_n)
    
    
        
        if t > t % T_s:
            t = 0
            check[:] = 0
        
        t_n += dt
        

        if i == 0:
            solution_w[num_loop] = assemble(u_n*dx)
        if i == 1:
            solution_ab[num_loop] = assemble(u_n*dx)
            
        if i == 2:
            solution_t[num_loop] = assemble(u_n*dx)
    
        print(t)
        print(t_n)


    

solution_water = 0.99*1*(125/(536.507))*solution/(assemble(Constant(1)*dx(mesh)))
solution_abeta = (9.9940036e-13)*(1.678e-9)*(125/(536.507))*solution/(assemble(Constant(1)*dx(mesh)))
solution_csftau = 


plt.plot(time, solution[:-1]/(assemble(Constant(1)*dx(mesh))))
plt.ylim(0.7, 1.32)
plt.show()