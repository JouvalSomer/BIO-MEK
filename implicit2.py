import numpy as np
from matplotlib import pyplot as plt
import datetime
import scipy.linalg

T = 10**4 # Total time
D1 = 10e-3 # The diffusion coefficient in the perivascular space
D2 = D1 # The diffusion coefficient in the brain parenchyma
L = 6 # # Length [mm]
Nx = 600 # 
Nt = 6000 # 
I1 = 1 # Initial condition in the perivascular space
I2 = 2/3 # Initial condition in the brain parenchyma
tol = 0.001 # Tolerance at equilibrium

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

# Boundary condition
A[Nx,Nx] = 1 + 2*F2
A[Nx,Nx-1]= -2*F2
A[0,0] = 1+2*F1
A[0,1] =  -2*F1

# Set initial condition u(x,0) = I(x)
c_n[0:Nx//2+1] = I1
c_n[Nx//2+1:Nx+1] = I2
    
# Timestep
for n in range(0, Nt):

    # Plotting the consentration
    fig, ax = plt.subplots()
    time = datetime.timedelta(seconds=t[n])
    ax.plot(x, c_n, color='k', label=f"Consentaration at {time}")
    ax.set_ylim([0.2, 1.2])

    # Plotting the green line between the perivascular space and brain parenchyma
    x_values_separation = [L/2,L/2]
    y_values_separation = [0.47,1.18]
    ax.plot(x_values_separation, y_values_separation, color='green',
            linestyle='dashed', linewidth=0.5, 
            label='Perivascular space (left) and brain parenchyma (right) barrier')

    # Plottin upper and lower start consentaration lines, red and blue respectively
    x_values = [0, L]
    start_consentaration_perivascular_space = [1, 1]
    start_consentaration_brain_parenchyma = [2/3, 2/3]
    ax.plot(x_values, start_consentaration_perivascular_space, color='red',
            linestyle='-.', linewidth=0.5, 
            label='Start consentaration in perivascular space')
    ax.plot(x_values, start_consentaration_brain_parenchyma,color='blue',
            linestyle='-.', linewidth=0.5, 
            label='Start consentaration in brain parenchyma')

    # Plotting the shaded background and titles of the two domains
    ax.axvspan(xmin=0, xmax=3, ymin=0.29, ymax=0.96, alpha=0.2, color='gray')
    ax.text(0.5, 1.07, 'Perivascular space', fontsize=12)
    ax.text(3.5, 1.07, 'Brain parenchyma', fontsize=12)

    plt.xlabel('Lengths in mm')
    plt.ylabel('Scaled concentration')
    plt.legend()
    plt.savefig(f'Plots_imp/implicit_{int(t[n])}.png')
    plt.show()

    # Space
    for i in range(0, Nx+1):
        b[i] = c_n[i]

    c[:] = scipy.linalg.solve(A, b)
    c_n[:] = c

    first = round(c_n[0],10) # First value of consentration in array (at x = 0)
    last = round(c_n[-1],10) # Last value of consentration in array (at x = L)
    if abs(first - last) < tol:
        print(f'Average consentration at equilibrium = {np.average(c_n)}')
        print(f'Time at equilibrium = {time}')
        break
