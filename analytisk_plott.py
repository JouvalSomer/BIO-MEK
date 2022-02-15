import numpy as np
import matplotlib.pyplot as plt

'Constants'
c = 1 # Scaled concentration 
N = 10**3 # Number of iterations
L = 6 # Length
D = 10**(-3) # The diffusion coefficient
a0_over_two = 5*c/6 # a0 

'Defining the space and time arrays'
x = np.linspace(0, L, 801)
T = np.linspace(0, 10**3, 101)
T[-1] = 10**10 # Setting the last time to a larg number to see equilibrium

'The time and iteraton loop'
for t in T:
    u = 0   
    for k in range(1,N):
        Xx = (np.sin(np.pi*k/2) * (np.cos((k*np.pi*x)/L))) / (k)
        Tt = np.exp((-D * ((k*np.pi)/L)**2) * t)
        u += Xx * Tt
    
    # Plotting the consentration
    u = a0_over_two + u*(2*c)/(3*np.pi)
    plt.plot(x, u, color='magenta', label=f"Consentaration at t = {t} seconds")
    plt.ylim([0.2, 1.2])
    
    'Plotting lines'
    # Perivascular space and brain parenchyma separation
    x_values_separation = [L/2,L/2]
    y_values_separation = [0.566,1.1]
    plt.plot(x_values_separation, y_values_separation, color='green',
            linestyle='dashed', linewidth=0.5, label='Perivascular space and brain parenchyma separation')

    # Start consentarations
    x_values = [0, L]
    start_consentaration_perivascular_space = [1, 1]
    start_consentaration_brain_parenchyma = [2/3, 2/3]
    plt.plot(x_values, start_consentaration_perivascular_space, color='red',
            linestyle='-.', linewidth=0.5, label='Start consentaration in perivascular space')
    plt.plot(x_values, start_consentaration_brain_parenchyma,color='blue',
            linestyle='-.', linewidth=0.5, label='Start consentaration in brain parenchyma')
    
    plt.xlabel('Lengths in mm')
    plt.ylabel('Scaled concentration')
    plt.title('Concentration diffution over time')
    plt.legend()
    plt.savefig(f'Plots/test{t}.png')
    plt.show()