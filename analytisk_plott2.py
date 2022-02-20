import numpy as np
import datetime
import matplotlib.pyplot as plt

"""Constants"""
c_ = 1 # Scaled concentration 
N = 10**3 # Number of iterations
L = 6 # Length [mm]
D = 10e-3 # The diffusion coefficient
a0_over_two = 5*c_/6 # a0 constant in the Fourier series
tol = 0.001 # Tolerance at equilibrium

"""Defining the space and time arrays"""
x = np.linspace(0, L, 601)
T = np.linspace(0, 10**4, 6001)

"""The time and iteraton loop"""
for t in T:
    c = 0   
    for k in range(1,N):
        Xx = (np.sin(np.pi*k/2) * (np.cos((k*np.pi*x)/L))) / (k)
        Tt = np.exp((-D * ((k*np.pi)/L)**2) * t)
        c += Xx * Tt
    
    # Adding the a0 term and the constant thats outside the sum
    c = a0_over_two + c*(2*c_)/(3*np.pi)

    """Plotting"""
    # Plotting the consentration
    fig, ax = plt.subplots()
    time = datetime.timedelta(seconds=t) 
    ax.plot(x, c, color='magenta', label=f"Consentaration at {time}")
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

    plt.xlabel('Lengths [mm]')
    plt.ylabel('Scaled concentration')
    plt.legend()
    plt.savefig(f'Plots/{int(t)}.png')
    plt.show()

    # Printing the final time and value of the consentration at that time
    first = round(c[0],10) # First value of consentration in array (at x = 0)
    last = round(c[-1],10) # Last value of consentration in array (at x = L)
    if abs(first - last) < tol:
        print(f'Average consentration at equilibrium = {np.average(c)}')
        print(f'Time at equilibrium = {time}')
        break 
