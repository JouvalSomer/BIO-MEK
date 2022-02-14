import numpy as np
import matplotlib.pyplot as plt

'Definerer konstanter'
c = 1 # Skalert konsentrasjon 
N = 10**3 # Antall iterasjoner
L = 8 # Teoretisk lengde
D = 10**(-3) # Diffusjonskonstant
a0 = 5*c/6


'Definerer rom/tidsarrayene'
x = np.linspace(0, L, 801)
T = np.linspace(0, 10000, 101) 

'Tid/iterasjonsløkken'
for t in T:
    u = 0   
    for n in range(1,N):
        Xx = (np.sin(np.pi*n/2) * (np.cos((np.pi*n*x)/L))) / (3*np.pi*n)
        Tt = np.exp((-((np.pi*n)/L)**2) * D * t)
        u += Xx * Tt
    
    'Plotter konsentrasjonen for hvert tidssteg'
    u = a0 + u*(2*c)/(3*np.pi)
    plt.plot(x, u, label=f"t={t}")
    plt.legend()
    plt.ylim([0.81, 0.855])
    plt.savefig(f'Plots/test{t}.png')
    plt.show()