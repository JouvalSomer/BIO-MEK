import matplotlib.pyplot as plt
import numpy as np

def branch2(S_P, D, angle):
    """
    """
    global L0
    global counter
    D_n = D * (2**(2/3)) / 2
    L = 100*D_n
    if counter == 0:
        N_P = S_P
    else:
        N_P = (S_P[0] + L*np.cos(angle), S_P[1] + L*np.sin(angle))

    plt.plot(N_P[0], N_P[1], 'b.')
    
    if L > L0:
        counter += 1
        branch2(N_P, D_n, angle)
        branch2(N_P, D_n, -angle)
        
        
if __name__=="__main__":
    # Global variables
    L0 = 50 # Smallest length [micrometers]
    counter = 0 
    
    angle = 7*np.pi/36
    S_P = (0,0)
    D = 3
    branch2(S_P, D, angle)
    plt.show()
