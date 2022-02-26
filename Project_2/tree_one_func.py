import matplotlib.pyplot as plt
import numpy as np

def branch2(S_P, D, angle):
    """
    """
    # Defining global variables
    global L0 # Smallest length of a capillary
    global first_point # True or Fase depending on whether we are on the first point or not

    # Calculating lengths from diameter
    D_n = D * (2**(2/3)) / 2
    L = 100*D_n

    # Goes in it True
    if first_point:
        plt.plot(S_P[0], S_P[1], 'r.')  # Plots first point 
        N_P = (S_P[0] + L, S_P[1])  # Sets the next point; e.g. from (0,0) to (L,0)
        first_point = False 

    else:
        N_P = (S_P[0] + L*np.cos(angle), S_P[1] + L*np.sin(angle))

    plt.plot(N_P[0], N_P[1], 'b.')

    if L > L0:
        # Runs the function until we hit smallest wanted length
        branch2(N_P, D_n, angle) # Next point up
        branch2(N_P, D_n, -angle) # Next point down
        
        
if __name__=="__main__":
    # Global variables
    L0 = 50 # Smallest length [micrometers]
    first_point = True
    
    angle = 7*np.pi/36
    L = 500
    S_P = (0,0)
    D = 3
    branch2(S_P, D, angle)
    plt.show()
