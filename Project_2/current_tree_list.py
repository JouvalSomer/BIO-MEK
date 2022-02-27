import matplotlib.pyplot as plt
import numpy as np

def branch2(S_P, D, angle):
    """
    """
    # Defining global variables
    global L0 # Smallest length of a capillary
    global first_point # True or Fase depending on whether we are on the first point or not

    # Calculating new diameter
    D_n = D * (2**(2/3)) / 2

    # Calculating new angles (for now they are the same)
    # r1 = r2 = D_n/2
    # np.random.choice([r1, r2], size = 2)
    # r0 = D
    # theta_1 = np.arccos((r0**4 + r1**4 - r2**4) * (2*r0**2 * r1**2)**(-1))  
    # theta_2 = np.arccos((r0**4 + r2**4 - r1**4) * (2*r0**2 * r2**2)**(-1))

    # New length
    L = 100*D_n

    # Goes in it True
    if first_point:
        points.append(S_P)
        N_P = (S_P[0] + L, S_P[1])  # Sets the next point; e.g. from (0,0) to (L,0)
        first_point = False

    else:
        N_P = (S_P[0] + L*np.cos(angle), S_P[1] + L*np.sin(angle))
    
    points.append(N_P)

    if L > L0:
        # Runs the function until we hit smallest wanted length
        branch2(N_P, D_n, angle) # Next point up
        branch2(N_P, D_n, -angle) # Next point down
    
        return points


if __name__=="__main__":
    points = []
    angle = 7*np.pi/36
    S_P = (0,0)
    L0 = 50
    first_point = True
    D = 2
    points = branch2(S_P, D, angle)

    for i, point in enumerate(points):
        if i == 0:
            plt.plot(point[0], point[1], 'r.')
        else:
            plt.plot(point[0], point[1], 'b.')
    plt.show()
