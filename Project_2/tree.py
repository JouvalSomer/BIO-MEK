from typing import Counter
import matplotlib.pyplot as plt
import numpy as np

def diameter_reduction(start_diameter, min_cappilary_diameter):
    """ The function takes in a diameter and returns a list 
        of all diameters from biggest to smallest possible after N iterations with Murry's low.
    Arguments
        start_diameter, int: The start diameter of an artery
    Retrun
        diameters, list: List of all diameters from biggest to smallest
    """
    factor = (2**(2/3)) / 2
    diameters = [] # List of all diameters
    D = start_diameter
    while D > min_cappilary_diameter:
        D = round((D * factor), 14)
        diameters.append(D)
    diameters[-1] = min_cappilary_diameter
    return diameters, len(diameters)

def length_of_cell(diameters):
    """ The function takes in an array of diameters and returns the coresponding lengths of the cells.
    Arguments
        diameters, array: An array of diameters of capilars
    Return
        lengths, array: An array of coresponding lengths of the cells
    """
    lengths = diameters*100
    return lengths

def branch2(S_P, L, angle):
    """
    """
    global L0
    global counter

    if counter == 0:
        N_P = S_P
    else:
        N_P = (S_P[0] + L*np.cos(angle), S_P[1] + L*np.sin(angle))
    plt.plot(N_P[0], N_P[1], 'b.')
    if L > L0:
        counter += 1
        branch2(N_P, L*0.67, angle)
        branch2(N_P, L*0.67, -angle)

# def branch1(angle, start_point, lengths, L0):

#     global counter

#     if lengths[counter] > lengths[-2]:
#         l = lengths[counter]
#         counter += 1

#         # Defining and plotting the starting point
#         start_x = start_point[0]
#         start_y = start_point[1]
#         print('start')
#         plt.plot(start_x, start_y, '.')

#         # Going up from starting point
#         next_x = start_x + l*np.cos(angle)
#         next_y = start_y + l*np.sin(angle)
#         plt.plot(next_x, next_y, '.')
#         print('up')

#         if l > L0:
#             # Calling the function againg with up-point as starting point
#             start_point = (next_x, next_y)
#             branch(angle, start_point, lengths, L0)

#         # Going down from starting point
#         next_x = start_x + l*np.cos(-angle)
#         next_y = start_y + l*np.sin(-angle)
#         plt.plot(next_x, next_y, '.')
#         print('down')
        
#         if l > L0:
#             # Calling the function againg with down-point as starting point
#             start_point = (next_x, next_y)
#             branch(angle, start_point, lengths, L0)


if __name__=="__main__":
    min_cappilary_diameter = 0.1 # 3*1e-6
    start_diameter = 3
    diameters, number_of_iterations = diameter_reduction(start_diameter, min_cappilary_diameter)
    diameters = np.array(diameters)

    lengths = length_of_cell(diameters)
    # l = lengths[0]
    angle = 7*np.pi/36
    # start_point = (1,1)
    L = 500
    S_P = (0,0)
    L0 = 50
    counter = 0
    branch2(S_P, L, angle)
    # branch1(angle, start_point, lengths, L0)

    plt.show()
