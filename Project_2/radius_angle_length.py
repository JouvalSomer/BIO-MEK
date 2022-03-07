import numpy as np

def new_radius(r):
    """
    The function calculates the new radii in a bifurcation according to Murray's law.

        Arguments:
            r - float: The starting radius.
        
        Return:
            (r1, r2) - tuple - (float, float):
                A tuple consisting of the two new radii, r1 and r2, respectively.
    """
    r1 = r2 = r * (2**(2/3)) / 2
    return (r1, r2)

def new_angles(r0, r1, r2):
    """
    The function calculates the new angles in a bifurcation according to Murray's law.

        Arguments:
            r0 - float: The starting radius.
            r1 - float: The first new radii.
            r2 - float: The second new radii.
        
        Return:
            (theta_1, theta_2) - tuple - (float, float):
                A tuple consisting of the two new angles, theta_1 and theta_2, respectively.
    """
    theta_1 = np.arccos((r0**4 + r1**4 - r2**4) * (2*r0**2 * r1**2)**(-1))  
    theta_2 = np.arccos((r0**4 + r2**4 - r1**4) * (2*r0**2 * r2**2)**(-1))

    return (theta_1, theta_2)

def length(r):
    """
    This function calculates the langth of a cell.

    Arguments:
        r - float: The radius of the cell.
        
    Return:
        L - float: The lengths of the cell
    """
    L = 100*2*r
    return L
