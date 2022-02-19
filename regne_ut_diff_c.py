#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:06:23 2022

@author: jacob
"""

from scipy import *
from scipy.special import erfc 
import matplotlib.pyplot as plt 


# dynamic viscosity of water 0.6913 mPa, radius 1.7 A 
# calculate in mm, g, s 


# size in Dalton 
molecule_sizes = { "H2O": 18, "O2": 32,  "Sugar": 180, "Gadovist": 604, "Amyloid": 4500,   "CSF-tau": 45000}  




def stokes_einstein(T=37+273.15, nu=0.0006913 , r=1.4e-7):  
  kb = 1.380649*1.0e-23 * 10**9 
  return kb*T/(6*pi*nu*r)   



#erfc is not in radial coord

# self diffusivity coefficient of water at 37 degree C  

D_self_H20 = 3.03e-3#3.0e-3 
tortuosity = 1.66 
ADC_H20 = D_self_H20*tortuosity**2

print (D_self_H20, ADC_H20) 
print (stokes_einstein())

for molecule in molecule_sizes.keys():  
    r_estimate = pow(molecule_sizes[molecule], 0.3333)/pow(molecule_sizes["H2O"], 0.3333) * 1.4e-7  
    print (molecule , " size ", molecule_sizes[molecule], " r estimate ",  r_estimate, stokes_einstein(r=r_estimate), "mm^/s ",  stokes_einstein(r=r_estimate)/stokes_einstein(), " relative to H2O") 


xx = linspace(0, 200*10**-3, 1000)
tortousity = 1.6 


r_o2 =  pow(molecule_sizes["O2"], 0.3333)/pow(molecule_sizes["H2O"], 0.3333) * 1.4e-7  
D = stokes_einstein(r=r_o2) * tortuosity**2  
analytical_solution_5s_o2= erfc(xx / (2*sqrt( D * 5))) 

r_sugar =  pow(molecule_sizes["Sugar"], 0.3333)/pow(molecule_sizes["H2O"], 0.3333) * 1.4e-7  
D = stokes_einstein(r=r_sugar) * tortuosity**2  
analytical_solution_5s_sugar= erfc(xx / (2*sqrt( D * 5))) 

r_csf_tau =  pow(molecule_sizes["CSF-tau"], 0.3333)/pow(molecule_sizes["H2O"], 0.3333) * 1.4e-7  
D = stokes_einstein(r=r_csf_tau) * tortuosity**2 
analytical_solution_5s_csf_tau= erfc(xx / (2*sqrt( D * 5))) 



plt.gcf().subplots_adjust(bottom=0.25, left=0.25)
plt.ylim(0,1)
plt.rc('xtick', labelsize=24) 
plt.rc('ytick', labelsize=24) 
plt.xlabel("distance [mm]", size=24)
plt.ylabel("concentration", size=24)
plt.plot(xx, analytical_solution_5s_o2, "o", linewidth=7)
plt.plot(xx, analytical_solution_5s_sugar, "r", linewidth=7)
plt.plot(xx, analytical_solution_5s_csf_tau, "b", linewidth=7)
plt.legend(["O2", "Sugar", "CSF-tau"], prop={"size" : 24}, loc=1)
plt.savefig("5s_50micron.png")
plt.show()




