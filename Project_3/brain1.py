#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:35:22 2022

@author: jacob
"""
import glob
from re import U
import torch
import numpy as np
import matplotlib.pyplot as plt

import os, json

''' Base NN functions '''

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU ", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("Using CPU")

class Net(torch.nn.Module):

    def __init__(self, num_hidden_units, num_hidden_layers, inputs, outputs=1):
        
        super(Net, self).__init__()        
        
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers
        
        # Dimensions of input/output
        self.inputs =  inputs
        self.outputs = outputs
        
        self.input_layer = torch.nn.Linear(self.inputs, self.num_hidden_units)
        
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(
            self.num_hidden_units, self.num_hidden_units)
            for i in range(self.num_hidden_layers - 1)])

        self.output_layer = torch.nn.Linear(self.num_hidden_units, self.outputs)
        
        # Use hyperbolic tangent as activation:
        self.activation = torch.nn.Tanh()
        
    def forward(self, x):
        """[Compute NN output]

        Args:
            x ([torch.Tensor]): input tensor
        Returns:
            [torch.Tensor]: [The NN output]
        """
        # Transform the shape of the Tensor to match what is expected by torch.nn.Linear
        x = torch.unsqueeze(x, 1) 
    
        out = self.input_layer(x)
        
        # The first hidden layer:
        out = self.activation(out)

        # The other hidden layers:
        for i, linearLayer in enumerate(self.linear_layers):
            out = linearLayer(out)
            out = self.activation(out)

        # No activation in the output layer:
        out = self.output_layer(out)

        out = torch.squeeze(out, 1)

        return out
    
#%%
''' Import data'''
    


path_to_data = "/home/jacob/BIO-MEK/BIO-MEK/Project_3/data/"

dataset = "brain2dclipp1"


brainmask = np.load(path_to_data + dataset +  "/masks/mask.npy")
box = np.load(path_to_data + dataset +  "/masks/box.npy")

roi = brainmask * box

def load_images(path_to_data, dataset):
    path_to_concentrations = path_to_data + dataset +  "/concentrations/"
    
    images = {}
    
    for cfile in os.listdir(path_to_concentrations):
        
        c = np.load(path_to_concentrations + cfile)
        
        images[cfile[:-4]] = c
        
    return images
    

images = load_images(path_to_data, dataset)

images.keys()

param_dict ={"a": 1}
param_dict["b"] = 2


exportfolder = "/home/jacob/BIO-MEK/BIO-MEK/Project_3/"
with open(exportfolder + 'my_parameters.json', 'w') as fp:
    json.dump(param_dict, fp, sort_keys=True, indent=4)
    
with open(exportfolder + 'my_parameters.json', 'r') as data_file:    
    loaded_dict = json.load(data_file)
    







#%%    
''' Define grid '''

def make_coordinate_grid(images):
    """ Create a (n x n x 2) array where arr[i,j, :] = (x_i, y_i) is the position of voxel (i,j)"""
    n = 256

    # We want to assign coordinates to every voxel, so the shape of the meshgrid has to be the same as the image
    assert n == images[next(iter(images.keys()))].shape[0]
    assert n == images[next(iter(images.keys()))].shape[1]
    
    coordinate_axis = np.linspace(-0.5, 0.5, n)
    
    XX, YY = np.meshgrid(coordinate_axis, coordinate_axis, indexing='ij')
    
    arr = np.array([XX, YY])

    coordinate_grid = np.swapaxes(arr, 0, 1)
    coordinate_grid = np.swapaxes(coordinate_grid, 1, 2)
    
    return coordinate_grid

coordinate_grid = make_coordinate_grid(images)


def get_domain_coordinates(coordinate_grid, mask):
    return coordinate_grid[mask]   
    
xy = get_domain_coordinates(coordinate_grid, mask=brainmask)

def get_input_output_pairs(coordinate_grid, mask, images):
    
    input_output_pairs = {}
    
    xy = get_domain_coordinates(coordinate_grid, mask)
    
    for timekey, image in images.items():
        
        xyt = np.zeros((xy.shape[0], 3))
        xyt[..., :2] = xy
        xyt[..., -1] = float(timekey)
        
        input_output_pairs[timekey] = (xyt, image[mask])
        
    return input_output_pairs

#datadict = get_input_output_pairs(coordinate_grid, mask=brainmask, images=images)


    
#%%
'''fff'''


l = glob.glob("lister\[0-9]*.txt")

for f in l:
    t = f.split('\\', 1)[1].split(".txt")[0]
    print(t)

n_bc = np.size(ts)

spatial_dim = 2

T = int(float(ts[-1]))

n_residual = xy.shape[0]



#%%


datadict = get_input_output_pairs(coordinate_grid, mask=roi, images=images)

t = "45.60"
xyt = datadict[t][0]
data_at_t = datadict[t][1]

plt.figure(dpi=200)
plt.scatter(xyt[..., 0], xyt[..., 1], c=data_at_t)
# plt.xlim(0, 0.1)
plt.colorbar()
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.show()


