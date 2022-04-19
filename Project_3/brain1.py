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
    
xy = get_domain_coordinates(coordinate_grid, mask=roi)

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


l = glob.glob(path_to_data +  dataset +  "/concentrations/*")
ts = []
for f in l:
    t = f.split(".npy")[0]
    
    ts.append(t)
    #print(t)

ts[:] = (elem[int(len(path_to_data +  dataset +"/concentrations/")):] for elem in ts)


ts.sort()
ts_i = np.array(ts, dtype=(float))



n_bc = np.size(ts)

spatial_dim = 2

T = float(ts[-1])

n_residual = xy.shape[0]


D_init = 0.01

D_param = torch.tensor(D_init, device=device)

u_nn = Net(num_hidden_units=16, num_hidden_layers=2, inputs=3).to(device)

params = list(u_nn.parameters())   + [D_param]


datadict = get_input_output_pairs(coordinate_grid, mask=roi, images=images)



#xyt = torch.stack([torch.from_numpy(xy), torch.from_numpy(ts_i)],dim=2).reshape(ts_i.shape[0], spatial_dim + 1)


#xyt = datadict[t][0]


#data_at_t = datadict[t][1]
#%%

loss_function=torch.nn.MSELoss(reduction="mean")

lbfgs_optim = torch.optim.LBFGS(params,
                                max_iter=1000,
                                line_search_fn="strong_wolfe")

losses = []




def pde_loss(nn, residual_points):
    
    # We want to compute derivatives with respect to the input:
    residual_points.requires_grad = True
    # t.requires_grad = True
    # Evaluate NN:
    
    u = nn(residual_points) # .squeeze()
    
    ones = torch.ones_like(u)
    print(u.shape)
    print(residual_points.shape)
    
    # Compute gradients, note the create_graph=True (it defaults to False)

    # here you compute grad u ("defined" as [du/dx, du/dt] in our code), not du_dx
    # du_dx, = torch.autograd.grad(outputs=u,
    grad_u, = torch.autograd.grad(outputs=u,
                             inputs=residual_points,
                             grad_outputs=ones,
                             create_graph=True)
    du_dx = torch.unsqueeze(grad_u[:, 0], -1)
    du_dy = torch.unsqueeze(grad_u[:, 1], -1)
    du_dt = torch.unsqueeze(grad_u[:, -1], -1)

    # breakpoint()

    ddu_dxx, = torch.autograd.grad(outputs=du_dx,
                                 inputs=residual_points,
                                 grad_outputs=ones,
                                 create_graph=True)
    ddu_dyy, = torch.autograd.grad(outputs=du_dy,
                                 inputs=residual_points,
                                 grad_outputs=ones,
                                 create_graph=True)





    # breakpoint()

    # The residual corresponding to -d^2 u/ dx^2 = f
    # ---------------------------------------------------------------------------------------------------------
    # BZ you have to use the parameter here:
    residual = du_dt - D_param * ddu_dxx - D_param * ddu_dyy
    print(residual.shape)
    # Evaluate \sum (-d^2 u/ dx^2 - f - 0)^2 (could also do something like torch.mean(residual ** 2))
    return loss_function(residual, torch.zeros_like(residual))


def closure():
    
    lbfgs_optim.zero_grad()
    
    # Compute losses:
    #boundary_loss_value = boundary_loss(u_nn, boundary_points=boundary_samples, boundary_values=boundary_values)
    
    # BZ: pass the stacked space-time points here
    pde_loss_value = pde_loss(u_nn, residual_points=xyt)
    
    loss =  pde_loss_value
    
    # Compute gradients of the loss w.r.t weights:
    if loss.requires_grad:
        loss.backward()
    
    # Log the loss:
    losses.append(loss.item())
        
    return loss


lbfgs_optim.step(closure)

plt.figure()
plt.semilogy(losses)
plt.ylabel("Loss")
plt.xlabel("Iteration")







#%%

for i in ts:

    xyt = datadict[i][0]
    data_at_t = datadict[i][1]

    plt.figure(dpi=200)
    plt.scatter(xyt[..., 0], xyt[..., 1], c=data_at_t)
    # plt.xlim(0, 0.1)
    plt.colorbar()
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title(i)
    plt.show()


