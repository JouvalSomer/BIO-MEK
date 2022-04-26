#from re import U
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os, json, glob


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU ", torch.cuda.get_device_name())
    using_gpu = True
else:
    device = torch.device("cpu")
    print("Using CPU")
    using_gpu = False


class Net(torch.nn.Module):

    def __init__(self, num_hidden_units, num_hidden_layers, inputs, outputs=1, inputnormalization=None):
        
        super(Net, self).__init__()        
        
        self.num_hidden_units = num_hidden_units
        self.num_hidden_layers = num_hidden_layers
        self.inputnormalization = inputnormalization
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
        if self.inputnormalization is not None:
            x = inputnormalization(x)
        x = torch.unsqueeze(x, 1) 

        # x[..., -1] = x[..., -1] / tmax
    
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

"PARAMETERS"

max_iters = 500

n_pde = int(1e6)
T_final = 1
n_residual = 5000
spatial_dim = 2

D_init = 0.0001
D_true = 0.1
N=20
n_data = 100
pde_w = 1

torch.manual_seed(123)

use_true_solution = False
use_scan_data = True


#%%
if use_scan_data == True:
    ''' Import data'''
        
    
    
    path_to_data = "/home/lemmet/Documents/Python_Scripts/BIO-MEK/Project_3/data/"
    
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
    
    
    # exportfolder = "/home/jacob/BIO-MEK/BIO-MEK/Project_3/"
    # with open(exportfolder + 'my_parameters.json', 'w') as fp:
    #     json.dump(param_dict, fp, sort_keys=True, indent=4)
        
    # with open(exportfolder + 'my_parameters.json', 'r') as data_file:    
    #     loaded_dict = json.load(data_file)
        


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

#%%
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
    
    datadict = get_input_output_pairs(coordinate_grid, mask=roi, images=images)
    
    '''fff'''


    l = glob.glob(path_to_data +  dataset +  "/concentrations/*")
    ts = []
    for f in l:
        t = f.split(".npy")[0]
        
        ts.append(t)
        
    
    ts[:] = (elem[int(len(path_to_data +  dataset +"/concentrations/")):] for elem in ts)
    
    
    ts.sort()
    ts_i = np.array(ts, dtype=(float)).reshape(np.shape(np.array(ts, dtype=(float)))[0], 1)


#%%



    data_list = []
    input_list = []
    
    
    for current_time in ts:
        
        xyt = torch.tensor(datadict[current_time][0]).float()
        if using_gpu == True:
        # breakpoint()
            xyt = xyt.cuda()
    
        assert spatial_dim == 2
    
        u_true = torch.tensor(datadict[current_time][1]).float()
        if using_gpu == True:
        
            u_true = u_true.cuda()
    
        data_list.append(u_true)
        input_list.append(xyt)
        
if use_true_solution == True:

    def true_solution(xyt):
        return torch.sin(np.pi * xyt[:, 0]) * torch.sin(np.pi * xyt[:, 1]) *  torch.exp((-D_true * 2*np.pi**2)* xyt[:, -1])



    t_data = torch.linspace(0,T_final,N)
#%%


def init_collocation_points(coords, num_points, t_max, t_min ):
    with torch.no_grad():

        assert len(coords.shape) == 2, "Assert mask has been applied"

        random_ints = torch.randint(high=coords.size(0), size=(num_points,), device=coords.device)    
        coords = coords[random_ints, :]
    
        a = (np.random.rand(coords.shape[0]))
        
        random_times = torch.from_numpy(a).to(coords.device)
        t = (random_times * (t_max - t_min) + t_min)

        coords[..., -1] = t

        print("Initialized collocation points with mean t = ",
            format(torch.mean(t).item(), ".2f"),
            ", min t = ", format(torch.min(t).item(), ".2f"),
            ", max t = ", format(torch.max(t).item(), ".2f"))

    return coords



tmax = float(max(datadict.keys()))
tmin = float(min(datadict.keys()))

pde_points = init_collocation_points(xyt, num_points=int(1e5), t_max=tmax, t_min=tmin)



class InputNormalization():

    def __init__(self, xyt, tmin, tmax):

        self.maximum = torch.max(xyt, axis=0)[0]
        self.minimum = torch.min(xyt, axis=0)[0]
        self.maximum[..., -1] = tmax
        self.minimum[..., -1] = tmin

        # breakpoint()

    def __call__(self, input):

        re = 2 * (input - self.minimum) / (self.maximum - self.minimum) - 1
        # re = (input - self.minimum) / (self.maximum - self.minimum)

        # print(torch.min(re), torch.max(re))

        return re
    
inputnormalization = InputNormalization(xyt, tmin, tmax)


#%%

D_param = torch.tensor(D_init, device=device)

solve_inverse = True
if solve_inverse:
    D_param = torch.nn.Parameter(D_param)
    D_param = D_param.to(device)

u_nn = Net(num_hidden_units=32, num_hidden_layers=5, inputs=3, inputnormalization=inputnormalization).to(device)


params = list(u_nn.parameters())

if solve_inverse:
    assert isinstance(D_param, torch.nn.Parameter)
    params = params + [D_param]


loss_function=torch.nn.MSELoss(reduction="mean")

lbfgs_optim = torch.optim.LBFGS(params,
                                max_iter=1000,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-16, tolerance_change=1e-16)#, max_eval=10)
# optimizer = torch.optim.Adam(params=params)

losses = []




def data_loss(nn, input_list, data_list):
    loss = 0.
    # Evaluate the NN at the boundary:
    for input_, data in zip(input_list, data_list):
        predictions = torch.squeeze(nn(input_))
        loss = loss + loss_function(predictions, data)
        
    return loss


def pderesidual(coords, nn, D):
        """
        coords = pde_points
        nn = neural network
        D = diffusion coefficient
        
        """
        
        assert isinstance(D, torch.nn.Parameter)
        assert coords.shape[-1] == 3, "your array should have size N x 3"
        
        coords.requires_grad = True
        output = nn(coords).squeeze()

        ones = torch.ones_like(output)

        output_grad, = torch.autograd.grad(outputs=output,
                                        inputs=coords,
                                        grad_outputs=ones,
                                        create_graph=True)
        doutput_dt = output_grad[..., -1]
        doutput_dx = output_grad[..., 0]
        doutput_dy = output_grad[..., 1]
        
        ddoutput_dxx, = torch.autograd.grad(outputs=doutput_dx,
                                            inputs=coords,
                                            grad_outputs=ones,
                                            create_graph=True)

        ddoutput_dyy, = torch.autograd.grad(outputs=doutput_dy,
                                            inputs=coords,
                                            grad_outputs=ones,
                                            create_graph=True)

        ddoutput_dxx = ddoutput_dxx[..., 0]
        ddoutput_dyy = ddoutput_dyy[..., 1]

        laplacian = (ddoutput_dxx + ddoutput_dyy)

        residual = doutput_dt - D * laplacian

        assert output.shape == residual.shape

        return loss_function(residual, torch.zeros_like(residual))

D_during_train =[]

dloss = []
pdeloss = []
counter = 0
def closure():
    global counter
    lbfgs_optim.zero_grad()
    
    # Compute losses:
    data_loss_value = data_loss(u_nn,  input_list, data_list)
    
    # BZ: pass the stacked space-time points hereddu_dxx = torch.unsqueeze(ddu_dxx1[:, 0], -1)
    pde_loss_value = pderesidual(pde_points, u_nn, D=D_param)
    
    loss = data_loss_value + pde_loss_value
    
    # Compute gradients of the loss w.r.t weights:
    if loss.requires_grad:
        loss.backward()
        
    
    dloss.append(data_loss_value.item())
    pdeloss.append(pde_loss_value.item())
    # Log the loss:
    losses.append(loss.item())
    D_during_train.append(D_param.item())
    print(counter)
    counter = counter + 1
    
    return loss

e = -1
lambda1 = lambda x: 10 ** (e * x / max_iters)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# in the for loop:




# for i in range(max_iters):
#     #  Free all intermediate values:
#     optimizer.zero_grad()
    
#     # Forward pass:
#     data_loss_value = data_loss(u_nn,  input_list, data_list)
#     #xt_res
#     #with torch.no_grad():
#      #   xt_res = xt_res * torch.rand_like(xt_res)
#     # BZ: pass the stacked space-time points here
    
#     #pde_loss_value = torch.tensor(0.)
#     pde_loss_value = pde_w * pderesidual(pde_points, u_nn, D=D_param)
        
#     loss = data_loss_value + pde_loss_value
    
#     # Backward pass, compute gradient w.r.t. weights and biases
#     loss.backward()
    
#     D_during_train.append(D_param.item())
#     # Log the loss to make a figure
#     losses.append(loss.item())
#     dloss.append(data_loss_value.item())
#     pdeloss.append(pde_loss_value.item())
    
#     # Update the weights and biases
#     optimizer.step()
#     scheduler.step()
#     print(i)
#     print(f"D = {D_param.item()}")
lbfgs_optim.step(closure)

plt.figure()
plt.semilogy(losses, label='total loss')

dloss = torch.tensor(dloss)
dloss = dloss.cpu()
pdeloss = torch.tensor(pdeloss)
pdeloss = pdeloss.cpu()
plt.semilogy(dloss, label='data')
plt.semilogy(pdeloss, label='pde')
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.legend()


plt.figure()
plt.plot(D_during_train)
plt.ylabel("D")
plt.xlabel("Iteration")

plt.figure()
for i,t in enumerate(ts):

    # if i != 10:
    #     continue

    if i % 2 == 0:
        xyt_cpu = xyt.cpu()
        plt.figure(dpi=200)
        plt.plot(xyt_cpu[..., 0], xyt_cpu[..., 1], marker=".", linewidth=0, markersize=0.1, color="k")
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.scatter(xyt_cpu[..., 0], xyt_cpu[..., 1], c=datadict[t][1], vmin=0., vmax=1.)
        plt.colorbar()
        plt.figure(dpi=200)
        
        xyt[:, -1] = float(t)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        
        
        u_prediction=u_nn(xyt)
        
        xyt = xyt.cpu()
        u_prediction = u_prediction.cpu()
        
        plt.plot(xyt[..., 0], xyt[..., 1], marker=".", linewidth=0, markersize=0.1, color="k")
        plt.scatter(xyt[..., 0], xyt[..., 1], c=np.squeeze(u_prediction.detach().numpy(),1), vmin=0., vmax=1.)
        plt.colorbar()
        if using_gpu == True:
            xyt = xyt.to(device)
            u_prediction = u_prediction.to(device)
   

plt.show()

print(D_param)
print('D_init', D_init)
