import torch
import numpy as np
import matplotlib.pyplot as plt
import os, glob


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
#%%

"PARAMETERS"
max_iters = 15000 #Max iterations

n_pde = int(1e5) #Number of residual points

spatial_dim = 2

D_init = 1. #Intial guess for D

pde_w = 5 #PDE-weights

e = -2 #Step-size factor
    # optimizer.step()
    # scheduler.step()
learning_rate = 1e-2 #learning rate for adams, default 1e-3
learning_rate_D = 1e-3
torch.manual_seed(123) #Seed for rand. functions
# dataset = "brain2dclipp1"
dataset = "brain2dsmooth10"


#%%

''' INIT '''

D_during_train =[]
dloss = []
pdeloss = []
data_list = []
input_list = []
losses = []
counter = 0

#%%

''' Import data'''
    
path_to_data = os.getcwd() + "/data/"



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


#%%    
''' Define grid '''


def make_coordinate_grid(shape=256):

    """ Create a (n x n x 2) array where arr[i,j, :] = (x_i, y_i) is the position of voxel (i,j)"""

    lx = 1
    coordinate_axis =np.linspace(0,lx, shape)
    XX,YY = np.meshgrid(coordinate_axis, coordinate_axis,indexing='ij')
    arr = np.array([XX,YY])

    coordinate_grid = np.swapaxes(arr,0, 1)
    
    coordinate_grid = np.swapaxes(coordinate_grid,1, 2)
    
    dimensions = [133.05135033600635,169.79950394304987]
    
    coordinate_grid[:, :,0] *= dimensions[0]
    
    coordinate_grid[:, :,1] *= dimensions[1]
    
    # dimensions = [133.05135033600635, 169.79950394304987]
    #coordinate_grid = make_coordinate_grid(images)
    return coordinate_grid

coordinate_grid = make_coordinate_grid(shape=256)


def get_domain_coordinates(coordinate_grid, mask):
    return coordinate_grid[mask]   
if dataset == "brain2dsmooth10":
    
    xy = get_domain_coordinates(coordinate_grid, mask=roi)
    
if dataset == "brain2dclipp1":

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
if dataset == "brain2dsmooth10":
    datadict = get_input_output_pairs(coordinate_grid, mask=roi, images=images)
    
if dataset == "brain2dclipp1":
    
    datadict = get_input_output_pairs(coordinate_grid, mask=roi, images=images)
    
#%%
'''Get timedata'''


l = glob.glob(path_to_data +  dataset +  "/concentrations/*")
ts = []
for f in l:
    t = f.split(".npy")[0]
    ts.append(t)

ts[:] = (elem[int(len(path_to_data +  dataset +"/concentrations/")):] for elem in ts)
ts.sort()
ts_i = np.array(ts, dtype=(float)).reshape(np.shape(np.array(ts, dtype=(float)))[0], 1)


#%%

''' Create space-time tensor '''

for current_time in ts:
    
    xyt = torch.tensor(datadict[current_time][0]).float()
    if using_gpu == True:

        xyt = xyt.cuda()

    assert spatial_dim == 2

    u_true = torch.tensor(datadict[current_time][1]).float()
    if using_gpu == True:
    
        u_true = u_true.cuda()
    
    data_list.append(u_true)
    input_list.append(xyt)
    

#%%

''' Residual points '''

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


    def __call__(self, input):

        re = 2 * (input - self.minimum) / (self.maximum - self.minimum) - 1


        return re
    
inputnormalization = InputNormalization(xyt, tmin, tmax)


#%%

''' Optimizer '''

D_param = torch.tensor(D_init, device=device)

solve_inverse = True
if solve_inverse:
    D_param = torch.nn.Parameter(D_param)
    D_param = D_param.to(device)

u_nn = Net(num_hidden_units=32, num_hidden_layers=5, inputs=3, inputnormalization=inputnormalization).to(device)


params = list(u_nn.parameters())

loss_function=torch.nn.MSELoss(reduction="mean")

optimizer = torch.optim.Adam([
                {'params': params, "lr" : learning_rate},
                {'params': D_param, 'lr': learning_rate_D}
            ])


#%%
''' Losses '''

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

lambda1 = lambda x: 10 ** (e * x / max_iters)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

#%%
''' Optimization loop  '''

for i in range(max_iters):
    #  Free all intermediate values:
    optimizer.zero_grad()
    
    # Forward pass:
    data_loss_value = data_loss(u_nn,  input_list, data_list)
    #xt_res
    #with torch.no_grad():
      #   xt_res = xt_res * torch.rand_like(xt_res)
    # BZ: pass the stacked space-time points here
    
    #pde_loss_value = torch.tensor(0.)
    pde_loss_value = pde_w * pderesidual(pde_points, u_nn, D=D_param)
        
    loss = data_loss_value + pde_loss_value
    
    # Backward pass, compute gradient w.r.t. weights and biases
    loss.backward()
    
    D_during_train.append(D_param.item())
    # Log the loss to make a figure
    losses.append(loss.item())
    dloss.append(data_loss_value.item())
    pdeloss.append(pde_loss_value.item())
    
    # Update the weights and biases
    optimizer.step()
    scheduler.step()
    if i % 10 == 0:
        print('iteration = ',i)
        print('Loss = ',loss.item())
        print(f"D = {D_param.item()}")
        # lbfgs_optim.step(closure)


#%%

''' Plot loss during training '''

plt.figure(dpi = 500)
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


'''Plot D during training'''
plt.figure(dpi=500)
#plt.ylim(6e-06, 1e-05)
plt.plot(D_during_train)
plt.ylabel("D")
plt.xlabel("Iteration")
plt.plot(D_during_train)
plt.ylabel("D")
plt.xlabel("Iteration")


#%%
''' Plot true and NN data '''

plt.figure()
for i,t in enumerate(ts):

    xyt_cpu = xyt.cpu()
    plt.figure(dpi=500)
    plt.plot(xyt_cpu[..., 0], xyt_cpu[..., 1], marker=".", linewidth=0, markersize=0.1, color="k")
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.scatter(xyt_cpu[..., 0], xyt_cpu[..., 1], c=datadict[t][1], vmin=0., vmax=1.)
    plt.colorbar()
    plt.figure(dpi=500)
    
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

print("D in SI = ", D_param*1e-06*3600)
print('D_init', D_init)
