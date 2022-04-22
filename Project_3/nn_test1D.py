#from re import U
import torch
import numpy as np
import matplotlib.pyplot as plt


max_iters = 5000


torch.manual_seed(123)

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

T_final = 1
n_residual = 5000
spatial_dim = 1

n_bc = 50
#%%
# Define boundary points and values u(x) on boundary:
#boundary_points = torch.tensor([0., 1.], device=device)
# boundary_values = torch.zeros_like(boundary_points)

#t_boundary = torch.linspace(0, T_final, n_bc, device=device)
#random_ints = torch.randint(high=boundary_points.size(0), size=(n_bc,), device=device)    
#boundary_samples = boundary_points[random_ints]
#boundary_values = torch.zeros_like(boundary_samples)

# Your boundary points are just spatial points, but you want to enforce the BC at all times

#boundary_samples = torch.stack([boundary_samples, t_boundary]).reshape(t_boundary.shape[0], spatial_dim + 1)
#%%

D_init = 1.
D_true = 0.1

def true_solution(xt):
    return torch.sin(np.pi * xt[:, 0]) * torch.exp((-D_true * np.pi**2)* xt[:, -1])
N=20
n_data = 100
pde_w = 100
# Define interior points:
#residual_points = torch.linspace(0, 1, n_residual, device=device)
#t = torch.rand(n_residual, device=device) * T_final

t_data = torch.linspace(0,T_final,N)

data_list = []
input_list = []
for current_time in t_data:
    x_points = torch.linspace(0, 1, n_data)
    t = torch.zeros_like(x_points) + current_time
    xt = torch.vstack([x_points, t]).transpose(1,0)
    assert spatial_dim == 1
    #print('xt',xt)
    u_true = true_solution(xt)
    #plt.plot(u_true)
    data_list.append(u_true)
    input_list.append(xt)
    #print(np.shape(u_true))
    #print(np.shape(xt))

#%%
id_x = torch.randint(0, n_residual -1, (n_residual,))
x_res = torch.linspace(0, 1, n_residual)[id_x]

id_t = torch.randint(0, n_residual -1, (n_residual,))
t_res = torch.linspace(0, T_final, n_residual)[id_t]

xt_res = torch.vstack([x_res, t_res]).transpose(1,0)


#----------------------------------------------------------------
# BZ:
# We need residual_points and t to be in one tensor

#xt = torch.stack([residual_points, t]).reshape(t.shape[0], spatial_dim + 1)

#%%

#print(xt.shape)
# xt[:, 1] is the time
# xt[:, 0] is the spatial coordinate

# Let us try to learn gravitation from the data
# Constrct a Parameter such that PyTorch can optimize it:
 # [0.01, 0.01]

# D_1 = 0.01
# True solution 
# BZ: Evalutate at the "stacked" tensor xt
#true_solution = torch.sin(np.pi * xt[:, 0]) * torch.exp(-D_init * np.pi**2* xt[:, 1])


T_sol = true_solution(xt=xt)
# Initialize with bad guess:
# BZ:
# It looks like you are solving a forward problem here, i.e., you have boundary data and the PDE, and try to solve the PDE.
# Then D needs to be fixed! 
D_param = torch.tensor(D_init, device=device)



solve_inverse = True
if solve_inverse:
    D_param = torch.nn.Parameter(D_param)
    D_param = D_param.to(device)

# CREATE DATA FOR INVERSE PROBLEM
#if solve_inverse:
    # Define interior points:
    #n_data = 100
    #x = torch.linspace(0, 1, n_data, device=device)
    #t = torch.rand(n_data, device=device) * T_final
    #data_coordinates = torch.stack([residual_points, t]).reshape(n_data, spatial_dim + 1)
    #data = true_solution(data_coordinates)



#plt.semilogy(losses)
u_nn = Net(num_hidden_units=16, num_hidden_layers=2, inputs=2).to(device)

#print(list(u_nn.parameters())[0])
#print(list(u_nn.parameters())[1])
#print([D_param])


params = list(u_nn.parameters())  ## + [D_param]

if solve_inverse:
    assert isinstance(D_param, torch.nn.Parameter)
    params = params + [D_param]


loss_function=torch.nn.MSELoss(reduction="mean")

lbfgs_optim = torch.optim.LBFGS(params,
                                max_iter=1000,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-16, tolerance_change=1e-16, max_eval=10)
optimizer = torch.optim.Adam(params=params)

losses = []




def data_loss(nn, input_list, data_list):
    loss = 0.
    # Evaluate the NN at the boundary:
    for input_, data in zip(input_list, data_list):
        predictions = torch.squeeze(nn(input_))
        #print('pred',np.shplt.semilogy(losses)ape(predictions))
        #print('data',np.shape(data))
        loss = loss + loss_function(predictions, data)
    
    return loss


def pde_loss(nn, residual_points):
    
    # We want to compute derivatives with respect to the input:
    residual_points.requires_grad = True
    # t.requires_grad = True
    # Evaluate NN:
    
    u = nn(residual_points) # .squeeze()
    
    ones = torch.ones_like(u)
    #print('u',u.shape)
    #print('resid',residual_points.shape)
    
    # Compute gradients, note the create_graph=True (it defaults to False)

    # here you compute grad u ("defined" as [du/dx, du/dt] in our code), not du_dx
    # du_dx, = torch.autograd.grad(outputs=u,
    grad_u, = torch.autograd.grad(outputs=u,
                             inputs=residual_points,
                             grad_outputs=ones,
                             create_graph=True)
    du_dx = torch.unsqueeze(grad_u[:, 0], -1)
    du_dt = torch.unsqueeze(grad_u[:, -1], -1)

    # breakpoint()

    ddu_dxx1, = torch.autograd.grad(outputs=du_dx,
                                 inputs=residual_points,
                                 grad_outputs=ones,
                                 create_graph=True)
    
    
    ddu_dxx = torch.unsqueeze(ddu_dxx1[:, 0], -1)
    # du_dt, = torch.autograd.grad(outputs=u,
    #                          inputs=residual_points,
    #                          grad_outputs=torch.ones_like(residual_points),
    #                          create_graph=True)




    # breakpoint()

    # The residual corresponding to -d^2 u/ dx^2 = f
    # ---------------------------------------------------------------------------------------------------------
    # BZ you have to use the parameter here:
    residual = du_dt - D_param * ddu_dxx
    #print(residual.shape)
    # Evaluate \sum (-d^2 u/ dx^2 - f - 0)^2 (could also do something like torch.mean(residual ** 2))
    return loss_function(residual, torch.zeros_like(residual))

D_during_train =[]
plt.semilogy(losses)
def closure():
    
    lbfgs_optim.zero_grad()
    
    # Compute losses:
    data_loss_value = data_loss(u_nn,  input_list, data_list)
    
    # BZ: pass the stacked space-time points hereddu_dxx = torch.unsqueeze(ddu_dxx1[:, 0], -1)
    pde_loss_value = pde_loss(u_nn, residual_points=xt_res)
    
    loss = data_loss_value + pde_loss_value
    
    # Compute gradients of the loss w.r.t weights:
    if loss.requires_grad:
        loss.backward()
        
    
    
    # Log the loss:
    losses.append(loss.item())
        
    return loss

e = -1
lambda1 = lambda x: 10 ** (e * x / max_iters)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

# in the for loop:


dloss = []
pdeloss = []


for i in range(max_iters):
    #  Free all intermediate values:
    optimizer.zero_grad()
    
    # Forward pass:
    data_loss_value = data_loss(u_nn,  input_list, data_list)
    #xt_res
    #with torch.no_grad():
     #   xt_res = xt_res * torch.rand_like(xt_res)
    # BZ: pass the stacked space-time points here
    pde_loss_value = pde_loss(u_nn, residual_points=xt_res)
        
    loss = data_loss_value + pde_w*pde_loss_value
    
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
    
#lbfgs_optim.step(closure)

plt.figure()
plt.semilogy(losses, label='total loss')
plt.semilogy(dloss, label='data')
plt.semilogy(pdeloss, label='pde')
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.legend()


plt.figure()
plt.plot(D_during_train)
plt.axhline(y=D_true, linestyle='--', color='k')
plt.ylabel("D")
plt.xlabel("Iteration")

plt.figure()
for i,t in enumerate(t_data):#[0, 0.5, 1.]:
    #x_points = torch.linspace(0, 1, n_residual, device=device)


    xt = torch.zeros((x_points.shape[0], 2))
    xt[:, 0] = x_points
    xt[:, 1] = t

    u_prediction = u_nn(xt)

    plt.plot(x_points, u_prediction.detach().numpy(), label="t=" + format(t, ".2f"), color='r')
    plt.plot(x_points, data_list[i].numpy(),label='t='+ format(t, ".2f"), linestyle='--', color='b' )
    


xt = torch.zeros((x_points.shape[0], 2))
xt[:, 0] = x_points
xt[:, 1] = 0.2

u_prediction = u_nn(xt)

plt.plot(x_points, u_prediction.detach().numpy(), label="t=" + format(0.2, ".2f"), color='k')


plt.xlabel("x")
plt.ylabel("u(x, t)")
#plt.legend()

plt.show()
print(D_param)
print('D_init', D_init)
print('D_true', D_true)