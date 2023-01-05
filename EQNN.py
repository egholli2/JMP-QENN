#%%
import numpy as np
import torch
import torch.optim as optim
from math import pi
from torch.autograd import grad
from torch.autograd import Variable
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time
import copy
from scipy.integrate import odeint
import torch.nn as nn
import torch.nn.functional as F
dtype=torch.float

def L2_loss(u, v):
  return ((u-v)**2).mean()

def perturbPoints(grid,t0,tf,sig):
#   stochastic perturbation of the evaluation points
#   force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]  
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    t.data[2] = torch.ones(1,1)*(-1)
    t.data[t<t0]=t0 - t.data[t<t0]
    t.data[t>tf]=2*tf - t.data[t>tf]
    t.data[0] = torch.ones(1,1)*t0

    t.data[-1] = torch.ones(1,1)*tf
    #t.requires_grad = True
    return t

def parametricSolutions(t, nn, tf, x1):
    # parametric solutions 
    N1,N2 = nn(t)
    dt =t-t0
#### THERE ARE TWO PARAMETRIC SOLUTIONS. Uncomment f=dt 
    #f = (t/tf)*(1-t/tf)
    f = (t-tf)*(t+tf)/tf
#     f=dt
    psi_hat  = x1  + f*N1
    return psi_hat

def SHOpotential(Xs):
  # Gives the potential at each point
  # Takes in tensor of x points, gives back tensor of V at each point
  k = 4

  Xsnp = Xs.data.numpy()
  Vnp = k*Xsnp**2/2
  Vtorch = torch.from_numpy(Vnp)
  return Vtorch

class EQNN(nn.Module):
    def __init__(self, hidden_dim):
        super(EQNN,self).__init__()
        self.linear1 = nn.Linear(2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1, bias=None)
        
        self.Ein = nn.Linear(1,1)

    def forward(self, t):
        In1 = self.Ein(torch.ones_like(t))
        h = torch.sin(self.linear1(torch.cat((t,In1),1)))
        h = torch.sin(self.linear2(h))
        h = torch.sin(self.linear3(h))
        return h, In1

def hamEqs_Loss(x, psi, E, V):
    psi_dx = grad(psi, x, grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    psi_ddx = grad(psi_dx, x, grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    f = psi_ddx/2 + (E-V)*psi
    L  = (f.pow(2)).mean();
    return L

def train(points, model, nsteps, t0, tf, BC):
    print('Training...')
    TeP0 = time.time()
    oc = 0
    Lhistory = []
    Ehistory = []
    Elist = []
    solns = []
    for i in range(nsteps):
        r = perturbPoints(points, t0, tf, sig=0.03*tf)
        r = r.reshape((-1,1))
        V = SHOpotential(r)
        nn, En = model(r)
        psi = parametricSolutions(r, model, tf, BC)
        Loss = hamEqs_Loss(r,psi,En,V)
        Loss += (torch.sqrt(torch.dot(psi[:,0],psi[:,0])) - 1)**2
        if Loss < 1e-3 and i >= nsteps/2:
            solns.append(copy.deepcopy(model))
            oc += 1
        if oc == 1:
            Elist.append(En[0].detach().numpy()[0])
        if oc >= 1:
            psi_history = parametricSolutions(r, solns[0], tf, BC)
            L_ortho = torch.dot(psi_history[:,0], psi[:,0])**2
            Loss += L_ortho
        Loss.backward()
        Ehistory.append(En[0].data.tolist()[0])
        optimizer.step()
        optimizer.zero_grad()
        Lhistory.append(Loss.item())
    print('Done!')
    TePf = time.time()
    runTime = TePf - TeP0
    print('Time:',runTime)
    return model, Lhistory, Ehistory

t0 = -4.
tf = 4.
xBC1=0.
n_train = 100
nsteps = int(6e4)
x = torch.linspace(t0,tf,n_train,requires_grad=True)
t = np.linspace(1, nsteps, nsteps)

model = EQNN(50)
optimizer = optim.Adam(params=model.parameters(), lr = 0.008, betas = [0.999, 0.9999])
training = train(x, model, nsteps, t0, tf, xBC1)

#%%
nTest = n_train; tTest = torch.linspace(t0-.1,tf+.1,nTest)
tTest = tTest.reshape(-1,1);
tTest.requires_grad=True
t_net = tTest.detach().numpy()

#%%
psi =parametricSolutions(tTest,training[0],tf,xBC1)
psi=psi.data.numpy()
#tru = np.sin(np.pi*t_net)*np.max(-1*psi)
#plt.plot(t_net, tru, '-r', linewidth = 1, label = 'True')
plt.figure()
plt.xlim(t0,tf)
plt.plot(t_net, psi, '-b', linewidth=1, label = 'ANN')
plt.legend()
plt.plot(t_net, np.zeros(len(t_net)),'--k', linewidth=3)
plt.xlabel('x')
plt.ylabel('$\psi(x)$')
plt.grid('on')

plt.figure()
plt.plot(t, training[1], 'r')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Loss History')

plt.figure()
plt.plot(t, training[2], 'g')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.title('Energy History')
plt.show()


#%%
