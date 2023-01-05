#%%
from cmath import pi
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import time
import copy
from scipy.integrate import odeint
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D 

dtype=torch.float

def L2_loss(u, v):
  return ((u-v)**2).mean()


def perturbPoints(grid,t0,tf,sig=0.5):
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
    #t.requires_grad = False
    return t

def region(X,Y):
    R = torch.sqrt((X/tf)**2 + (Y/wf)**2)/np.sqrt(2)
    Theta = torch.linspace(0,2*pi,n_train,requires_grad=True)
    New_x = torch.sqrt(R)*torch.cos(Theta)
    New_y = torch.sqrt(R)*torch.sin(Theta)
    return torch.sort(New_x)[0], torch.sort(New_y)[0]


def parametricSolutions(t, w, nn, x1):
    # parametric solutions 
    N1 = nn(t,w)[0]
    f = C*(1-(t/tf)**2-(w/wf)**2)
    psi  = x1  + f*N1
    return psi


class EQNN(nn.Module):
    def __init__(self, hidden_dim):
        super(EQNN,self).__init__()
        self.linear1 = nn.Linear(3, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1, bias=None)
        
        self.Ein = nn.Linear(1,1)
    
    def forward(self,t,w):
        In1 = self.Ein(torch.ones_like(t))
        h = torch.sin(self.linear1(torch.cat((t,w,In1),-1)))
        h = torch.sin(self.linear2(h))
        h = torch.sin(self.linear3(h))
        return h, C*torch.abs(In1)
    
def hamEqs_Loss(x, y, psi, E, V):
    psi_dx = grad(psi, x, grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    psi_ddx = grad(psi_dx, x, grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    psi_dy = grad(psi, y, grad_outputs=torch.ones(y.shape, dtype=dtype), create_graph=True)[0]
    psi_ddy = grad(psi_dy, y, grad_outputs=torch.ones(y.shape, dtype=dtype), create_graph=True)[0]
    f = psi_ddx + psi_ddy + (E-V)*psi
    L  = (f.pow(2)).mean();
    return L

def region(X,Y):
    New_x = []
    New_y = []
    j = 0
    while j != n_train-1:
        x_0 = 4*np.random.rand()-2
        y_0 = 4*np.random.rand()-2
        if x_0**2 + (y_0**2)/wf**2 <= 1:
            New_x.append(x_0)
            New_y.append(y_0)
            j += 1
        else:
            continue
    X_t = torch.tensor(New_x, requires_grad = True)
    Y_t = torch.tensor(New_y, requires_grad = True)
    return X_t, Y_t

def train(t, w, model, nsteps, t0, tf, w0, wf, BC):
    print('Training...')
    walle = 1
    V = 0
    Lhistory = []
    Enhistory = []
    solns = []
    oc = 0
    for i in range(nsteps):
        xt = perturbPoints(x, t0, tf)
        yw = perturbPoints(y, w0, wf)
        X, Y = region(xt,yw)
        X = X.reshape((-1,1))
        Y = Y.reshape((-1,1))
        # xt = xt.reshape((-1,1))
        # yw = yw.reshape((-1,1))
        nn, En = model(X,Y)
        psi = parametricSolutions(X, Y, model, BC)
        Loss = hamEqs_Loss(X,Y,psi,En,V) 
        #Loss += (torch.sqrt(torch.dot(psi[:,0],psi[:,0])) - 1)**2
        Loss += (torch.dot(psi[:,0],psi[:,0]) - 1)**2
        if Loss < 1e-4 and i >= nsteps/2:
            solns.append(copy.deepcopy(model))
            oc += 1
        if oc >= 1:
            psi_history = parametricSolutions(X, Y, solns[0], BC)
            L_ortho = torch.dot(psi_history[:,0], psi[:,0])**2
            Loss += L_ortho
        Lhistory.append(Loss.item())
        Enhistory.append(En[0].data.tolist()[0])
        Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print('Done!')
    return model, Lhistory, Enhistory


xBC1=0.
n_train = 100
t0 = -1.
tf = 1.
w0 = -np.sqrt(2)
wf = -w0
x = torch.linspace(t0,tf,n_train,requires_grad=True)
y = torch.linspace(w0,wf,n_train,requires_grad=True)
C = 1

model = EQNN(150)
optimizer = optim.Adam(params=model.parameters(), lr = 5e-4, betas=[0.999, 0.9999])
steps = int(8e4)
learning = train(x, y, model, steps,t0, tf, w0, wf, xBC1)
training = learning[0]
Lhistory = learning[1]
Ehistory = learning[2]
t = np.linspace(1,len(Lhistory),len(Lhistory))
nTest = n_train

#%%
tTest = x.reshape(-1,1); wTest = y.reshape(-1,1)
psi = np.zeros((nTest,nTest))

for i in range(len(tTest)):
    for j in range(len(wTest)):
        psi[j][i] = parametricSolutions(tTest[i],wTest[j],training,xBC1).detach().numpy()
        if (tTest[i]/tf)**2 + (wTest[j]/wf)**2 > 1:
            psi[j][i] = 0

x = x.detach().numpy(); y = y.detach().numpy()
#psi = psi
#%%        
extreme = 1
step_level = 0.2
max_level = + extreme + step_level / 2
min_level = - extreme - step_level / 2

X, Y = np.meshgrid(x,y) 
fig = plt.figure(figsize=(9,9))
ax = plt.axes()
plt.contourf(X, Y, -psi/np.max(-psi),     
    norm = mcolors.TwoSlopeNorm(vcenter = 0.), 
    cmap = plt.cm.bwr, 
    levels = np.arange(min_level, max_level + step_level, step_level))
plt.axis('equal')
plt.axis('off')
plt.show()

fig2 = plt.figure(figsize=(9,9))
plt.plot(t,Lhistory,'b')
plt.yscale("log")
plt.xlabel('Timesteps')
plt.ylabel('Loss')
plt.title('Loss History for IEW')

fig3 = plt.figure(figsize=(9,9))
plt.plot(t,Ehistory,'r')
plt.xlabel('Timesteps')
plt.ylabel('Energy')
plt.title('Energy History for IEW')
#plt.xscale('log')
plt.show()

#%%
circle_actual = [5.78323, 14.6827, 26.3784, 30.4787, 40.7209]