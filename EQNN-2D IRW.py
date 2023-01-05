#%%
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
#import mpld3
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

def parametricSolutions(t, w, nn, tf, wf, x1):
    # parametric solutions 
    N1, N2 = nn(t,w)
    f = C*(t/tf)*(1-t/tf)*(w/wf)*(1-w/wf)
    psi = x1 + f*N1
    #print(psi_hat)
    return psi

def potential(Xs, Ys):
  # Gives the potential at each point
  # Takes in tensor of x points, gives back tensor of V at each point
  kx = 4
  ky = np.sqrt(17)

  Xsnp = Xs.data.numpy()
  Ysnp = Ys.data.numpy()
  Vnp = kx*Xsnp**2/2 + ky*Ysnp**2/2
  Vtorch = torch.from_numpy(Vnp)
  return Vtorch

class EQNN(nn.Module):
    def __init__(self, hidden_dim):
        super(EQNN,self).__init__()
        self.linear1 = nn.Linear(3, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1, bias=None)
        
        self.Ein = nn.Linear(1,1,False)
    
    def forward(self,t,w):
        In1 = self.Ein(torch.ones_like(t))
        h = torch.sin(self.linear1(torch.cat((t,w,In1),-1)))
        h = torch.sin(self.linear2(h))
        h = torch.sin(self.linear3(h))
        return h, C*torch.abs(In1)
    
def hamEqs_Loss(x, y, psi, E, V):
    psi_dx = grad(psi, x, grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    psi_ddx = grad(psi_dx, x, grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    psi_dy = grad(psi, y, grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    psi_ddy = grad(psi_dy, y, grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]
    f = psi_ddx + psi_ddy  + (E-V)*psi
    L  = (f.pow(2)).mean();
    return L

def region(X,Y):
    X_t = tf*torch.rand(n_train, requires_grad=True)
    Y_t = wf*torch.rand(n_train, requires_grad=True)
    return X_t, Y_t

def train(x, y, model, nsteps, t0, tf, w0, wf, BC):
    Lhistory = []
    Ehistory = []
    Elist = []
    solns = []
    oc = 0
    for i in range(nsteps):
        xt, yw = region(x,y)
        xt = xt.reshape((-1,1))
        yw = yw.reshape(-1,1)
        Vsho = potential(xt,yw)
        V = 0
        nn, En = model(xt,yw)
        psi = parametricSolutions(xt, yw, model, tf, wf, BC)
        Loss = hamEqs_Loss(xt, yw, psi,En,V) 
        Loss += (torch.sqrt(torch.dot(psi[:,0],psi[:,0])) - 1)**2
        if Loss < 1e-3 and i >= nsteps/2:
            solns.append(copy.deepcopy(model))
            oc += 1
        if oc == 1:
            Elist.append(En[0].detach().numpy()[0])
        if oc >= 1:
            psi_history = parametricSolutions(xt, yw, solns[0], tf, wf, BC)
            L_ortho = torch.dot(psi_history[:,0], psi[:,0])**2
            Loss += L_ortho
        Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        Lhistory.append(Loss.item())
        Ehistory.append(En[0].detach().numpy()[0])
    return model, Lhistory, Ehistory, Elist

t0 = 0.
tf = 1.
w0 = 0.
wf = np.sqrt(2)
xBC1=0.
n_train = 100
t = torch.linspace(t0,tf,n_train,requires_grad=True)
w = torch.linspace(w0,wf,n_train,requires_grad=True)
C = 3

model = EQNN(150)
steps = int(6e4)
optimizer = optim.Adam(params=model.parameters(), lr = 1e-3, betas=[0.999, 0.9999])
training = train(t, w, model, steps,t0, tf, w0, wf, xBC1)
QNN = training[0]
Lhistory = training[1]
Ehistory = training[2]
nTest = n_train 
tTest = t.reshape(-1,1); wTest = w.reshape(-1,1)
psi = np.zeros((nTest,nTest))

for i in range(len(tTest)):
    for j in range(len(wTest)):
        psi[j][i] = parametricSolutions(tTest[i],wTest[j],training[0],tf,wf,xBC1).detach().numpy()
x = tTest.detach().numpy(); y = wTest.detach().numpy()

#%%
extreme = 1
step_level = 0.2
max_level = + extreme + step_level / 2
min_level = - extreme - step_level / 2

X, Y = np.meshgrid(x,y) 
fig = plt.figure(figsize=(9,9))
ax = plt.axes()
plt.contourf(X, Y, psi/np.max(-psi),     
    norm = mcolors.TwoSlopeNorm(vcenter = 0.), 
    cmap = plt.cm.bwr, 
    levels = np.arange(min_level, max_level + step_level, step_level))
plt.axis('equal')
plt.axis('off')
plt.show()

tsteps = np.linspace(1,len(Lhistory),len(Lhistory))
fig2 = plt.figure(figsize=(9,9))
plt.plot(tsteps,Lhistory,'k')
plt.xlabel('Timesteps')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Loss History for 2D IRW')

fig3 = plt.figure(figsize=(9,9))
plt.plot(tsteps,Ehistory,'r')
plt.xlabel('Timesteps')
plt.ylabel('Energy')
plt.title('Energy History for 2D IRW')
plt.show()
# %%
# def per_diff(u, v):
#     return (np.abs(u-v)/np.abs(v)).mean()*100

# actual = [3.7011, 7.40222, 11.1034, 13.5709, 14.8046, 20.9732]#, 22.2076, 23.4418]#, 27.1429]
# Elist.pop(0)
# fig4 = plt.figure(figsize=(9,9))
# plt.plot(actual,Elist,'r.', markersize=15)
# plt.plot(Ehistory,Ehistory,'b')
# plt.xlabel('Actual Energy')
# plt.ylabel('Calculated Energy')
# plt.title('Calculated v. Actual Energy')
# plt.show()

# result = per_diff(np.array(Elist),np.array(actual))
# print('Percent Difference = {}%'.format(result)) 
# # %%
# def per_diff_non_av(u, v):
#     return (np.abs(u-v)/np.abs(v))*100
# print(per_diff_non_av(np.array(Elist),np.array(actual)))
# # %%
