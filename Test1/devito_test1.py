#==============================================================================
# Pyhton Modules and Imports
#==============================================================================
import numpy             as np
import matplotlib.pyplot as plt
import Plot as gf
#==============================================================================

#==============================================================================
# Devito Imports
#==============================================================================
from devito import *
from examples.seismic.source import RickerSource, Receiver, TimeAxis
from examples.seismic import Model
#==============================================================================

#==============================================================================
# Setup Configuration
#==============================================================================
ref_factor = 0            #mesh refinement (for original mesh ref_factor= 0)
so      = 20              #Space order of finite difference method
nptx    = 300 * (2**ref_factor) + 1  #Number of point in x
nptz    = 100 * (2**ref_factor) + 1  #Number of points in y
x0      = 0.
x1      = 3000.
compx    = x1-x0
z0      = 0.
z1      = 1000.
compz   = z1-z0
hxv     = compx/(nptx-1)
hzv     = compz/(nptz-1)
t0      = 0.0   # Milisencods
tn      = 1000   # Milisencods
f0      = 0.015  # KHertz
nrec    = nptx
nbl     = 0 # int(0.5*nptx)
#==============================================================================

#==============================================================================
# Model Construction
#==============================================================================
cp_top = 1.5
cp_bottom = 5.5
c = np.zeros((nptx,nptz))
c[:] = cp_top  # Top velocity (background)
cp_i = np.linspace(cp_top, cp_bottom, 5)
for i in range(1, 5):
        c[:,i*int(nptz / 5):] = cp_i[i]  # Bottom velocity

b = 1 / (0.31 * (1e3*c)**0.25)
b[c < 1.51] = 1.0
k = c * c / b
model = Model(vp = c, b=b, shape=(nptx,nptz),spacing=(hxv,hzv),nbl=nbl,space_order=so,origin=(x0,z0),extent=(compx,compz))
aspect_ratio      = model.shape[0]/model.shape[1]
#==============================================================================

#==============================================================================
# Symbolic Dimensions
#==============================================================================
x,z  = model.grid.dimensions
xx = x
zz_d = np.linspace(x0, x1, nptz)
t    = model.grid.stepping_dim
time = model.grid.time_dim
s    = time.spacing
isz = np.where(np.linspace(0, z1, nptz) == 20)[0][0]
#==============================================================================
#Parameters
#==============================================================================
ro = 1./model.b
K = model.vp **2 * ro
#==============================================================================
# Time Construction
#==============================================================================
cfl   = 0.25
dt0 = np.minimum(hxv,hzv)*cfl/np.max(c)
nt = int((tn-t0)/dt0)
time_range = TimeAxis(start=t0,stop=tn,num=nt)
#==============================================================================

#==============================================================================
# Ricker Source Construction
src = RickerSource(name='src',grid=model.grid,f0=f0,time_range=time_range)
xsource = x0 + 0.5*compx
zsource = 20.
src.coordinates.data[:] = [xsource,zsource]
#==============================================================================
# Symbolic Fields Construction
#==============================================================================
e = TimeFunction(name='p', grid = model.grid, staggered = NODE, space_order = so, time_order = 2)
m = VectorTimeFunction(name='v', grid = model.grid, space_order = so, time_order =2)
#==============================================================================

#==============================================================================
# Source Term Construction
src_p = src.inject(field = e.forward, expr = s * src/(hxv * hzv))
#==============================================================================
# Receiver Term Construction
#==============================================================================
rec = Receiver(name="rec",grid=model.grid,npoint=nrec,time_range=time_range)
rec.coordinates.data[:,0] = np.linspace(x0,x0+compx,num=nrec)
rec.coordinates.data[:,1] = 20.
rec_term = rec.interpolate(expr = e.forward * K)
#==============================================================================

#==============================================================================
# Symbolic Equation Construction
#==============================================================================
Eqe = Eq(e.forward, e + s * div(m/ro))
Eqm = Eq(m.forward, m + s * grad(e.forward*K))
#==============================================================================

# =============================================================================
#Boundary Conditions
#==============================================================================
bcm = []
bce = []

j = - 1
for i in range(so):
    bcm.append(Eq(m[0][t+1, x, j],m[0][t, x, i]))
    bcm.append(Eq(m[1][t+1, x, j],-m[1][t, x, i]))
    bce.append(Eq(e[t+1, x, j], e[t, x, i]))
    j = j - 1

j = nptz - 1
for i in range(nptz, nptz + so):
    bcm.append(Eq(m[0][t+1, x, i], m[0][t+1, x, j]))
    bcm.append(Eq(m[1][t+1, x, i], -m[1][t+1, x, j]))
    bce.append(Eq(e[t+1, x, i], e[t+1, x, j]))
    j = j - 1
#==============================================================================

#==============================================================================
# Operator Definition
#==============================================================================
op2 = Operator([Eqe] + bce + [Eqm] + bcm + src_p + rec_term)
#==============================================================================

#==============================================================================
# Operator Evolution
#==============================================================================
op2(dt=dt0,time=nt - 1)
#==============================================================================

#==============================================================================
# Graphical Plots
#==============================================================================

domain = [x0, x1, x0, x1]
gf.plot_image(e.data[0, :, :].T * k.T, extent = [x0, x1, z1, z0])
is1 = np.where(np.linspace(x0, x1, nptx) == 1330)[0][0]
gf.plot_seismic_traces([rec.data[:, is1]], 0, tn)
gf.plot_shotrecord(rec.data[:,:], (domain[0],domain[2]), (domain[1], domain[3]), 0, tn, factor=100)
plt.show()
