#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Juan Barrios
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import Plot as gf

ref_factor = 0
c1 = np.load("marmousi1_rf" + str(ref_factor) + ".npy")
so = 8
so2 = so//2
nx  = c1.shape[1]
nz  = c1.shape[0]  
xmin = 4000.
xmax  = 14000.
zmin = 0
zmax  = 3500.
tmax  = 2500.
dx = (xmax - xmin) / (nx - 1)
dz = (zmax - zmin) / (nz - 1)
x  = (np.arange(nx)*dx)
z  = (np.arange(nz)*dz)
isx = np.where(np.linspace(xmin, xmax, nx) == 8500)[0][0] + so
isz = np.where(np.linspace(zmin, zmax, nz) == 40)[0][0] + so

c = np.zeros((nx + 2*so, nz + 2*so))
c[so:-so, so:-so] = c1[:, :].T

for i in range(so):
    c[i, :] = c[so, :]
    c[:, i] = c[:, so]
    
for i in range(nx + so, nx + 2*so):
    c[i, :] = c[nx + so - 1, :]

for i in range(nz + so, nz + 2*so):
    c[:, i] = c[:, nz + so - 1]

b = 1 / (0.31 * (1e3*c)**0.25)
b[c < 1.51] = 1.0
rho = 1/b
k = c**2 * rho    

cfl = 0.5
dt = dx * cfl / np.max(c)
nt = round(tmax/dt)                     

f0  = 0.005
t0 = 1 / f0
time_values = np.arange(nt)*dt
a = 1
r = (np.pi * f0 * (time_values - t0))
src = a * (1-2.*r**2)*np.exp(-r**2)

v_x = np.zeros((nx + 2*so, nz + 2*so))                                     
v_y = np.zeros((nx + 2*so, nz + 2*so))                                   
p = np.zeros((nx + 2*so, nz + 2*so)) 
receiver = np.zeros((nt, nx))    
                                
start = time.time()
r5 = 1/dx
r6 = 1/dz
for it in range (nt):
    
    r7 = 0.5*(b[4:-5, 4:-5] + b[5:-4, 4:-5])*v_x[4:-5, 4:-5]
    r8 = 0.5*(b[4:-5, 4:-5] + b[4:-5, 5:-4])*v_y[4:-5, 4:-5]
    
    p[8:-8, 8:-8] = dt*(r5*(6.97544643e-4*(r7[:-7, 4:-3] - r7[7:, 4:-3]) 
                            + 9.5703125e-3*(-r7[1:-6, 4:-3] + r7[6:-1, 4:-3]) 
                            + 7.97526042e-2*(r7[2:-5, 4:-3] - r7[5:-2, 4:-3]) 
                            + 1.19628906*(-r7[3:-4, 4:-3] + r7[4:-3, 4:-3])) 
                        + r6*(6.97544643e-4*(r8[4:-3, :-7] - r8[4:-3, 7:]) 
                            + 9.5703125e-3*(-r8[4:-3, 1:-6] + r8[4:-3, 6:-1]) 
                            + 7.97526042e-2*(r8[4:-3, 2:-5] - r8[4:-3, 5:-2]) 
                            + 1.19628906*(-r8[4:-3, 3:-4] + r8[4:-3, 4:-3]))) + p[8:-8, 8:-8]
    # j = so - 1
    # for i in range(so + 1, 2*so + 1):
    #     p[j, :] = p[i, :]
    #     j = j - 1
    
    # j = nx + so - 1
    # for i in range(nx + so, nx + 2*so):
    #     p[i, :] = p[j, :]
    #     j = j -1
        
    j = so - 1
    for i in range(so, 2*so):
        p[:, j] = p[:, i]
        j = j - 1
        
    # j = nz + so - 1
    # for i in range(nz + so, nz + 2*so):
    #     p[:, i] = p[:, j]
    #     j = j - 1
    
    r9 = (c[5:-4, 5:-4]*c[5:-4, 5:-4])*p[5:-4, 5:-4]/b[5:-4, 5:-4]
    
    v_x[8:-8, 8:-8] = dt*r5*(6.97544643e-4*(r9[:-7, 3:-4] - r9[7:, 3:-4]) 
                            + 9.5703125e-3*(-r9[1:-6, 3:-4] + r9[6:-1, 3:-4]) 
                            + 7.97526042e-2*(r9[2:-5, 3:-4] - r9[5:-2, 3:-4]) 
                            + 1.19628906*(-r9[3:-4, 3:-4] + r9[4:-3, 3:-4])) + v_x[8:-8, 8:-8]
    
    v_y[8:-8, 8:-8] = dt*r6*(6.97544643e-4*(r9[3:-4, :-7] - r9[3:-4, 7:]) 
                            + 9.5703125e-3*(-r9[3:-4, 1:-6] + r9[3:-4, 6:-1]) 
                            + 7.97526042e-2*(r9[3:-4, 2:-5] - r9[3:-4, 5:-2]) 
                            + 1.19628906*(-r9[3:-4, 3:-4] + r9[3:-4, 4:-3])) + v_y[8:-8, 8:-8]
    
    p[isx, isz] = p[isx, isz] + dt * src[it] / (dx * dz)
    
    j = so - 1
    for i in range(so, 2*so):
        v_x[:, j] = v_x[:, i]
        v_y[:, j] = -v_y[:, i]
        j = j - 1
        
    # j = nz + so - 1
    # for i in range(nz + so, nz + 2*so):
    #     v_x[:, i] = v_x[:, j]
    #     v_y[:, i] = -v_y[:, j]
    #     j = j - 1
    
    # j = so - 1
    # for i in range(so, 2*so):
    #     v_x[j, :] = -v_x[i, :]
    #     v_y[j, :] = v_y[i, :]
    #     j = j - 1
    
    # j = nx + so - 1
    # for i in range(nx + so, nx + 2*so):
    #     v_x[i, :] = -v_x[j, :]
    #     v_y[i, :] = v_y[j, :]
    #     j = j -1
        
    receiver[it, :] = p[so:-so, isz].T * k[so:-so, isz].T

end = time.time()
comp_time = end - start
domain = [xmin, xmax, zmin, zmax]
gf.plot_image(p[so:-so, so:-so].T * k[so:-so, isz].T, extent = [xmin, xmax, zmax, zmin])
is1 = np.where(np.linspace(xmin, xmax, nx) == 7000)[0][0]
gf.plot_seismic_traces([receiver[:, is1]], 0, tmax)
gf.plot_shotrecord(receiver[:,:], (domain[0],domain[2]), (domain[1], domain[3]), 0, tmax, factor=100)
plt.show()
