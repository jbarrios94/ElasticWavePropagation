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
so = 2
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
dt = np.minimum(dx, dz) * cfl / np.max(c)
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

r4 = 1/dx
r5 = 1/dz
for it in range(nt):
    #----------------------------------Df2-----------------------------------
    p[2:-2, 2:-2] = 0.5*dt*(-r4*(b[1:-3, 2:-2] + b[2:-2, 2:-2])*v_x[1:-3, 2:-2] 
                            + r4*(b[2:-2, 2:-2] + b[3:-1, 2:-2])*v_x[2:-2, 2:-2] 
                            - r5*(b[2:-2, 1:-3] + b[2:-2, 2:-2])*v_y[2:-2, 1:-3] 
                            + r5*(b[2:-2, 2:-2] + b[2:-2, 3:-1])*v_y[2:-2, 2:-2]) + p[2:-2, 2:-2]
    
    # j = so - 1
    # for i in range(so, 2*so):
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
        
    r6 = c[2:-1, 2:-1]*c[2:-1, 2:-1]*p[2:-1, 2:-1]/b[2:-1, 2:-1]
    
    v_x[2:-2, 2:-2] = dt*(r4*(-r6[:-1, :-1]) + r4*r6[1:, :-1]) + v_x[2:-2, 2:-2]
    v_y[2:-2, 2:-2] = dt*(r5*(-r6[:-1, :-1]) + r5*r6[:-1, 1:]) + v_y[2:-2, 2:-2]
    
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
