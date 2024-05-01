# -*- coding: utf-8 -*-
"""
@author: Juan Barrios
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import Plot as gf

def Domain(LimInfx, LimSupx, LimInfy, LimSupy, NumVolx, NumVoly):
    dx = (LimSupx - LimInfx)/(NumVolx - 1)
    dy = (LimSupy - LimInfy)/(NumVoly - 1)
    xn = np.linspace(LimInfx, LimSupx + dx, NumVolx + 1)
    yn = np.linspace(LimInfy, LimSupy + dy, NumVoly + 1)
    xc = np.zeros(NumVolx + 4)
    yc = np.zeros(NumVoly + 4)
    xc[2:-2] = xn[:-1]
    yc[2:-2] = yn[:-1]
    Xc, Yc = np.meshgrid(xc,yc)
    Xn, Yn = np.meshgrid(xn, yn)
    return Xc, Yc, Xn, Yn, dx, dy


def Riemann_Elasticity_2Dx(q, rho, k, c):
    r11 = 1/np.sqrt(k*rho)[:,:-1]
    r13 = -1/np.sqrt(k*rho)[:,1:]
    dfq1 = -((q[1]/rho)[:,1:] - (q[1]/rho)[:,:-1])
    dfq2 = -((q[0]*k)[:,1:] - (q[0]*k)[:,:-1])
    betha1 = (dfq1 - r13*dfq2)/(r11 - r13)
    betha3 = (-dfq1 + r11*dfq2)/(r11 - r13)
    W1 = np.zeros(dimx)
    W3 = np.zeros(dimx)
    W1[0] = betha1*r11
    W1[1] = betha1
    W3[0] = betha3*r13
    W3[1] = betha3
    return W1, W3


def Riemann_Elasticity_2Dy(q, rho, k, c):
    r11 = 1/np.sqrt(k*rho)[:-1,:]
    r13 = -1/np.sqrt(k*rho)[1:,:]
    dgq1 = -((q[2]/rho)[1:,:] - (q[2]/rho)[:-1,:])
    dgq3= -((q[0]*k)[1:,:] - (q[0]*k)[:-1,:])
    betha1 = (dgq1 - r13*dgq3)/(r11 - r13)
    betha3 = (-dgq1 + r11*dgq3)/(r11 - r13)
    W1 = np.zeros(dimy)
    W3 = np.zeros(dimy)
    W1[0] = betha1*r11
    W1[2] = betha1
    W3[0] = betha3*r13
    W3[2] = betha3
    return W1, W3


def Fimm2Dx(s, W1, W2):
    dim = np.shape(W1)
    F = np.zeros((dim[0], dim[1],dim[2] - 2))
    norm = np.sum(W1[:,:,1:-1]**2,0)
    norm += (norm == 0. ) * 1.
    theta = np.sum(W1[:,:,2:]*W1[:,:,1:-1],0)/norm
    W1L = SuperBee(theta, CFL)*W1[:,:,1:-1]
    norm = np.sum(W2[:,:,1:-1]**2,0)
    norm += (norm == 0 ) * 1.
    theta = np.sum(W2[:,:,0:-2]*W2[:,:,1:-1],0)/norm
    W2L = SuperBee(theta, CFL)*W2[:,:,1:-1]
    F[:,...] = ((-(1.-dt/dx*s[:,1:-2])*W1L) +
        ((1.-dt/dx*s[:,2:-1])*W2L))
    F = F * 0.5
    return F

def Fimm2Dy(s, W1, W2):
    dim = np.shape(W1)
    F = np.zeros((dim[0], dim[1] - 2,dim[2]))
    norm = np.sum(W1[:,1:-1,:]**2,0)
    norm += (norm == 0. ) * 1.
    theta = np.sum(W1[:,2:,:]*W1[:,1:-1,:],0)/norm
    W1L = SuperBee(theta, CFL)*W1[:,1:-1,:]
    norm = np.sum(W2[:,1:-1,:]**2,0)
    norm += (norm == 0 ) * 1.
    theta = np.sum(W2[:,0:-2,:]*W2[:,1:-1,:],0)/norm
    W2L = SuperBee(theta, CFL)*W2[:,1:-1,:]
    F[:,...] = ((-(1.-dt/dy*s[1:-2,:])*W1L) +
        ((1.-dt/dy*s[2:-1,:])*W2L))
    F = F * 0.5
    return F

def SuperBee(theta, cfl):
    shape = np.shape(theta)
    a = np.zeros(shape)
    b = np.ones(shape)
    c = np.min([b , 2.*theta],0)
    d = np.min([2.*b, theta],0)
    phy = np.max([a,c,d],0)
    return phy


def Ultrabee(theta, cfl):
    shape = np.shape(theta)
    ccfl = np.ones(shape) * cfl
    a = np.ones(shape) * 0.001
    b = np.ones(shape) * 0.999
    c = np.ones(shape)
    cfmod1 = np.max([a, ccfl], 0)
    cfmod2 = np.min([b, ccfl], 0)
    a = 2 * theta/cfmod1
    T1 = np.min([c, a], 0)
    b = 2/(1 - cfmod2)
    T2 = np.min([theta, b], 0)
    return np.max([np.zeros(shape), T1, T2], 0)

def MC(theta, cfl):
    shape = np.shape(theta)
    a = np.zeros(shape)
    b = np.ones(shape)
    c = np.min([(b + theta)/2. , 2.*theta, 2*b],0)
    phy = np.max([a,c],0)
    return phy

def MaxMod(a):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.max(a,0) + (k2.all(0))*np.min(a,0)

def MinMod(a):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.min(a,0) + (k2.all(0))*np.max(a,0)

def BCx(q, op, op1):
    if op1 == 0:
        if op == 1:
            q[:,:,0] = q[:,:,-4]
            q[:,:,1] = q[:,:,-3]
        if op == 2:
            q[:,:,0] = q[:,:,2]
            q[:,:,1] = q[:,:,2]
            return q
        if op == 3:
            q[:,:,1] = q[:,:,2]
            q[:,:,0] = q[:,:,3]
            q[1,:,0] = -q[1,:,3] 
            q[1,:,1] = -q[1,:,2]
            return q
    if op1 == 1:
        if op == 1:
            q[:,:,-1] = q[:,:,3]
            q[:,:,-2] = q[:,:,2]
        if op == 2:
            q[:,:,-1] = q[:,:,-3]
            q[:,:,-2] = q[:,:,-3]
            return q
        if op == 3:
            q[:,:,-2] = q[:,:,-3]
            q[:,:,-1] = q[:,:,-4]
            q[1,:,-1] = -q[1,:,-4] 
            q[1,:,-2] = -q[1,:,-3]
            return q

def BCy(q, op, op1):
    if op1 == 0:
        if op == 1:
            q[:,0,:] = q[:,-4,:]
            q[:,1,:] = q[:,-3,:]
        if op == 2:
            q[:,0,:] = q[:,2,:]
            q[:,1,:] = q[:,2,:]
            return q
        if op == 3:
            q[:,1,:] = q[:,2,:]
            q[:,0,:] = q[:,3,:]
            q[2,0,:] = -q[2,3,:] 
            q[2,1,:] = -q[2,2,:]
            return q
    if op1 == 1:
        if op == 1:
            q[:,-1,:] = q[:,3,:]
            q[:,-2,:] = q[:,2,:]
        if op == 2:
            q[:,-1,:] = q[:,-3,:]
            q[:,-2,:] = q[:,-3,:]
            return q
        if op == 3:
            q[:,-1,:] = q[:,-4,:]
            q[:,-2,:] = q[:,-3,:]
            q[2,-1,:] = -q[2,-4,:] 
            q[2,-2,:] = -q[2,-3,:]
            return q

ref_factor = 0
c1 = np.load("gdm_rf" + str(ref_factor) + ".npy")
Lim_Infx = 20000; Lim_Supx = 50000; NumVolx = c1.shape[1]
Lim_Infy = 0; Lim_Supy = 8000; NumVoly = c1.shape[0]
t_inicial = 0.; t_final = 5000
CFL = 0.5
Xc, Yc, Xn, Yn, dx, dy = Domain(Lim_Infx, Lim_Supx, Lim_Infy, Lim_Supy, NumVolx, NumVoly)
c = np.ones((NumVoly + 4, NumVolx + 4))
c[2:-2, 2:-2] = c1[:, :]
c[0,:] = c[2,:]; c[-1,:] = c[-3,:]
c[1,:] = c[2,:]; c[-2,:] = c[-3,:]
c[:,0] = c[:,2]; c[:,-1] = c[:,-3]
c[:,1] = c[:,2]; c[:,-2] = c[:,-3]
        
b = 1 / (0.31 * (1e3*c)**0.25)
b[c < 1.51] = 1.0
Rho = 1/b
K = c**2 * Rho

VelMax = np.max(c)
dt = (np.minimum(dx, dy) * CFL)/VelMax 

q = np.zeros((3,NumVoly + 4,NumVolx + 4))
q = BCx(q, 3, 0)
q = BCx(q, 3, 1)
q = BCy(q, 3, 1)
q = BCy(q, 3, 0)
qb = np.copy(q) 
Nt = int(round(t_final/dt))
receiver = np.zeros((Nt, NumVolx))
dim = np.shape(q)
dimx = (dim[0], dim[1], dim[2]-1)
dimy = (dim[0], dim[1]-1, dim[2])
isx = NumVolx//2
isy = np.where(np.linspace(Lim_Infy, Lim_Supy, NumVoly) == 32)[0][0]
time_values = np.arange(Nt)*dt
f0  = 0.005
t0 = 1./f0
a = 1.
r = (np.pi * f0 * (time_values - t0))
src = a * (1-2.*r**2)*np.exp(-r**2)

F = np.zeros((3,NumVoly+4,NumVolx+3))
G = np.zeros((3,NumVoly+3,NumVolx+4))

start = time.time()
for i in range(Nt):
    W1, W3 = Riemann_Elasticity_2Dx(q, Rho, K, c)
    qb[:,:,2:-2] = qb[:,:,2:-2] - dt/dx * (W3[:,:,1:-2] + W1[:,:,2:-1])
    F[:,:,1:-1] = Fimm2Dx(c, W1, W3)
    qb[:,:,2:-2] = qb[:,:,2:-2] - dt/dx * (F[:,:,2:-1] - F[:,:,1:-2])
    qb = BCx(qb, 3, 0)
    qb = BCx(qb, 3, 1)
    W1, W3 = Riemann_Elasticity_2Dy(qb, Rho, K, c)
    qb[:,2:-2,:] = qb[:,2:-2,:] - dt/dy * (W3[:,1:-2,:] + W1[:,2:-1,:])
    G[:,1:-1,:] = Fimm2Dy(c, W1, W3)
    qb[:,2:-2,:] = qb[:,2:-2,:] - dt/dy * (G[:,2:-1,:] - G[:,1:-2,:])
    qb[0, isy + 2, isx + 2] = qb[0, isy + 2, isx + 2] + dt * src[i] / (dx * dy)
    receiver[i, :] = (qb[0]*K)[isy + 2,2:-2]
    qb = BCy(qb, 3, 1)
    qb = BCy(qb, 3, 0)
    q = np.copy(qb)

end = time.time()
comp_time = end - start
Stress = (q[0]*K)[2:-2,2:-2]
domain = [Lim_Infx, Lim_Supx, Lim_Infy, Lim_Supy]
gf.plot_image(Stress, extent = [Lim_Infx, Lim_Supx, Lim_Supy, Lim_Infy])
is1 = np.where(np.linspace(Lim_Infx, Lim_Supx, NumVolx) == 33350)[0][0]
gf.plot_seismic_traces([receiver[:, is1]], 0, t_final)
gf.plot_shotrecord(receiver[:,:], (domain[0],domain[2]), (domain[1], domain[3]), 0, t_final, factor=100)
plt.show()

