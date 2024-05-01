# -*- coding: utf-8 -*-
"""
@author: Juan Barrios
"""

import time
import numpy             as np
import matplotlib.pyplot as plt
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

def Flux(u,i,axe):
    f = np.copy(u)
    if axe == 1:
        f[0,:,:] = -u[1,:,:]/Rho[:,i[0]:i[1]]
        f[1,:,:] = -u[0,:,:]*K[:,i[0]:i[1]]
        f[2,:,:] = 0.
    elif axe == 2:
        f[0,:,:] = -u[2,:,:]/Rho[i[0]:i[1],:]
        f[1,:,:] = 0.
        f[2,:,:] = -u[0,:,:]*K[i[0]:i[1],:]
    return f
    
def Limiter(cp,cm,cpm,lim,axe, ddx):
    sig1 = MinMod(np.array([2.*cm/ddx,cp/ddx]), axe)
    sig2 = MinMod(np.array([cm/ddx,2.*cp/ddx]), axe)
    return MaxMod(np.array([sig1,sig2]), axe)

def MaxMod(a, axe):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.max(a,0) + (k2.all(0))*np.min(a,0)

    
def MinMod(a, axe):
    k1 = a > 0.
    k2 = a < 0.
    return (k1.all(0))*np.min(a,0) + (k2.all(0))*np.max(a,0)

def CUp(q, c, lim):
    cpx = q[:,:,2:] - q[:,:,1:-1] 
    cmx = q[:,:,1:-1] - q[:,:,:-2]
    cpmx = q[:,:,2:] - q[:,:,:-2]
    qW = q[:,:,1:-1] - dx/2.*  Limiter(cpx, cmx, cpmx, lim, 1, dx)
    qE = q[:,:,1:-1] + dx/2. *  Limiter(cpx, cmx, cpmx, lim, 1, dx)  

    cpy = q[:,2:,:] - q[:,1:-1,:] 
    cmy = q[:,1:-1,:] - q[:,:-2,:]
    cpmy = q[:,2:,:] - q[:,:-2,:]
    qS = q[:,2:-1,:] - dy/2.*  Limiter(cpy[:,1:,:],cmy[:,1:,:],cpmy[:,1:,:],lim, 2, dy)
    qN = q[:,1:-2,:] + dy/2.*  Limiter(cpy[:,:-1,:],cmy[:,:-1,:],cpmy[:,:-1,:],lim, 2, dy)
    
    
    qNE = qE[:,1:-1,:] + dy/2.*  Limiter(cpy[:,:,1:-1],cmy[:,:,1:-1],cpmy[:,:,1:-1],lim, 2, dy)
    qNW = qW[:,1:-1,1:] + dy/2.*  Limiter(cpy[:,:,2:-1],cmy[:,:,2:-1],cpmy[:,:,2:-1],lim, 2, dy)
    qSE = qE[:,2:-1,:] - dy/2.*  Limiter(cpy[:,1:,1:-1],cmy[:,1:,1:-1],cpmy[:,1:,1:-1],lim, 2, dy)
    qSW = qW[:,2:-1,1:] - dy/2.*  Limiter(cpy[:,1:,2:-1],cmy[:,1:,2:-1],cpmy[:,1:,2:-1],lim, 2, dy)

    ap = np.maximum(c[:,1:-2], np.maximum(c[:,2:-1], 0.))
    am = np.minimum(-c[:,1:-2], np.minimum(-c[:,2:-1], 0.))
    
    bp = np.maximum(c[1:-2,:], np.maximum(c[2:-1,:], 0.))
    bm = np.minimum(-c[1:-2,:], np.minimum(-c[2:-1,:], 0.))
    
    apmam = ap - am
    bpmbm = bp - bm
    
    Wintx = (ap*qW[:,:,1:] - am*qE[:,:,:-1] - (Flux(qW[:,:,1:],[2,-1], 1) - Flux(qE[:,:,:-1],[1,-2], 1)))
    Wintx /= apmam
    T1 = (qNW[:,2:,:] - Wintx[:,2:-2,:])/apmam[2:-2,:]
    T2 = (Wintx[:,2:-2,:] - qNE[:,2:,:-1])/apmam[2:-2,:]
    T3 = (qSW[:,1:,:] - Wintx[:,2:-2,:])/apmam[2:-2,:]
    T4 = (Wintx[:,2:-2,:] - qSE[:,1:,:-1])/apmam[2:-2,:]
    qchiux = MinMod(np.array([T1,T2,T3,T4]), 1)
    # qchiux = 0
    
    Winty = (bp*qS - bm*qN - (Flux(qS,[2,-1], 2) - Flux(qN,[1,-2], 2)))
    Winty /= bpmbm
    T1 = (qSW[:,:,1:] - Winty[:,:,2:-2])/bpmbm[:,2:-2]
    T2 = (Winty[:,:,2:-2] - qNW[:,:-1,1:])/bpmbm[:,2:-2]
    T3 = (qSE[:,:,1:-1] - Winty[:,:,2:-2])/bpmbm[:,2:-2]
    T4 = (Winty[:,:,2:-2] - qNE[:,:-1,2:])/bpmbm[:,2:-2]
    qchiuy = MinMod(np.array([T1,T2,T3,T4]), 2)
    # qchiuy = 0

    apxam = ap * am
    bpxbm = bp * bm
    Hx = (ap*Flux(qE[:,:,:-1],[1,-2], 1) - am*Flux(qW[:,:,1:],[2,-1], 1))[:,2:-2,:]
    Tcx = ((qW[:,:,1:] - qE[:,:,:-1])/apmam)[:,2:-2,:] - qchiux
    Hx = Hx/apmam[2:-2,:] + (apxam[2:-2,:] * Tcx)
    Hx = - (Hx[:,:,1:] - Hx[:,:,:-1])/dx
    
    Hy = (bp*Flux(qN,[1,-2], 2) - bm*Flux(qS,[2,-1], 2))[:,:,2:-2]
    Tcy = ((qS - qN)/bpmbm)[:,:,2:-2] - qchiuy
    Hy = Hy/bpmbm[:,2:-2] + (bpxbm[:,2:-2] * Tcy)
    Hy = - (Hy[:,1:,:] - Hy[:,:-1,:])/dy
    
    H = Hx + Hy
    return H

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
            q[1,:,0] = -q[1,:,0] 
            q[1,:,1] = -q[1,:,1]
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
            q[1,:,-1] = -q[1,:,-1]
            q[1,:,-2] = -q[1,:,-2]
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
            q[2,0,:] = -q[2,0,:] 
            q[2,1,:] = -q[2,1,:]
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
            q[:,-1,:] = q[:,-3,:]   
            q[:,-2,:] = q[:,-4,:]
            q[2,-1,:] = -q[2,-1,:] 
            q[2,-2,:] = -q[2,-2,:]
            return q
        
        

ref_factor = 0
c1 = np.load("gdm_rf" + str(ref_factor) + ".npy")
Lim_Infx = 20000; Lim_Supx = 50000; NumVolx = c1.shape[1]
Lim_Infy = 0; Lim_Supy = 8000; NumVoly = c1.shape[0]
t_inicial = 0.; t_final = 5000
CFL = 0.25 
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

start = time.time()
for i in range(Nt):
    #RK2 ---------------------------------------------------------------------
    qb1 = q[:,2:-2,2:-2] + dt*(CUp(q, c, 0))
    qb[:,2:-2,2:-2] = qb1
    qb1 = 0.5 * q[:,2:-2,2:-2] + 0.5*(qb1 + dt*(CUp(qb, c, 0)))
    qb[:,2:-2,2:-2] = qb1
    qb[0, isy + 2, isx + 2] = qb[0, isy + 2, isx + 2] + dt * src[i] / (dx * dy)
    receiver[i, :] = (q[0]*K)[isy + 2,2:-2]
    qb = BCx(qb, 3, 0)
    qb = BCx(qb, 3, 1)
    qb = BCy(qb, 3, 1)
    qb = BCy(qb, 3, 0)
    q = np.copy(qb)
    #-------------------------------------------------------------------------
    
end = time.time()
comp_time = end - start
Stress = (q[0]*K)[2:-2,2:-2]
domain = [Lim_Infx, Lim_Supx, Lim_Infy, Lim_Supy]
gf.plot_image(Stress, extent = [Lim_Infx, Lim_Supx, Lim_Supy, Lim_Infy])
is1 = np.where(np.linspace(Lim_Infx, Lim_Supx, NumVolx) == 33350)[0][0]
gf.plot_seismic_traces([receiver[:, is1]], 0, t_final)
gf.plot_shotrecord(receiver[:,:], (domain[0],domain[2]), (domain[1], domain[3]), 0, t_final, factor=100)
plt.show()
