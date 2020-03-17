#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Meshfree method simulation using moving least squares (MLS)

@author: Sam Maloney
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as la

from ConvectionDiffusionMlsSim import ConvectionDiffusionMlsSim
from timeit import default_timer

import warnings
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

def gaussian(points):
    A = 1.0
    ndim = points.shape[1]
    r0 = (0.5, 0.5, 0.5)[0:ndim]
    sigma = (0.15, 0.15, 0.15)[0:ndim]
    return np.exp( -0.5*A*np.sum(((points - r0)/sigma )**2, 1) )

def hat(points):
    return np.hstack((points > 0.25, points < 0.75)).all(1).astype('float64')

# N is the number of grid cells along one dimension,
# therefore the number of nodes equals N*N
N = 30
dt = 0.01
velocity = np.array([0.1, 0.], dtype='float64')
theta = np.pi/4
diffusivity = 0.01*np.array([[np.cos(theta)**2, np.sin(theta)*np.cos(theta)],
                             [np.sin(theta)*np.cos(theta), np.sin(theta)**2]])
print(f'N = {N}\ndt = {dt}\n'
      f'velocity = {velocity}\n'
      f'diffusivity =\n{diffusivity}')

kwargs={
    'N' : N,
    'dt' : dt,
    'u0' : gaussian,
    'velocity' : velocity,
    'diffusivity' : diffusivity,
    'Nquad' : 2,
    'support' : 1.9,
    'form' : 'cubic',
    'quadrature' : 'gaussian',
    'basis' : 'linear'}

precon='ilu'
tolerance = 1e-10

# # allocate arrays for convergence testing
# start = 1
# stop = 5
# nSamples = stop - start + 1
# N_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
# E_inf = np.empty(nSamples, dtype='float64')
# E_2 = np.empty(nSamples, dtype='float64')

start_time = default_timer()
    
# Initialize simulation
mlsSim = ConvectionDiffusionMlsSim(**kwargs)
mlsSim.computeSpatialDiscretization()

current_time = default_timer()
print(f'Set-up time = {current_time-start_time} s')
print('Condition Number =', mlsSim.cond('fro'))

# M = mlsSim.M.A
# K = mlsSim.K.A
# A = mlsSim.A.A
# KA = mlsSim.KA.A

start_time = default_timer()

mlsSim.step(200, tol=tolerance, atol=tolerance)

current_time = default_timer()
print(f'Simulation time = {current_time-start_time} s')
    
mlsSim.solve()

# # loop over timesteps 
# nSteps = 100
# for iStep in range(nSteps):
    
        
# compute the analytic solution and error norms
u_exact = kwargs['u0'](mlsSim.uNodes())
E_inf = la.norm(mlsSim.u - u_exact, np.inf)
E_2 = la.norm(mlsSim.u - u_exact)/N
print('max error =', E_inf)
print('L2 error  =', E_2)
    
#     current_time = default_timer()
    
#     print(f'Simulation time = {current_time-start_time} s')
    
# ##### End of loop over timesteps #####

    
    
##### Begin Plotting Routines #####

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,7)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
# plt.subplots_adjust(hspace = 0.3, wspace = 0.25)

# plot the result
plt.subplot(121)
plt.tripcolor(mlsSim.nodes[:,0], mlsSim.nodes[:,1],
              mlsSim.u[mlsSim.periodicIndices], shading='gouraud')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Final MLS solution')
plt.margins(0,0)

# # plot error
difference = mlsSim.u - u_exact
plt.subplot(122)
plt.tripcolor(mlsSim.nodes[:,0], mlsSim.nodes[:,1],
              difference[mlsSim.periodicIndices],
              shading='gouraud',
              cmap='seismic',
              vmin=-np.max(np.abs(difference)),
              vmax=np.max(np.abs(difference)))
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Error')
plt.margins(0,0)

# # plot the error convergence
# plt.subplot(223)
# plt.loglog(N_array, E_inf, '.-', label=r'$E_\infty$')
# plt.loglog(N_array, E_2, '.-', label=r'$E_2$')
# plt.minorticks_off()
# plt.xticks(N_array, N_array)
# plt.xlabel(r'$N$')
# plt.ylabel(r'Magnitude of Error Norm')
# plt.title('MLS Error Norms')
# plt.legend(fontsize='x-large')

# # plot the intra-step order of convergence
# plt.subplot(224)
# logN = np.log(N_array)
# logE_inf = np.log(E_inf)
# logE_2 = np.log(E_2)
# order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
# order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
# intraN = np.logspace(start+0.5, stop-0.5, num=nSamples-1, base=2.0)
# plt.plot(intraN, order_inf, '.-', label=r'$E_\infty$')
# plt.plot(intraN, order_2, '.-', label=r'$E_2$')
# plt.plot([N_array[0], N_array[-1]], [2, 2], 'k:', label='Expected')
# plt.xlim(N_array[0], N_array[-1])
# plt.xscale('log')
# plt.minorticks_off()
# plt.xticks(N_array, N_array)
# # plt.ylim(1, 3)
# # plt.yticks([1, 1.5, 2, 2.5, 3])
# plt.ylim(0, 3)
# plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
# plt.xlabel(r'$N$')
# plt.ylabel(r'Intra-step Order of Convergence')
# plt.title('MLS Order of Accuracy')
# plt.legend(fontsize='x-large')
# plt.margins(0,0)

# # plt.savefig(f"MLS_{method}_{form}_{k}k_{Nquad}Q_{mlsSim.support*mlsSim.N}S.pdf",
# #     bbox_inches = 'tight', pad_inches = 0)