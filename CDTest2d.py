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
    r0 = np.repeat(0.5, ndim)
    sigma = np.repeat(0.1, ndim)
    return A*np.exp( -0.5*np.sum(((points - r0)/sigma )**2, 1) )

def hat(points):
    return np.hstack((points > 0.25, points < 0.75)).all(1).astype('float64')

def sinusoid(points):
    return np.sin(2.0*np.pi*np.sum(points, axis=1))

# N is the number of grid cells along one dimension,
# therefore the number of nodes equals N*N
N = 30
dt = 0.01
velocity = np.array([0.1, 0.2], dtype='float64')
# theta = 3.0*np.pi/4.0
# diffusivity = 0.01*np.array([[np.cos(theta)**2, np.sin(theta)*np.cos(theta)],
#                                [np.sin(theta)*np.cos(theta), np.sin(theta)**2]])
# diffusivity += 0.*np.eye(2)
diffusivity = 0.
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
    'support' : ('circular', 1.8),
    'form' : 'gaussian',
    'quadrature' : 'uniform',
    'basis' : 'linear'}

precon='ilu'
tolerance = 1e-10

start_time = default_timer()
    
# Initialize simulation
mlsSim = ConvectionDiffusionMlsSim(**kwargs)
mlsSim.computeSpatialDiscretization()
mlsSim.precondition(precon)

current_time = default_timer()
print(f'Set-up time = {current_time-start_time} s')
print('Condition Number =', mlsSim.cond('fro'))

# # Store dense versions of the internal arrays for debugging
# M = mlsSim.M.A
# K = mlsSim.K.A
# A = mlsSim.A.A
# KA = mlsSim.KA.A

start_time = default_timer()

mlsSim.step(1000, tol=tolerance, atol=tolerance)

current_time = default_timer()
print(f'Simulation time = {current_time-start_time} s')
    
# Compute true approximation from nodal coefficients
mlsSim.solve()
     
# compute the analytic solution and error norms
u_exact = kwargs['u0'](mlsSim.uNodes())
E_inf = la.norm(mlsSim.u - u_exact, np.inf)
E_2 = la.norm(mlsSim.u - u_exact)/N
print('max error =', E_inf)
print('L2 error  =', E_2)


##### Begin Plotting Routines #####

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(7.75,3)
plt.subplots_adjust(hspace = 0.3, wspace = 0.2)

SMALL_SIZE = 7
MEDIUM_SIZE = 8
BIGGER_SIZE = 10
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plot the result
plt.subplot(121)
plt.tripcolor(mlsSim.nodes[:,0], mlsSim.nodes[:,1],
              mlsSim.u[mlsSim.periodicIndices], shading='gouraud')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
# plt.title('Final MLS solution')
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
plt.ylabel(r'$y$', rotation=0)
# plt.title('Error')
plt.margins(0,0)

## Plot for progress report
# fig = plt.gcf()
# fig.clf()
# fig.set_size_inches(7.75,3)
# plt.subplots_adjust(hspace = 0.3, wspace = 0.2)

# SMALL_SIZE = 7
# MEDIUM_SIZE = 8
# BIGGER_SIZE = 10
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.subplot(121)
# plt.tripcolor(mlsSim.nodes[:,0], mlsSim.nodes[:,1],
#               difference_gaussian[mlsSim.periodicIndices],
#               shading='gouraud',
#               cmap='seismic',
#               vmin=-np.max(np.abs(difference_gaussian)),
#               vmax=np.max(np.abs(difference_gaussian)))
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.colorbar()
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$', rotation=0)
# plt.margins(0,0)

# # # plot error
# plt.subplot(122)
# plt.tripcolor(mlsSim.nodes[:,0], mlsSim.nodes[:,1],
#               difference_hat[mlsSim.periodicIndices],
#               shading='gouraud',
#               cmap='seismic',
#               vmin=-np.max(np.abs(difference_hat)),
#               vmax=np.max(np.abs(difference_hat)))
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.colorbar()
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$', rotation=0)
# plt.margins(0,0)

# plt.savefig(f"MLS_convection_only.pdf",
#     bbox_inches = 'tight', pad_inches = 0)