#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:25:47 2020

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ConvectionDiffusionMlsSim import ConvectionDiffusionMlsSim

def gaussian(points):
    A = 1.0
    x0 = 0.5
    y0 = 0.5
    xsigma = 0.15
    ysigma = 0.15
    return np.exp(-0.5*A*( (points[:,0] - x0)**2/xsigma**2 + 
                           (points[:,1] - y0)**2/ysigma**2 ) )

def hat(points):
    return np.hstack((points > 0.25, points < 0.75)).all(1).astype('float64')

n = 3
dt = 0.1
velocity = np.array([0.1, 0.1], dtype='float64')
diffusivity = 0.0

kwargs={
    'N' : n,
    'dt' : dt,
    'u0' : gaussian,
    'velocity' : velocity,
    'diffusivity' : diffusivity,
    'Nquad' : 2,
    'support' : 3,
    'form' : 'cubic',
    'quadrature' : 'gaussian' }

precon='ilu'
tolerance = 1e-10
    
# Initialize simulation
mls = ConvectionDiffusionMlsSim(**kwargs)
# mls.computeSpatialDiscretization()

N = 16
points = ( np.indices((N+1, N+1), dtype='float64').T.reshape(-1,2) ) / N
    
phi_tmp = np.apply_along_axis(mls.phi, 1, points, mls.nodes)

phis = np.empty((len(points), mls.nNodes), dtype='float64')
for i in range(mls.nNodes):
    phis[:,i] = np.sum(phi_tmp[:,mls.periodicIndices == i], axis=1)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,13)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.2)

# subplots = [337, 338, 339, 334, 335, 336, 331, 332, 333]
# subplots = [447, 448, 449, 444, 445, 446, 441, 442, 443]

for j in range(n):
    for i in range(n):
        # plot the result
        plt.subplot(n,n,n*n-(j+1)*n+i+1)
        plt.tripcolor(points[:,0], points[:,1], phis[:,i*n+j], shading='gouraud'
                      # , vmax=1.0
                      , vmin=0.0)
        plt.colorbar()
        # ax = plt.subplot(subplots[i], projection='3d')
        # surf = ax.plot_trisurf(points[:,0], points[:,1], phis[:,i],
        #                     cmap='viridis', linewidth=0, antialiased=False,
        #                     vmin=0.0, vmax=1.0)
        # plt.colorbar(surf, shrink=0.75, aspect=7)
        if i == 0:
            plt.xlabel(r'$x$')
        if j == 0:
            plt.ylabel(r'$y$')
        plt.title('$\Phi_{{{0}}}$'.format(i*n+j))
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        # plt.xticks([0.0, 0.5, 1.0])
        # plt.yticks([0.0, 0.5, 1.0])
        # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.margins(0,0)

# plt.savefig('MLS_shape_functions5.pdf', bbox_inches='tight', pad_inches=0)