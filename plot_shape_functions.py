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
    ndim = points.shape[1]
    r0 = (0.5, 0.5, 0.5)[0:ndim]
    sigma = (0.1, 0.1, 0.1)[0:ndim]
    return np.exp( -0.5*A*np.sum(((points - r0)/sigma )**2, 1) )

def hat(points):
    return np.hstack((points > 0.25, points < 0.75)).all(1).astype('float64')

N = 100
ndim = 1

n = 10
dt = 0.1
velocity = np.array([0.1, 0.1], dtype='float64')
diffusivity = 0.0*np.eye(ndim)

kwargs={
    'N' : n,
    'dt' : dt,
    'u0' : gaussian,
    'velocity' : velocity,
    'diffusivity' : diffusivity,
    'ndim' : ndim,
    'Nquad' : 4,
    'support' : 2.5,
    'form' : 'quartic',
    'quadrature' : 'uniform',
    'basis' : 'quadratic'}

precon='ilu'
tolerance = 1e-10
    
# Initialize simulation
mls = ConvectionDiffusionMlsSim(**kwargs)
# mls.computeSpatialDiscretization()

points = ( np.indices(np.repeat(N+1, ndim), dtype='float64')
           .T.reshape(-1,ndim) ) / N
    
phi_tmp = np.apply_along_axis(mls.phi, 1, points, mls.nodes)

phis = np.empty((len(points), mls.nNodes), dtype='float64')
for i in range(mls.nNodes):
    phis[:,i] = np.sum(phi_tmp[:,mls.periodicIndices == i], axis=1)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')

if ndim == 1:

    dphi_tmp = np.apply_along_axis(lambda p, n : mls.dphi(p,n)[1], 1, points,
                                   mls.nodes).reshape(-1,len(mls.nodes))
    d2phi_tmp = np.apply_along_axis(lambda p, n : mls.d2phi(p,n)[1], 1, points,
                                   mls.nodes).reshape(-1,len(mls.nodes))
    dphis = np.empty((len(points), mls.nNodes), dtype='float64')
    d2phis = np.empty((len(points), mls.nNodes), dtype='float64')
    for i in range(mls.nNodes):
        dphis[:,i] = np.sum(dphi_tmp[:,mls.periodicIndices == i], axis=1)
        d2phis[:,i] = np.sum(d2phi_tmp[:,mls.periodicIndices == i], axis=1)

    fig.set_size_inches(15,4.5)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    phisToPlot = [int(n/2)]
    # phisToPlot = range(n)
    
    plt.subplot(1,3,1)
    for i in phisToPlot:
        plt.plot(points, phis[:,i],label=f'$\Phi_{i}$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\Phi(x)$')
        plt.legend()
    
    plt.subplot(1,3,2)
    for i in phisToPlot:
        plt.plot(points, dphis[:,i],label=f'$\Phi_{i}$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\frac{d\Phi(x)}{dx}$',rotation=0)
        plt.legend()
    
    plt.subplot(1,3,3)
    for i in phisToPlot:
        plt.plot(points, d2phis[:,i],label=f'$\Phi_{i}$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\frac{d^2\Phi(x)}{dx^2}$',rotation=0)
        plt.legend()

if ndim == 2:
    fig.set_size_inches(15,13)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.2)
    
    for j in range(n):
        for i in range(n):
            # plot the result
            plt.subplot(n,n,n*n-(j+1)*n+i+1)
            plt.tripcolor(points[:,0], points[:,1], phis[:,i*n+j], shading='gouraud'
                          # , vmax=1.0
                          , vmin=0.0)
            plt.colorbar()
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
##### End ndim == 2 #####

# plt.savefig('MLS_shape_functions5.pdf', bbox_inches='tight', pad_inches=0)