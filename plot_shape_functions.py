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

n = 100
ndim = 1

N = 18
dt = 0.1
velocity = np.array([0.1, 0.1, 0.1], dtype='float64')[0:ndim]
diffusivity = 0.0*np.eye(ndim)

kwargs={
    'N' : N,
    'dt' : dt,
    'u0' : gaussian,
    'velocity' : velocity,
    'diffusivity' : diffusivity,
    'ndim' : ndim,
    'Nquad' : 4,
    'support' : 3,
    'form' : 'quintic',
    'quadrature' : 'uniform',
    'basis' : 'linear'}

precon='ilu'
tolerance = 1e-10
    
# Initialize simulation
mls = ConvectionDiffusionMlsSim(**kwargs)
mls.computeSpatialDiscretization()

points = ( np.indices(np.repeat(n+1, ndim), dtype='float64')
           .T.reshape(-1,ndim) ) / n
phis = np.zeros((len(points), mls.nNodes), dtype='float64')
for i, point in enumerate(points):
    indices, local_phis = mls.phi(point)
    for j, phi in enumerate(local_phis):
        phis[i,mls.periodicIndices[indices[j]]] += phi
    
# phi_tmp = np.apply_along_axis(lambda p: mls.phi(p)[1], 1, points)

# phis = np.empty((len(points), mls.nNodes), dtype='float64')
# for i in range(mls.nNodes):
#     phis[:,i] = np.sum(phi_tmp[:,mls.periodicIndices == i], axis=1)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')

if ndim == 1:

    dphis = np.zeros((len(points), mls.nNodes), dtype='float64')
    d2phis = np.zeros((len(points), mls.nNodes), dtype='float64')
    for i, point in enumerate(points):
        indices, _, dphi = mls.dphi(point)
        d2phi = mls.d2phi(point)[2]
        dphis[i,mls.periodicIndices[indices]] = dphi.reshape(-1,)
        d2phis[i,mls.periodicIndices[indices]] = d2phi.reshape(-1,)

    fig.set_size_inches(15,4.5)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    # phisToPlot = [int(N/2)]
    # phisToPlot = range(N)
    
    ##### Use to plot speicfic functions with derivates #####
    phisToPlot = [0]
    factor = np.sin(2*np.pi*np.arange(0., phis.shape[1])/phis.shape[1])
    phis = np.sum(factor*phis, axis=1).reshape(-1,1)
    dphis = np.sum(factor*dphis, axis=1).reshape(-1,1)
    d2phis = np.sum(factor*d2phis, axis=1).reshape(-1,1)
    
    plt.subplot(1,3,1)
    for i in phisToPlot:
        plt.plot(points, phis[:,i],label=f'$\Phi_{i}$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\Phi$', rotation=0)
        plt.legend()
    
    plt.subplot(1,3,2)
    for i in phisToPlot:
        plt.plot(points, dphis[:,i],label=f'$\Phi_{i}$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\Phi_x$', rotation=0)
        plt.legend()
    
    plt.subplot(1,3,3)
    for i in phisToPlot:
        plt.plot(points, d2phis[:,i],label=f'$\Phi_{i}$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\Phi_{xx}$', rotation=0)
        plt.legend()

if ndim == 2:
    fig.set_size_inches(15,13)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.2)
    
    for j in range(N):
        for i in range(N):
            # plot the result
            plt.subplot(N,N,N*N-(j+1)*N+i+1)
            plt.tripcolor(points[:,0], points[:,1], phis[:,i*N+j], shading='gouraud'
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
            plt.title('$\Phi_{{{0}}}$'.format(i*N+j))
            plt.xticks([0, 1])
            plt.yticks([0, 1])
            # plt.xticks([0.0, 0.5, 1.0])
            # plt.yticks([0.0, 0.5, 1.0])
            # plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.margins(0,0)
##### End ndim == 2 #####

# plt.savefig('MLS_shape_functions5.pdf', bbox_inches='tight', pad_inches=0)