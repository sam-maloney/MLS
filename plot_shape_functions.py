#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:25:47 2020

@author: samal
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from MlsSim import MlsSim

def g(points):
    k = 1
    return np.sin(k*np.pi*points[:,0]) * np.sinh(k*np.pi*points[:,1])

n=5
mls = MlsSim(n-1, g=g, support=2, form='cubic')

N = 64
points = ( np.indices((N+1, N+1), dtype='float64').T.reshape(-1,2) ) / N
indices = np.arange(mls.nNodes, dtype = 'uint32')

phis = np.empty((len(points), mls.nNodes), dtype='float64')

for i, point in enumerate(points):
    phis[i] = mls.shapeFunctions0(point, indices)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,13)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.2)

# subplots = [337, 338, 339, 334, 335, 336, 331, 332, 333]
subplots = [447, 448, 449, 444, 445, 446, 441, 442, 443]

for i in range(n):
    for j in range(n):
        # plot the result
        plt.subplot(n,n,n*n-(i+1)*n+j+1)
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

plt.savefig('MLS_shape_functions5.pdf', bbox_inches='tight', pad_inches=0)