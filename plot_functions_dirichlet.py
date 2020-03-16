#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:25:47 2020

@author: samal
"""

import numpy as np
import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from PoissonMlsSim import PoissonMlsSim

def g(points):
    k = 1
    return np.sin(k*np.pi*points[:,0]) * np.sinh(k*np.pi*points[:,1])

def one(points):
    return np.ones(len(points), dtype='float64')

def x(points):
    return points[:,0]

def y(points):
    return points[:,1]

def xpy(points):
    return points[:,0] + points[:,1]

def x2(points):
    return points[:,0]**2

def y2(points):
    return points[:,1]**2

def xy(points):
    return points[:,0] * points[:,1]

def x2y2(points):
    return points[:,0]**2 * points[:,1]**2

def sinx(points):
    return np.sin(np.pi*points[:,0])

def sin2x(points):
    return np.sin(2*np.pi*points[:,0])

def siny(points):
    return np.sin(np.pi*points[:,1])

def sin2y(points):
    return np.sin(2*np.pi*points[:,1])

def sinxy(points):
    return np.sin(np.pi*(points[:,0]*points[:,1]))

def sinxpy(points):
    return np.sin(np.pi*(points[:,0]+points[:,1]))

def sinxsiny(points):
    return np.sin(np.pi*points[:,0])*np.sin(np.pi*points[:,1])

func = xpy

kwargs={
    'Nquad' : 2,
    'support' : 3,
    'form' : 'cubic',
    'method' : 'collocation',
    'quadrature' : 'gaussian',
    'basis' : 'quadratic'}

mls = PoissonMlsSim(8, g, **kwargs)

N = 64
points = ( np.indices((N+1, N+1), dtype='float64').T.reshape(-1,2) ) / N
indices = np.arange(mls.nNodes, dtype = 'uint32')

phis = np.apply_along_axis(mls.phi, 1, points, mls.nodes)

A = np.apply_along_axis(mls.phi, 1, mls.nodes, mls.nodes)
b = func(mls.nodes)
u = la.solve(A,b)

approximate_function = phis@u
exact_function = func(points)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,6)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.1)

# plot the result
# plt.subplot(221)
# plt.tripcolor(points[:,0], points[:,1], approximate_function, shading='gouraud')
# plt.colorbar()
ax = plt.subplot(121, projection='3d')
surf = ax.plot_trisurf(points[:,0], points[:,1], approximate_function,
                       cmap='viridis', linewidth=0, antialiased=False
                        # , vmin=0, vmax=2
                       )
# ax.zaxis.set_ticks([0,0.5,1,1.5,2])
plt.colorbar(surf, shrink=0.75, aspect=7)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('MLS Approximation')

# # plot the result
# # plt.subplot(222)
# # plt.tripcolor(points[:,0], points[:,1], exact_function, shading='gouraud')
# # plt.colorbar()
# ax = plt.subplot(222, projection='3d')
# surf = ax.plot_trisurf(points[:,0], points[:,1], exact_function,
#                        cmap='viridis', linewidth=0, antialiased=False)
# plt.colorbar(surf, shrink=0.75, aspect=7)
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$')
# plt.title('Exact Function')

# plot the error
difference = approximate_function - exact_function
# plt.subplot(223)
# plt.tripcolor(points[:,0], points[:,1], difference, shading='gouraud')
# plt.colorbar()
ax = plt.subplot(122, projection='3d')
surf = ax.plot_trisurf(points[:,0], points[:,1], difference,
                       cmap='seismic', linewidth=0, antialiased=False,
                       vmin=-np.max(np.abs(difference)),
                       vmax=np.max(np.abs(difference)))
# ax.axes.set_zlim3d(bottom=-np.max(np.abs(difference)),
#                    top=np.max(np.abs(difference))) 
plt.colorbar(surf, shrink=0.75, aspect=7)
# plt.colorbar(surf, vmin=-np.max(np.abs(difference)),
             # vmax=np.max(np.abs(difference)))
# ax.zaxis.set_ticks([0, 0.001, 0.002, 0.003])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Error')

# plt.savefig('MLS_xy.pdf', bbox_inches='tight', pad_inches=0)