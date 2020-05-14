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

def one(points, derivative=0):
    if derivative == 0:
        return np.ones(len(points))
    elif derivative in [1, 2]:
        return np.zeros((len(points), 2))

def x(points, derivative=0):
    if derivative == 0:
        return points[:,0]
    elif derivative == 1:
        return np.hstack((np.ones(len(points)), np.zeros(len(points))))
    elif derivative == 2:
        return np.zeros((len(points), 2))

def y(points, derivative=0):
    if derivative == 0:
        return points[:,1]
    elif derivative == 1:
        return np.hstack((np.zeros(len(points)), np.ones(len(points))))
    elif derivative == 2:
        return np.zeros((len(points), 2))

def xpy(points, derivative=0):
    if derivative == 0:
        return points[:,0] + points[:,1]
    elif derivative == 1:
        return np.ones((len(points), 2))
    elif derivative == 2:
        return np.zeros((len(points), 2))

def x2(points, derivative=0):
    if derivative == 0:
        return points[:,0]**2
    elif derivative == 1:
        return np.hstack((2.*points[:,0:1], np.zeros(len(points))))
    elif derivative == 2:
        return np.hstack((2.*np.ones(len(points)), np.zeros(len(points))))

def y2(points, derivative=0):
    if derivative == 0:
        return points[:,1]**2
    elif derivative == 1:
        return np.hstack((np.zeros(len(points)), 2.*points[:,1:2]))
    elif derivative == 2:
        return np.hstack((np.zeros(len(points)), 2.*np.ones(len(points))))

def xy(points, derivative=0):
    if derivative == 0:
        return points[:,0] * points[:,1]
    elif derivative == 1:
        return np.hstack((points[:,1:2], points[:,0:1]))
    elif derivative == 2:
        return 2.*np.zeros((len(points), 2))

def x2y2(points, derivative=0):
    if derivative == 0:
        return points[:,0]**2 * points[:,1]**2
    elif derivative in [1, 2]:
        raise NotImplementedError("Derivatives not defined for given func.")

def x2py2(points, derivative=0):
    if derivative == 0:
        return points[:,0]**2 + points[:,1]**2
    elif derivative == 1:
        return np.hstack((2.*points[:,0:1], 2.*points[:,1:2]))
    elif derivative == 2:
        return 2.*np.ones((len(points), 2))

def sinx(points, derivative=0):
    if derivative == 0:
        return np.sin(np.pi*points[:,0])
    elif derivative == 1:
        return np.hstack((np.pi*np.cos(np.pi*points[:,0]), np.zeros(len(points))))
    elif derivative == 2:
        return np.hstack((-np.pi**2*np.sin(np.pi*points[:,0]), np.zeros(len(points))))

def sin2x(points, derivative=0):
    if derivative == 0:
        return np.sin(2*np.pi*points[:,0])
    elif derivative == 1:
        return np.hstack((2*np.pi*np.cos(2*np.pi*points[:,0]), np.zeros(len(points))))
    elif derivative == 2:
        return np.hstack((-4*np.pi**2*np.sin(2*np.pi*points[:,0]), np.zeros(len(points))))

def siny(points, derivative=0):
    if derivative == 0:
        return np.sin(np.pi*points[:,1])
    elif derivative == 1:
        return np.hstack((np.zeros(len(points)), np.pi*np.cos(np.pi*points[:,1])))
    elif derivative == 2:
        return np.hstack((np.zeros(len(points)), -np.pi**2*np.sin(np.pi*points[:,1])))

def sin2y(points, derivative=0):
    if derivative == 0:
        return np.sin(2*np.pi*points[:,1])
    elif derivative == 1:
        return np.hstack((np.zeros(len(points)), 2*np.pi*np.cos(2*np.pi*points[:,1])))
    elif derivative == 2:
        return np.hstack((np.zeros(len(points)), -4*np.pi**2*np.sin(2*np.pi*points[:,1])))

def sinxy(points, derivative=0):
    if derivative == 0:
        return np.sin(np.pi*(points[:,0]*points[:,1]))
    elif derivative in [1, 2]:
        raise NotImplementedError("Derivatives not defined for given func.")

def sinxpy(points, derivative=0):
    if derivative == 0:
        return np.sin(np.pi*(points[:,0]+points[:,1]))
    elif derivative in [1, 2]:
        raise NotImplementedError("Derivatives not defined for given func.")

def sinxsiny(points, derivative=0):
    if derivative == 0:
        return np.sin(np.pi*points[:,0])*np.sin(np.pi*points[:,1])
    elif derivative in [1, 2]:
        raise NotImplementedError("Derivatives not defined for given func.")

func = xy

kwargs={
    'Nquad' : 2,
    'support' : ('circular', 3),
    'form' : 'quartic',
    'method' : 'galerkin',
    'quadrature' : 'uniform',
    'basis' : 'quadratic'}

mls = PoissonMlsSim(8, g, **kwargs)

n = 64

points = ( np.indices((n+1, n+1), dtype='float64').T.reshape(-1,2) ) / n

phis = np.zeros((len(points), mls.nNodes), dtype='float64')
dphidx = np.zeros((len(points), mls.nNodes), dtype='float64')
dphidy = np.zeros((len(points), mls.nNodes), dtype='float64')
for i, point in enumerate(points):
    indices, local_phis = mls.phi(point)
    gradphi = mls.dphi(point)[2]
    phis[i, indices] += local_phis
    dphidx[i, indices] += gradphi[:,0]
    dphidy[i, indices] += gradphi[:,1]

A = np.zeros((mls.nNodes, mls.nNodes), dtype='float64')
for i, node in enumerate(mls.nodes):
    indices, local_phis = mls.phi(node)
    A[i, indices] += local_phis
    
u = la.solve(A, func(mls.nodes))

approximate_function = phis@u
exact_function = func(points)


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

exact_gradient = func(points, derivative=1)
approximate_dx = dphidx@u
approximate_dy = dphidy@u

# plot the error in dx
difference_dx = approximate_dx - exact_gradient[:,0]
ax = plt.subplot(121, projection='3d')
surf = ax.plot_trisurf(points[:,0], points[:,1], difference_dx,
                       cmap='seismic', linewidth=0, antialiased=False,
                       vmin=-np.max(np.abs(difference_dx)),
                       vmax=np.max(np.abs(difference_dx)))
plt.colorbar(surf, shrink=0.75, aspect=7)
# ax.zaxis.set_ticks([0, 0.001, 0.002, 0.003])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('dx Error')

# plot the error in dy
difference_dy = approximate_dy - exact_gradient[:,1]
ax = plt.subplot(122, projection='3d')
surf = ax.plot_trisurf(points[:,0], points[:,1], difference_dy,
                       cmap='seismic', linewidth=0, antialiased=False,
                       vmin=-np.max(np.abs(difference_dy)),
                       vmax=np.max(np.abs(difference_dy)))
plt.colorbar(surf, shrink=0.75, aspect=7)
# ax.zaxis.set_ticks([0, 0.001, 0.002, 0.003])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('dy Error')

# # plot the result
# # plt.subplot(221)
# # plt.tripcolor(points[:,0], points[:,1], approximate_function, shading='gouraud')
# # plt.colorbar()
# ax = plt.subplot(121, projection='3d')
# surf = ax.plot_trisurf(points[:,0], points[:,1], approximate_function,
#                        cmap='viridis', linewidth=0, antialiased=False
#                         # , vmin=0, vmax=2
#                        )
# # ax.zaxis.set_ticks([0,0.5,1,1.5,2])
# plt.colorbar(surf, shrink=0.75, aspect=7)
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$')
# plt.title('MLS Approximation')

# # # plot the result
# # # plt.subplot(222)
# # # plt.tripcolor(points[:,0], points[:,1], exact_function, shading='gouraud')
# # # plt.colorbar()
# # ax = plt.subplot(222, projection='3d')
# # surf = ax.plot_trisurf(points[:,0], points[:,1], exact_function,
# #                        cmap='viridis', linewidth=0, antialiased=False)
# # plt.colorbar(surf, shrink=0.75, aspect=7)
# # plt.xlabel(r'$x$')
# # plt.ylabel(r'$y$')
# # plt.title('Exact Function')

# # plot the error
# difference = approximate_function - exact_function
# # plt.subplot(223)
# # plt.tripcolor(points[:,0], points[:,1], difference, shading='gouraud')
# # plt.colorbar()
# ax = plt.subplot(122, projection='3d')
# surf = ax.plot_trisurf(points[:,0], points[:,1], difference,
#                        cmap='seismic', linewidth=0, antialiased=False,
#                        vmin=-np.max(np.abs(difference)),
#                        vmax=np.max(np.abs(difference)))
# # ax.axes.set_zlim3d(bottom=-np.max(np.abs(difference)),
# #                    top=np.max(np.abs(difference))) 
# plt.colorbar(surf, shrink=0.75, aspect=7)
# # ax.zaxis.set_ticks([0, 0.001, 0.002, 0.003])
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$')
# plt.title('Error')

# plt.savefig('MLS_xy.pdf', bbox_inches='tight', pad_inches=0)