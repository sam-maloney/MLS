#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:09:36 2020

@author: samal
"""

import numpy as np
# import scipy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt

from PoissonMlsSim import PoissonMlsSim
from ConvectionDiffusionMlsSim import ConvectionDiffusionMlsSim

def g(points):
    return np.zeros(len(points), dtype='float64')

def hat(points):
    return np.hstack((points > 0.25, points < 0.75)).all(1).astype('float64')

N = 3
dt = 0.01
velocity = np.array([0.1, 0.0], dtype='float64')
diffusivity = np.array([[0.01, 0.],[0., 0.01]], dtype='float64')
kwargs={
    'N' : N,
    'g' : g,
    'dt' : dt,
    'u0' : hat,
    'velocity' : velocity,
    'diffusivity' : diffusivity,
    'Nquad' : 1,
    'support' : -1,
    'form' : 'cubic',
    'method' : 'galerkin',
    'quadrature' : 'uniform',
    'perturbation' : 0 }

# mls = PoissonMlsSim(**kwargs)
mls = ConvectionDiffusionMlsSim(**kwargs)

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,4.5)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.25)
plt.rc('axes', axisbelow=True)

plt.subplot(131)
plt.scatter(mls.uNodes()[:,0], mls.uNodes()[:,1], s=100)
plt.scatter(mls.quads[:,0], mls.quads[:,1])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('1/cell')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.margins(0.0)
plt.xticks(np.arange(N+1)/N)
plt.yticks(np.arange(N+1)/N)
plt.grid()

kwargs['Nquad'] = 2
# mls = PoissonMlsSim(**kwargs)
mls = ConvectionDiffusionMlsSim(**kwargs)

plt.subplot(132)
plt.scatter(mls.nodes[:,0], mls.nodes[:,1], s=100)
plt.scatter(mls.quads[:,0], mls.quads[:,1])
plt.xlabel(r'$x$')
plt.title('4/cell, uniform spacing')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.margins(0.0)
plt.xticks(np.arange(N+1)/N)
plt.yticks(np.arange(N+1)/N)
plt.grid()

kwargs['Nquad'] = 2
kwargs['quadrature'] = 'gaussian'
# mls = PoissonMlsSim(**kwargs)
mls = ConvectionDiffusionMlsSim(**kwargs)

plt.subplot(133)
plt.scatter(mls.nodes[:,0], mls.nodes[:,1], s=100)
plt.scatter(mls.quads[:,0], mls.quads[:,1])
plt.xlabel(r'$x$')
plt.title('4/cell, Gaussian spacing')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.margins(0.0)
plt.xticks(np.arange(N+1)/N)
plt.yticks(np.arange(N+1)/N)
plt.grid()

# plt.savefig("MLS_points.pdf", bbox_inches='tight', pad_inches=0)