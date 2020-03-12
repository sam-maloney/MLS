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

from PoissonMlsSim import PoissonMlsSim
from timeit import default_timer

import warnings
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

# wavenumber for boundary function u(x,1) = g(x,y) = sinh(k*pi)
k = 1
def g(points):
    k = 1
    return np.sin(k*np.pi*points[:,0]) * np.sinh(k*np.pi*points[:,1])
            
kwargs={
    'Nquad' : 2,
    'support' : -1,
    'form' : 'cubic',
    'method' : 'collocation',
    'quadrature' : 'gaussian' }

precon='ilu'
tolerance = 1e-10

# allocate arrays for convergence testing
start = 1
stop = 5
nSamples = stop - start + 1
N_array = np.logspace(start, stop, num=nSamples, base=2, dtype='int32')
E_inf = np.empty(nSamples, dtype='float64')
E_2 = np.empty(nSamples, dtype='float64')

start_time = default_timer()

# loop over N to test convergence where N is the number of
# grid cells along one dimension, each cell forms 2 triangles
# therefore number of nodes equals (N+1)*(N+1)
for iN, N in enumerate(N_array):

    print('N =', N)
    
    # allocate arrays and compute boundary values
    mlsSim = PoissonMlsSim(N, g, **kwargs)
    
    # Assemble the stiffness matrix and solve for the approximate solution
    mlsSim.assembleStiffnessMatrix()
    mlsSim.solve(preconditioner=precon, tol=tolerance, atol=tolerance)
    
    # compute the analytic solution and error norms
    u_exact = g(mlsSim.nodes)
    E_inf[iN] = np.linalg.norm(mlsSim.u - u_exact, np.inf)
    E_2[iN] = np.linalg.norm(mlsSim.u - u_exact)/N
    
    end_time = default_timer()
    
    print('Condition Number =', mlsSim.cond('fro'))
    
    print('max error =', E_inf[iN])
    print('L2 error  =', E_2[iN])
    print(f'Elapsed time = {end_time-start_time} s')
    
##### End of loop over N #####

print(f'min(E_2) = {np.min(E_2)}')
print(f'min(E_inf) = {np.min(E_inf)}')

    
    
##### Begin Plotting Routines #####

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(15,15)
mpl.rc('axes', titlesize='xx-large', labelsize='x-large')
mpl.rc('xtick', labelsize='large')
mpl.rc('ytick', labelsize='large')
plt.subplots_adjust(hspace = 0.3, wspace = 0.25)

# plot the result
plt.subplot(221)
plt.tripcolor(mlsSim.nodes[:,0], mlsSim.nodes[:,1], mlsSim.u, shading='gouraud')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Final MLS solution')
plt.margins(0,0)

# plot analytic solution
plt.subplot(222)
plt.tripcolor(mlsSim.nodes[:,0], mlsSim.nodes[:,1], u_exact, shading='gouraud')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Analytic solution')
plt.margins(0,0)

# plot the error convergence
plt.subplot(223)
plt.loglog(N_array, E_inf, '.-', label=r'$E_\infty$')
plt.loglog(N_array, E_2, '.-', label=r'$E_2$')
plt.minorticks_off()
plt.xticks(N_array, N_array)
plt.xlabel(r'$N$')
plt.ylabel(r'Magnitude of Error Norm')
plt.title('MLS Error Norms')
plt.legend(fontsize='x-large')

# plot the intra-step order of convergence
plt.subplot(224)
logN = np.log(N_array)
logE_inf = np.log(E_inf)
logE_2 = np.log(E_2)
order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logN[1:] - logN[0:-1])
order_2 = (logE_2[0:-1] - logE_2[1:])/(logN[1:] - logN[0:-1])
intraN = np.logspace(start+0.5, stop-0.5, num=nSamples-1, base=2.0)
plt.plot(intraN, order_inf, '.-', label=r'$E_\infty$')
plt.plot(intraN, order_2, '.-', label=r'$E_2$')
plt.plot([N_array[0], N_array[-1]], [2, 2], 'k:', label='Expected')
plt.xlim(N_array[0], N_array[-1])
plt.xscale('log')
plt.minorticks_off()
plt.xticks(N_array, N_array)
# plt.ylim(1, 3)
# plt.yticks([1, 1.5, 2, 2.5, 3])
plt.ylim(0, 3)
plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
plt.xlabel(r'$N$')
plt.ylabel(r'Intra-step Order of Convergence')
plt.title('MLS Order of Accuracy')
plt.legend(fontsize='x-large')
plt.margins(0,0)

# plt.savefig(f"MLS_{method}_{form}_{k}k_{Nquad}Q_{mlsSim.support*mlsSim.N}S.pdf",
#     bbox_inches = 'tight', pad_inches = 0)