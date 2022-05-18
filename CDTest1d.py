#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Meshfree method simulation using moving least squares (MLS)

@author: Samuel A. Maloney
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as la

from ConvectionDiffusionMlsSim import ConvectionDiffusionMlsSim
from timeit import default_timer

import warnings
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

def gaussian(points):
    # integral for A=1 and sigma=0.1 is 0.25066268375731304228
    A = 1.0
    ndim = points.shape[1]
    r0 = (0.5, 0.5, 0.5)[0:ndim]
    sigma = (0.1, 0.1, 0.1)[0:ndim]
    return np.exp( -0.5*A*np.sum(((points - r0)/sigma )**2, 1) )

def hat(points):
    return np.hstack((points > 0.25, points < 0.75)).all(1).astype('float64')

def sinusoid(points):
    return np.sin(2.0*np.pi*np.sum(points, axis=1))

# N is the number of grid cells along one dimension,
# therefore the number of nodes equals N*N
# N = 200
# dt = 0.125
velocity = 1
diffusivity = 0.
D = diffusivity

print(
      # f'N = {N}\n'
      # f'dt = {dt}\n'
      f'velocity = {velocity}\n'
      f'diffusivity = {" ".join(repr(diffusivity).split())}')

kwargs={
    # 'N' : N,
    # 'dt' : dt,
    'u0' : sinusoid,
    'velocity' : velocity,
    'diffusivity' : diffusivity,
    'ndim' : 1,
    'Nquad' : 2,
    'support' : 1.5,
    'form' : 'quartic',
    'quadrature' : 'uniform',
    'basis' : 'linear'}

precon='ilu'
tolerance = 1e-10

# allocate arrays for convergence testing
start = 4
stop = 8
nSamples = stop - start + 1
dt_array = np.logspace(start, stop, num=nSamples, base=0.5)
N_array = np.logspace(start, stop, num=nSamples, base=2, dtype='uint32')
E_inf = np.empty(nSamples, dtype='float64')
E_2 = np.empty(nSamples, dtype='float64')

for i, dt in enumerate(dt_array):

    N = N_array[i]
    print(f'\nN  = {N}\ndt = {dt}')
    kwargs['dt'] = dt
    kwargs['N'] = N

    start_time = default_timer()

    # Initialize simulation
    mlsSim = ConvectionDiffusionMlsSim(**kwargs)
    mlsSim.computeSpatialDiscretization()
    mlsSim.precondition(precon)

    current_time = default_timer()
    print(f'Set-up time [s]     = {current_time-start_time}')
    # print('Condition Number =', mlsSim.cond('fro'))

    start_time = default_timer()

    if velocity != 0:
        mlsSim.step(int(1./dt/abs(velocity)), tol=tolerance, atol=tolerance)
    else:
        mlsSim.step(int(1./dt), tol=tolerance, atol=tolerance)

    current_time = default_timer()
    print(f'Simulation time [s] = {current_time-start_time}')

    mlsSim.solve()

    # compute the analytic solution and error norms
    u_exact = kwargs['u0'](mlsSim.uNodes())*np.exp(-D*4.0*np.pi**2*mlsSim.time)
    E_inf[i] = la.norm(mlsSim.u - u_exact, np.inf)
    # E_inf[i] = np.abs(np.sum(np.abs(mlsSim.u))/N
    #                    - np.sum(np.abs(kwargs['u0'](mlsSim.uNodes())))/N)
    #                    - 0.25066268375731304228)
    E_2[i] = la.norm(mlsSim.u - u_exact)/np.sqrt(N)
    print('max error =', E_inf[i])
    print('L2 error  =', E_2[i])

##### End of loop over dt #####



##### Begin Plotting Routines #####

# clear the current figure, if opened, and set parameters
fig = plt.gcf()
fig.clf()
fig.set_size_inches(7.75,3)
plt.subplots_adjust(hspace = 0.3, wspace = 0.35)

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

# plot the result
plt.subplot(121)
plt.plot(mlsSim.nodes, mlsSim.u[mlsSim.periodicIndices])
plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
plt.xlabel(r'$x$')
plt.ylabel(r'$u$', rotation=0)
# plt.title('Final MLS solution')
plt.margins(0,0)

# # plot error
# difference = mlsSim.u - u_exact
# plt.subplot(122)
# plt.plot(mlsSim.nodes, difference[mlsSim.periodicIndices])
# plt.xlim(0.0, 1.0)
# plt.ylim(-np.max(np.abs(difference)), np.max(np.abs(difference)))
# plt.xlabel(r'$x$')
# plt.ylabel(r'$\mathrm{Difference}$')
# # plt.title('Error')
# plt.margins(0,0)

# plot the error convergence
ax1 = plt.subplot(122)
plt.loglog(dt_array, E_inf, '.-', label=r'$E_\infty$ magnitude')
plt.loglog(dt_array, E_2, '.-', label=r'$E_2$ magnitude')
plt.minorticks_off()
dt_labels = [f'$2^{{-{i}}}$' for i in range(start, stop+1)]
plt.xticks(dt_array, dt_labels)
plt.xlabel(r'$dt$')
plt.ylabel(r'Magnitude of Error Norm')

# plot the intra-step order of convergence
ax2 = ax1.twinx()
logdt = np.log(dt_array)
logE_inf = np.log(E_inf)
logE_2 = np.log(E_2)
order_inf = (logE_inf[0:-1] - logE_inf[1:])/(logdt[0:-1] - logdt[1:])
order_2 = (logE_2[0:-1] - logE_2[1:])/(logdt[0:-1] - logdt[1:])
intraN = np.logspace(start+0.5, stop-0.5, num=nSamples-1, base=0.5)
plt.plot(intraN, order_inf, '.:', linewidth=1, label=r'$E_\infty$ order')
plt.plot(intraN, order_2, '.:', linewidth=1, label=r'$E_2$ order')
plt.plot(plt.xlim(), [2, 2], 'k:', linewidth=1, label='Expected')
# plt.ylim(1, 3)
# plt.yticks([1, 1.5, 2, 2.5, 3])
plt.ylim(0, 4)
plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0])
plt.ylabel(r'Intra-step Order of Convergence')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower right')
plt.margins(0,0)

# plt.savefig(f"MLS_{kwargs['basis']}_{kwargs['form']}_{kwargs['Nquad']}Q_{kwargs['support']}S.pdf",
#     bbox_inches = 'tight', pad_inches = 0)